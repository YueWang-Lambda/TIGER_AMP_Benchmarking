# pylint: disable=import-error, wrong-import-position, too-many-arguments, too-many-positional-arguments, invalid-name
"""
ANIA Prediction Module for Local Inference

This module provides functionality for loading CGR-encoded feature CSV files,
reconstructing trained ANIA models from checkpoints, and generating MIC predictions
for user-provided AMP sequences.
"""

# ============================== Standard Library Imports ==============================
import argparse
import os
import sys
from multiprocessing import Lock

# ============================== Third-Party Library Imports ==============================
import pandas as pd
import torch

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)


# ============================== Project-Specific Imports ==============================
from src.models.deep_learning.ANIA import ANIA

# ============================== Global Lock ==============================
_inference_lock = Lock()


# ============================== Custom Function ==============================
def extract_cgr_features_for_inference(
    file_path: str,
    metadata_columns: list[str],
    feature_start_idx: int,
    feature_end_idx: int,
    height: int,
    width: int,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Extract metadata and CGR features from encoded CSV for ANIA inference.

    Parameters
    ----------
    file_path : str
        Path to CGR-encoded CSV file.
    metadata_columns : list[str]
        Metadata columns to preserve.
    feature_start_idx : int
        Start index of feature columns.
    feature_end_idx : int
        End index of feature columns.
    height : int
        Feature matrix height.
    width : int
        Feature matrix width.

    Returns
    -------
    tuple[pd.DataFrame, torch.Tensor]
        Metadata DataFrame and reshaped feature tensor.
    """
    # Load CSV
    df = pd.read_csv(file_path)

    # Validate file
    if df.empty:
        raise ValueError(f"'{file_path}' is empty.")

    # Extract metadata
    df_metadata = df[metadata_columns]

    # Extract features
    feature_columns = df.columns[feature_start_idx : feature_end_idx + 1]

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Validate dimensions
    expected_feature_dim = height * width * 11

    if X.shape[1] != expected_feature_dim:
        raise ValueError(
            f"Expected {expected_feature_dim} features "
            f"(11 x {height} x {width}), but got {X.shape[1]}."
        )

    # Convert to tensor
    X_tensor = torch.tensor(
        X.values,
        dtype=torch.float32,
    )

    X_tensor = X_tensor.view(
        -1,
        11,
        height,
        width,
    )

    return df_metadata, X_tensor


def load_trained_ania_model(
    checkpoint_path: str,
    device: str,
) -> ANIA:
    """
    Load trained ANIA model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file.
    device : str
        Device for model loading.

    Returns
    -------
    ANIA
        Loaded ANIA model.
    """
    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    hyperparams = checkpoint["hyperparams"]

    # Dynamically compute output channels
    inception1_out_channels = (
        hyperparams["inception1_branch1x1_channels"]
        + hyperparams["inception1_branch3x3_channels"]
        + hyperparams["inception1_branch5x5_channels"]
        + hyperparams["inception1_branch_pool_channels"]
    )

    inception2_out_channels = (
        hyperparams["inception2_branch1x1_channels"]
        + hyperparams["inception2_branch3x3_channels"]
        + hyperparams["inception2_branch5x5_channels"]
        + hyperparams["inception2_branch_pool_channels"]
    )

    # Initialize model
    model = ANIA(
        in_channels=11,
        inception1_out_channels=inception1_out_channels,
        inception2_out_channels=inception2_out_channels,
        **{
            key: value
            for key, value in hyperparams.items()
            if key in ANIA.__init__.__code__.co_varnames
        },
    )

    # Load weights
    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()

    return model


# ============================== Pipeline Entry Point ==============================
def predict_ania_from_csv(
    input_csv_path: str,
    model_ckpt_path: str,
    output_csv_path: str,
    resolution: int = 16,
    start_idx: int = 3,
    end_idx: int = 2818,
    device: str = "cuda:0",
) -> None:
    """
    Run ANIA inference on CGR-encoded CSV and save predictions.

    Parameters
    ----------
    input_csv_path : str
        Path to encoded input CSV.
    model_ckpt_path : str
        Path to trained ANIA checkpoint.
    output_csv_path : str
        Path to prediction output CSV.
    resolution : int, optional
        CGR resolution.
    start_idx : int, optional
        Feature start index.
    end_idx : int, optional
        Feature end index.
    device : str, optional
        Inference device.

    Returns
    -------
    None
    """
    with _inference_lock:
        # Extract features
        metadata_columns = [
            "ID",
            "Sequence",
            "Sequence Length",
        ]

        df_metadata, X_tensor = extract_cgr_features_for_inference(
            file_path=input_csv_path,
            metadata_columns=metadata_columns,
            feature_start_idx=start_idx,
            feature_end_idx=end_idx,
            height=resolution,
            width=resolution,
        )

        # Load model
        model = load_trained_ania_model(
            checkpoint_path=model_ckpt_path,
            device=device,
        )

        # Predict
        X_tensor = X_tensor.to(device)

        with torch.no_grad():
            predictions = model(X_tensor).cpu().squeeze().tolist()

        # Normalize single prediction
        if isinstance(predictions, float):
            predictions = [predictions]

        # Build result DataFrame
        df_predictions = df_metadata.copy()

        df_predictions["Predicted Log MIC Value"] = [
            round(pred, 4) for pred in predictions
        ]

        # Save output
        os.makedirs(
            os.path.dirname(output_csv_path),
            exist_ok=True,
        )

        df_predictions.to_csv(
            output_csv_path,
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ANIA prediction from CGR-encoded CSV."
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    args = parser.parse_args()

    predict_ania_from_csv(
        input_csv_path=args.csv,
        model_ckpt_path=args.ckpt,
        output_csv_path=args.output_path,
        device=args.device,
    )
