# pylint: disable=import-error, wrong-import-position, too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-statements
"""
Deep Learning Inference Module for ANIA

This module provides functionality for running local ANIA inference
on user-provided FASTA sequences through CGR encoding and trained ANIA checkpoints.
It supports structured logging, model checkpoint resolution, and prediction result generation.
"""

# ============================== Standard Library Imports ==============================
import logging
import os
import shutil
import subprocess
import sys
import uuid
from typing import Optional

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger

# ============================== Constants ==============================
SUPPORTED_STRAINS = {
    "Escherichia coli": "EC",
    "Pseudomonas aeruginosa": "PA",
    "Staphylococcus aureus": "SA",
}


# ============================== Custom Function ==============================
def validate_fasta_input(input_fasta: str) -> None:
    """
    Validate whether the input FASTA file exists and contains at least one valid sequence.

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the input FASTA file does not exist.
    ValueError
        If the FASTA file is empty or contains invalid FASTA format.
    """
    # Check file existence
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input FASTA file does not exist: '{input_fasta}'.")

    # Check file size
    if os.path.getsize(input_fasta) == 0:
        raise ValueError(f"Input FASTA file is empty: '{input_fasta}'.")

    # Initialize sequence validation variables
    sequence_count = 0
    current_header = None
    current_sequence = ""

    # Parse FASTA file
    with open(input_fasta, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Process header lines
            if line.startswith(">"):
                if current_header is not None and current_sequence:
                    sequence_count += 1

                current_header = line[1:].strip()
                current_sequence = ""

                if not current_header:
                    raise ValueError("FASTA header cannot be empty.")

            # Process sequence lines
            else:
                if current_header is None:
                    raise ValueError(
                        "Invalid FASTA format: sequence appears before any header."
                    )

                current_sequence += line

        # Count final sequence
        if current_header is not None and current_sequence:
            sequence_count += 1

    # Validate sequence count
    if sequence_count == 0:
        raise ValueError(f"No valid FASTA sequences found in '{input_fasta}'.")


def resolve_model_checkpoint(
    base_path: str,
    strain: str,
    model_input_path: Optional[str] = None,
) -> str:
    """
    Resolve the ANIA model checkpoint path.

    Parameters
    ----------
    base_path : str
        Absolute project root path.
    strain : str
        Full strain name.
    model_input_path : str, optional
        User-specified checkpoint path.

    Returns
    -------
    str
        Absolute path to the ANIA checkpoint file.

    Raises
    ------
    ValueError
        If the strain is unsupported.
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    # Validate strain
    if strain not in SUPPORTED_STRAINS:
        raise ValueError(
            f"Invalid strain: '{strain}'. Available strains: {list(SUPPORTED_STRAINS.keys())}"
        )

    # Resolve checkpoint path
    if model_input_path is not None:
        checkpoint_path = os.path.abspath(model_input_path)
    else:
        suffix = SUPPORTED_STRAINS[strain]
        checkpoint_path = os.path.join(base_path, "weights", f"ANIA_{suffix}.pt")

    # Validate checkpoint existence
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint does not exist: '{checkpoint_path}'."
        )

    return checkpoint_path


def run_local_cgr_encoding(
    script_path: str,
    input_fasta: str,
    output_cgr_csv: str,
    logger: CustomLogger,
) -> None:
    """
    Run CGR encoding through an external shell script.

    Parameters
    ----------
    script_path : str
        Path to the CGR encoding shell script.
    input_fasta : str
        Path to the input FASTA file.
    output_cgr_csv : str
        Path to save CGR-encoded CSV.
    logger : CustomLogger
        Logger instance for structured logging.

    Returns
    -------
    None
    """
    try:
        # Validate script existence
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"CGR encoding script not found: '{script_path}'.")

        # Create output directory
        os.makedirs(os.path.dirname(output_cgr_csv), exist_ok=True)

        # Log CGR encoding configuration
        logger.info(msg="/ Task: Run local CGR encoding")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"CGR encoding command:\n"
                f"  • Script: '{script_path}'\n"
                f"  • Input FASTA: '{input_fasta}'\n"
                f"  • Output CSV: '{output_cgr_csv}'"
            ),
            border="|",
            length=100,
        )

        # Run CGR encoding script
        subprocess.run(
            ["bash", script_path, input_fasta, output_cgr_csv],
            check=True,
        )

        # Validate output generation
        if not os.path.exists(output_cgr_csv):
            raise FileNotFoundError(
                f"CGR output CSV was not created: '{output_cgr_csv}'."
            )

        # Log successful completion
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully generated CGR-encoded CSV:\n'{output_cgr_csv}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except subprocess.CalledProcessError as exc:
        logger.exception(msg="CGR encoding shell script failed.")
        raise RuntimeError("CGR encoding failed.") from exc

    except FileNotFoundError:
        logger.exception(msg="FileNotFoundError in 'run_local_cgr_encoding()'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'run_local_cgr_encoding()'.")
        raise


def run_local_ania_prediction(
    script_path: str,
    input_cgr_csv: str,
    model_checkpoint: str,
    prediction_output_path: str,
    logger: CustomLogger,
) -> None:
    """
    Run ANIA prediction through an external shell script.

    Parameters
    ----------
    script_path : str
        Path to the ANIA prediction shell script.
    input_cgr_csv : str
        Path to the CGR-encoded CSV file.
    model_checkpoint : str
        Path to the trained ANIA checkpoint.
    prediction_output_path : str
        Path to save prediction results.
    logger : CustomLogger
        Logger instance for structured logging.

    Returns
    -------
    None
    """
    try:
        # Validate script existence
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"ANIA prediction script not found: '{script_path}'."
            )

        # Create output directory
        os.makedirs(os.path.dirname(prediction_output_path), exist_ok=True)

        # Log prediction configuration
        logger.info(msg="/ Task: Run local ANIA prediction")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"ANIA prediction command:\n"
                f"  • Script: '{script_path}'\n"
                f"  • Input CGR CSV: '{input_cgr_csv}'\n"
                f"  • Model checkpoint: '{model_checkpoint}'\n"
                f"  • Output CSV: '{prediction_output_path}'"
            ),
            border="|",
            length=100,
        )

        # Run prediction script
        subprocess.run(
            [
                "bash",
                script_path,
                input_cgr_csv,
                model_checkpoint,
                prediction_output_path,
            ],
            check=True,
        )

        # Validate output generation
        if not os.path.exists(prediction_output_path):
            raise FileNotFoundError(
                f"Prediction output CSV was not created: '{prediction_output_path}'."
            )

        # Log successful completion
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully saved ANIA prediction results:\n'{prediction_output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except subprocess.CalledProcessError as exc:
        logger.exception(msg="ANIA prediction shell script failed.")
        raise RuntimeError("ANIA prediction failed.") from exc

    except FileNotFoundError:
        logger.exception(msg="FileNotFoundError in 'run_local_ania_prediction()'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'run_local_ania_prediction()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_infer_ania_pipeline(
    base_path: str,
    strain: str,
    logger: CustomLogger,
    input_fasta: str,
    model_input_path: Optional[str] = None,
    prediction_output_path: Optional[str] = None,
    cgr_output_path: Optional[str] = None,
    encode_script_path: Optional[str] = None,
    predict_script_path: Optional[str] = None,
    keep_intermediate: bool = False,
) -> None:
    """
    Run the deep learning inference pipeline for ANIA on user-provided FASTA input.

    Parameters
    ----------
    base_path : str
        Absolute project root path.
    strain : str
        Full strain name.
    logger : CustomLogger
        Logger instance for structured logging.
    input_fasta : str
        Path to input FASTA file.
    model_input_path : str, optional
        Path to trained ANIA checkpoint.
    prediction_output_path : str, optional
        Path to save prediction CSV.
    cgr_output_path : str, optional
        Path to save intermediate CGR CSV.
    encode_script_path : str, optional
        Path to CGR encoding shell script.
    predict_script_path : str, optional
        Path to ANIA prediction shell script.
    keep_intermediate : bool, optional
        Whether to retain intermediate CGR CSV.

    Returns
    -------
    None
    """
    try:
        # Log the strain being inferenced
        logger.add_divider(level=logging.INFO, length=80, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Strain - '{strain}' -> Model - 'ANIA'",
            border="|",
            length=80,
        )
        logger.add_divider(level=logging.INFO, length=80, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)

        # Validate input FASTA
        input_fasta = os.path.abspath(input_fasta)
        validate_fasta_input(input_fasta)

        # Validate strain
        if strain not in SUPPORTED_STRAINS:
            raise ValueError(
                f"Invalid strain: '{strain}'. Available strains: {list(SUPPORTED_STRAINS.keys())}"
            )

        # Resolve strain suffix
        suffix = SUPPORTED_STRAINS[strain]

        # Resolve model checkpoint
        model_checkpoint = resolve_model_checkpoint(
            base_path=base_path,
            strain=strain,
            model_input_path=model_input_path,
        )

        # Define script paths
        if encode_script_path is None:
            encode_script_path = os.path.join(
                base_path,
                "scripts",
                "run_fasta_encoder.sh",
            )
        else:
            encode_script_path = os.path.abspath(encode_script_path)

        if predict_script_path is None:
            predict_script_path = os.path.join(
                base_path,
                "scripts",
                "run_ania_predictor.sh",
            )
        else:
            predict_script_path = os.path.abspath(predict_script_path)

        # Generate job identifier
        input_stem = os.path.splitext(os.path.basename(input_fasta))[0]
        job_id = f"{input_stem}_{suffix}_{uuid.uuid4().hex[:8]}"

        # Define prediction output path
        if prediction_output_path is None:
            prediction_output_path = os.path.join(
                base_path,
                "results",
                f"{job_id}_predict.csv",
            )
        else:
            prediction_output_path = os.path.abspath(prediction_output_path)

        # Define CGR intermediate path
        if cgr_output_path is None:
            cgr_output_path = os.path.join(
                base_path,
                "results",
                "tmp",
                f"{job_id}_cgr.csv",
            )
        else:
            cgr_output_path = os.path.abspath(cgr_output_path)

        # Run CGR encoding
        run_local_cgr_encoding(
            script_path=encode_script_path,
            input_fasta=input_fasta,
            output_cgr_csv=cgr_output_path,
            logger=logger,
        )
        logger.add_spacer(level=logging.INFO, lines=1)

        # Run ANIA prediction
        run_local_ania_prediction(
            script_path=predict_script_path,
            input_cgr_csv=cgr_output_path,
            model_checkpoint=model_checkpoint,
            prediction_output_path=prediction_output_path,
            logger=logger,
        )

        # Remove intermediate files if requested
        if not keep_intermediate:
            if os.path.exists(cgr_output_path):
                os.remove(cgr_output_path)

            cgr_dir = os.path.dirname(cgr_output_path)
            if os.path.exists(cgr_dir) and not os.listdir(cgr_dir):
                shutil.rmtree(cgr_dir)

            logger.log_with_borders(
                level=logging.INFO,
                message="Intermediate CGR files were removed.",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log successful completion
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Inference completed for 'ANIA':\n'{prediction_output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)

    except ValueError:
        logger.exception(msg="ValueError in 'run_infer_ania_pipeline()'.")
        raise

    except FileNotFoundError:
        logger.exception(msg="FileNotFoundError in 'run_infer_ania_pipeline()'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'run_infer_ania_pipeline()'.")
        raise
