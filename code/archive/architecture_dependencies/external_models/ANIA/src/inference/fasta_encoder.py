# pylint: disable=too-many-locals, too-many-branches
"""
FASTA Encoding Module for ANIA Inference

This module provides functionality for converting user-provided FASTA sequences
into Chaos Game Representation (CGR) and physicochemical property feature matrices
for downstream ANIA deep learning inference.
"""

# ============================== Standard Library Imports ==============================
import argparse
import os

# ============================== Third-Party Library Imports ==============================
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

# ============================== Constants ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
AAINDEX_CSV = os.path.join(BASE_PATH, "configs", "AAindex_properties.csv")
DEFAULT_PROPERTY_IDS = [
    "ARGP820101",
    "CHAM830107",
    "FAUJ880103",
    "GRAR740102",
    "JANJ780101",
    "KYTJ820101",
    "NAKH920104",
    "ROSM880102",
    "WERD780104",
    "ZIMJ680101",
]
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


# ============================== Custom Function ==============================
def is_valid_natural_sequence(sequence: str) -> bool:
    """
    Check whether a sequence contains only canonical natural amino acids.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    bool
        True if sequence contains only 20 natural amino acids.
    """
    return all(residue in VALID_AMINO_ACIDS for residue in sequence.upper())


def load_fasta_sequences(
    fasta_path: str,
) -> tuple[list[str], list[str]]:
    """
    Load sequence IDs and amino acid sequences from a FASTA file.

    This function performs:
    1. FASTA format validation
    2. Sequence normalization (uppercase + whitespace removal)
    3. Removal of sequences containing non-canonical amino acids

    Only sequences composed exclusively of the 20 natural amino acids
    (ACDEFGHIKLMNPQRSTVWY) are retained.

    Parameters
    ----------
    fasta_path : str
        Path to input FASTA file.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing:
        - Sequence IDs
        - Valid amino acid sequences

    Raises
    ------
    FileNotFoundError
        If the FASTA file does not exist.
    ValueError
        If FASTA formatting is invalid or no valid sequences remain.
    """
    # Validate File Existence
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: '{fasta_path}'.")
    if os.path.getsize(fasta_path) == 0:
        raise ValueError(f"FASTA file is empty: '{fasta_path}'.")

    # Initialize Storage
    seq_ids = []
    sequences = []
    current_id = None
    current_sequence = ""
    filtered_count = 0

    # Parse FASTA File
    with open(fasta_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Header Line
            if line.startswith(">"):
                # Save previous sequence if valid
                if current_id is not None:
                    normalized_sequence = (
                        current_sequence.replace(" ", "").replace("\t", "").upper()
                    )

                    if normalized_sequence:
                        if is_valid_natural_sequence(normalized_sequence):
                            seq_ids.append(current_id)
                            sequences.append(normalized_sequence)
                        else:
                            filtered_count += 1

                # Start new sequence
                current_id = line[1:].strip()
                current_sequence = ""

                # Validate header
                if not current_id:
                    raise ValueError("FASTA header cannot be empty.")

            # Sequence Line
            else:
                if current_id is None:
                    raise ValueError(
                        "Invalid FASTA format: sequence appears before any header."
                    )

                current_sequence += line

        # Save Final Sequence
        if current_id is not None:
            normalized_sequence = (
                current_sequence.replace(" ", "").replace("\t", "").upper()
            )

            if normalized_sequence:
                if is_valid_natural_sequence(normalized_sequence):
                    seq_ids.append(current_id)
                    sequences.append(normalized_sequence)
                else:
                    filtered_count += 1

    # Final Validation
    if len(seq_ids) != len(sequences):
        raise ValueError("Mismatch between FASTA headers and sequences.")

    if not seq_ids:
        raise ValueError("No valid natural amino acid sequences found in FASTA file.")

    return seq_ids, sequences


def compute_cgr(
    sequences: list[str],
    resolution: int,
) -> tuple[list[np.ndarray], list]:
    """
    Compute CGR matrices for amino acid sequences.

    Parameters
    ----------
    sequences : list[str]
        Input amino acid sequences.
    resolution : int
        CGR grid resolution.

    Returns
    -------
    tuple[list[np.ndarray], list]
        A tuple containing:
        - Flattened CGR vectors
        - Raw CGR R objects
    """
    # Import kaos package
    kaos = importr("kaos")

    cgr_vectors = []
    cgr_results = []

    # Compute CGR for each sequence
    for sequence in sequences:
        r_sequence = StrVector(list(sequence))
        cgr_result = kaos.cgr(
            r_sequence,
            seq_base="AMINO",
            res=resolution,
        )

        cgr_matrix = np.array(ro.r("matrix")(cgr_result.rx2("matrix")))

        cgr_vectors.append(cgr_matrix.flatten(order="F"))
        cgr_results.append(cgr_result)

    return cgr_vectors, cgr_results


def map_kmers_to_pixels(
    sequences: list[str],
    cgr_results: list,
    resolution: int,
    k: int = 3,
) -> list[dict]:
    """
    Map sequence k-mers to CGR pixel coordinates.

    Parameters
    ----------
    sequences : list[str]
        Input sequences.
    cgr_results : list
        Raw CGR objects from kaos.
    resolution : int
        CGR grid resolution.
    k : int, optional
        K-mer size.

    Returns
    -------
    list[dict]
        Pixel-to-kmer mappings for each sequence.
    """

    def coords_to_pixel(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        res: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert CGR coordinates into pixel indices.
        """
        x_pix = np.ceil((x_coords + 1) * res / 2).astype(int) - 1
        y_pix = np.ceil((y_coords + 1) * res / 2).astype(int) - 1

        return y_pix, x_pix

    all_pixel_maps = []

    # Process each sequence
    for sequence, cgr_result in zip(sequences, cgr_results):
        x_coords = np.array(cgr_result.rx2("x"))
        y_coords = np.array(cgr_result.rx2("y"))

        row_indices, col_indices = coords_to_pixel(
            x_coords,
            y_coords,
            resolution,
        )

        pixel_map = {}

        # Map k-mers
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i : i + k]
            pixel = (
                row_indices[i + k - 1],
                col_indices[i + k - 1],
            )

            pixel_map.setdefault(pixel, []).append(kmer)

        all_pixel_maps.append(pixel_map)

    return all_pixel_maps


def compute_property_maps(
    kmer_maps: list[dict],
    property_table: pd.DataFrame,
    property_id: str,
) -> list[dict]:
    """
    Compute property-based CGR pixel values from AAindex properties.

    Parameters
    ----------
    kmer_maps : list[dict]
        Pixel-to-kmer mappings.
    property_table : pd.DataFrame
        AAindex property lookup table.
    property_id : str
        Selected AAindex property ID.

    Returns
    -------
    list[dict]
        Pixel-to-property mappings.
    """
    property_maps = []

    # Compute property values for each sequence
    for pixel_kmers in kmer_maps:
        pixel_values = {}

        for pixel, kmers in pixel_kmers.items():
            unique_kmers = set(kmers)

            values = []

            for kmer in unique_kmers:
                try:
                    values.append(
                        sum(property_table.loc[aa, property_id] for aa in kmer)
                    )
                except KeyError:
                    continue

            pixel_values[pixel] = np.mean(values) if values else 0.0

        property_maps.append(pixel_values)

    return property_maps


# ============================== Pipeline Entry Point ==============================
def encode_fasta_to_cgr(
    fasta_path: str,
    output_csv_path: str,
    resolution: int = 16,
) -> None:
    """
    Encode FASTA sequences into ANIA-compatible CGR and property features.

    Parameters
    ----------
    fasta_path : str
        Path to input FASTA file.
    output_csv_path : str
        Path to save encoded CSV.
    resolution : int, optional
        CGR grid resolution.

    Returns
    -------
    None
    """
    # Load FASTA sequences
    seq_ids, sequences = load_fasta_sequences(fasta_path)

    # Build metadata DataFrame
    metadata_df = pd.DataFrame(
        {
            "ID": seq_ids,
            "Sequence": sequences,
            "Sequence Length": [len(seq) for seq in sequences],
        }
    )

    # Compute CGR features
    cgr_vectors, cgr_results = compute_cgr(
        sequences=sequences,
        resolution=resolution,
    )

    # Build CGR DataFrame
    cgr_df = pd.DataFrame(
        cgr_vectors,
        columns=[
            f"CGR(resolution={resolution}) | {i+1}-{j+1}"
            for i in range(resolution)
            for j in range(resolution)
        ],
    )

    cgr_df.insert(0, "ID", seq_ids)

    # Map k-mers
    pixel_kmer_maps = map_kmers_to_pixels(
        sequences=sequences,
        cgr_results=cgr_results,
        resolution=resolution,
    )

    # Load property table
    property_table = pd.read_csv(
        AAINDEX_CSV,
        index_col="AminoAcid",
    )

    # Compute property features
    property_feature_frames = []

    for property_id in DEFAULT_PROPERTY_IDS:
        property_maps = compute_property_maps(
            kmer_maps=pixel_kmer_maps,
            property_table=property_table,
            property_id=property_id,
        )

        property_vectors = []

        for property_map in property_maps:
            matrix = np.zeros((resolution, resolution))

            for (row, col), value in property_map.items():
                matrix[row, col] = value

            property_vectors.append(matrix.flatten(order="C"))

        property_feature_frames.append(
            pd.DataFrame(
                property_vectors,
                columns=[
                    f"PROP({property_id}) | {i+1}-{j+1}"
                    for i in range(resolution)
                    for j in range(resolution)
                ],
            )
        )

    # Merge all property features
    for property_df in property_feature_frames:
        cgr_df = pd.concat(
            [cgr_df, property_df],
            axis=1,
        )

    # Merge metadata and features
    final_df = metadata_df.merge(
        cgr_df,
        on="ID",
        how="left",
    )

    # Create output directory
    os.makedirs(
        os.path.dirname(output_csv_path),
        exist_ok=True,
    )

    # Save output CSV
    final_df.to_csv(
        output_csv_path,
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode FASTA sequences into CGR features for ANIA inference."
    )

    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to input FASTA file.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output CSV file.",
    )

    parser.add_argument(
        "--res",
        type=int,
        default=16,
        help="CGR resolution (default: 16).",
    )

    arguments = parser.parse_args()

    encode_fasta_to_cgr(
        fasta_path=arguments.fasta,
        output_csv_path=arguments.out,
        resolution=arguments.res,
    )
