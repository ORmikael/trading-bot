# Coordinate the complete source → normalized-data pipeline.
"""
Market Data Ingestion Layer

Responsible for reading extracted market-data CSV files.

Expected extracted format:

    EUR/USD,20260101 18:02:25.204,1.17286,1.17567

Columns:

    symbol
    timestamp
    bid
    ask

This module does NOT:
    - extract archives
    - modify source files
    - generate bars
    - perform strategy calculations
    - write Parquet files
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


# ============================================================
# SCHEMA
# ============================================================

COLUMNS = [
    "symbol",
    "timestamp",
    "bid",
    "ask",
]


# ============================================================
# READER
# ============================================================


def read_csv(
    path: Path,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """
    Read an extracted TrueFX CSV file.

    Parameters
    ----------
    path:
        Path to extracted CSV.

    chunksize:
        Optional number of rows per chunk.

        If None:
            returns one DataFrame.

        If provided:
            returns an iterator of DataFrames.

    Returns
    -------
    pandas.DataFrame or iterator[pandas.DataFrame]
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {path}"
        )

    return pd.read_csv(
        path,
        encoding="ascii",
        header=None,
        names=COLUMNS,
        chunksize=chunksize,
    )


# ============================================================
# DIRECTORY DISCOVERY
# ============================================================


def find_extracted_files(
    extracted_dir: Path,
) -> list[Path]:
    """
    Find extracted CSV datasets.

    Files are returned in deterministic order.
    """

    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"Extracted directory does not exist: "
            f"{extracted_dir}"
        )

    files = sorted(
        extracted_dir.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV datasets found in: "
            f"{extracted_dir}"
        )

    return files


# ============================================================
# DATASET ITERATOR
# ============================================================


def iter_datasets(
    extracted_dir: Path,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """
    Iterate through all extracted datasets.

    Yields:

        (
            source_file,
            dataframe
        )
    """

    for path in find_extracted_files(
        extracted_dir
    ):

        df = read_csv(path)

        yield path, df