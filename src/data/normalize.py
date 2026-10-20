# Convert valid source data into our canonical format.
"""
Market Data Normalization Layer

Converts extracted source data into the canonical internal
tick-data representation used by the rest of the system.

Input:

    data/extracted/EURUSD/TrueFX/*.csv

Output:

    data/normalized/EURUSD/ticks/*.parquet

Canonical schema:

    symbol
    timestamp
    bid
    ask
    spread

This module does NOT:
    - extract archives
    - validate source integrity
    - generate OHLC bars
    - run strategies
    - run backtests
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.ingest import read_csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTRACTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "extracted"
)

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "normalized"
)


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a source dataframe into the canonical
    tick-data schema.
    """

    required_columns = {
        "symbol",
        "timestamp",
        "bid",
        "ask",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    result = df[
        [
            "symbol",
            "timestamp",
            "bid",
            "ask",
        ]
    ].copy()

    # --------------------------------------------------------
    # Symbol
    # --------------------------------------------------------

    result["symbol"] = (
        result["symbol"]
        .astype(str)
        .str.replace(
            "/",
            "",
            regex=False,
        )
        .str.upper()
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        format="%Y%m%d %H:%M:%S.%f",
        errors="raise",
    )

    # --------------------------------------------------------
    # Prices
    # --------------------------------------------------------

    result["bid"] = pd.to_numeric(
        result["bid"],
        errors="raise",
    )

    result["ask"] = pd.to_numeric(
        result["ask"],
        errors="raise",
    )

    # --------------------------------------------------------
    # Spread
    # --------------------------------------------------------

    result["spread"] = (
        result["ask"]
        - result["bid"]
    )

    # --------------------------------------------------------
    # Canonical ordering
    # --------------------------------------------------------

    result = result[
        [
            "symbol",
            "timestamp",
            "bid",
            "ask",
            "spread",
        ]
    ]

    return result


# ============================================================
# SINGLE FILE
# ============================================================


def normalize_file(
    source_path: Path,
    destination_path: Path,
) -> Path:
    """
    Normalize one extracted CSV file into Parquet.
    """

    print(
        f"Normalizing: {source_path.name}"
    )

    df = read_csv(source_path)

    normalized = normalize_dataframe(df)

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized.to_parquet(
        destination_path,
        index=False,
    )

    print(
        f"  Rows: {len(normalized):,}"
    )

    print(
        f"  Output: {destination_path}"
    )

    return destination_path


# ============================================================
# DIRECTORY
# ============================================================


def normalize_directory(
    symbol: str,
    source: str,
) -> list[Path]:
    """
    Normalize all extracted datasets belonging to a
    symbol/source combination.

    Example:

        data/extracted/EURUSD/TrueFX/

    becomes:

        data/normalized/EURUSD/ticks/
    """

    source_dir = (
        EXTRACTED_DIR
        / symbol
        / source
    )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Extracted data directory does not exist: "
            f"{source_dir}"
        )

    csv_files = sorted(
        source_dir.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No extracted CSV files found: "
            f"{source_dir}"
        )

    destination_dir = (
        NORMALIZED_DIR
        / symbol
        / "ticks"
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs: list[Path] = []

    for csv_path in csv_files:

        destination_path = (
            destination_dir
            / f"{csv_path.stem}.parquet"
        )

        output = normalize_file(
            source_path=csv_path,
            destination_path=destination_path,
        )

        outputs.append(output)

    return outputs