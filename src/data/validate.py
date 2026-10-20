from __future__ import annotations
'''
Data validation and quality-control script

    Answers "is there anything  wrong with this dataset?"
    Produces: DATA QUALITY REPORT

"not perfectly clean" is not the same thing as "invalid."

DATA QUALITY REPORT    
===================

Dataset
-------
Symbol:              EURUSD
Rows:                755,175
Start:               2026-01-01 18:02:25.204
End:                 2026-01-30 21:59:56.406

Integrity
---------
Malformed rows:      0
Null values:         0
Duplicate rows:      0
Duplicate timestamps:0
Out-of-order ticks:  0

Quotes
------
Bid > Ask:            0
Zero spread:          108,119
Negative spread:      0

Price
------
Bid min:              1.15529
Bid max:              1.20812
Ask min:              1.15795
Ask max:              1.20824

Spread
-------
Median:               0.00001
Mean:                 0.00002
P99:                  0.00018
P99.9:                0.00082
Maximum:              0.00731

Timing
------
Median interval:      1.008s
Mean interval:        3.337s
Maximum gap:          172819.520s

Status
------
PASS
'''

"""
Market Data Validator

Validates extracted market-data CSV files.

Responsibilities:
    - Read datasets from data/extracted/
    - Check structural integrity
    - Check quote integrity
    - Analyze spread behavior
    - Analyze timestamp continuity
    - Produce a human-readable data quality report
    - Return PASS / WARNING / FAIL status

This module does NOT:
    - Extract archives
    - Modify source datasets
    - Clean or delete rows
    - Normalize data
    - Generate processed datasets
"""

"""
Market Data Validator

Validates extracted market-data CSV files.

Reads:

    data/extracted/<SYMBOL>/<SOURCE>/

Does NOT:

    - extract archives
    - modify source files
    - clean rows
    - normalize data
    - create bars
    - run backtests

Primary interface:

    validate_directory(symbol, source)

Returns:

    ValidationResult
"""


import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTRACTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "extracted"
)


COLUMNS = [
    "symbol",
    "timestamp",
    "bid",
    "ask",
]


# ============================================================
# THRESHOLDS
# ============================================================

MAX_NORMAL_SPREAD = 0.00100

MAX_NORMAL_GAP_SECONDS = 6 * 60 * 60

MAX_ZERO_SPREAD_RATIO = 0.10

LARGEST_SPREAD_COUNT = 20


# ============================================================
# RESULT OBJECTS
# ============================================================


@dataclass
class ValidationResult:
    """
    Validation result for one dataset.
    """

    file: Path
    symbol: str

    row_count: int
    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None

    malformed_rows: int
    null_values: int
    duplicate_rows: int
    duplicate_timestamps: int
    out_of_order_ticks: int

    bid_greater_than_ask: int
    zero_spread: int
    negative_spread: int
    positive_spread: int

    bid_min: float | None
    bid_max: float | None
    ask_min: float | None
    ask_max: float | None

    spread_min: float | None
    spread_mean: float | None
    spread_median: float | None
    spread_max: float | None

    spread_quantiles: dict[float, float]

    unusually_large_spreads: int
    unusually_large_gaps: int
    zero_spread_ratio: float

    median_tick_interval: float | None
    mean_tick_interval: float | None
    maximum_gap: float | None

    status: str
    failures: list[str] = field(
        default_factory=list
    )
    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def passed(self) -> bool:
        """Return True if no critical failures exist."""

        return self.status != "FAIL"


@dataclass
class ValidationSummary:
    """
    Validation result for an entire dataset directory.
    """

    symbol: str
    source: str
    results: list[ValidationResult]

    @property
    def passed(self) -> bool:
        """Return True if every dataset passed."""

        return all(
            result.status != "FAIL"
            for result in self.results
        )

    @property
    def failed(self) -> list[ValidationResult]:
        return [
            result
            for result in self.results
            if result.status == "FAIL"
        ]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [
            result
            for result in self.results
            if result.status == "WARNING"
        ]


# ============================================================
# LOADING
# ============================================================


def load_dataset(
    csv_path: Path,
) -> pd.DataFrame:
    """
    Read an extracted TrueFX CSV.

    TrueFX files contain:

        EUR/USD,timestamp,bid,ask

    No header row is expected.

    ASCII is used deliberately because the
    extracted TrueFX dataset has been verified
    as ASCII.
    """

    return pd.read_csv(
        csv_path,
        encoding="ascii",
        header=None,
        names=COLUMNS,
    )


# ============================================================
# PREPARATION
# ============================================================


def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Convert raw fields to their expected types.

    Returns:

        prepared dataframe
        malformed row count
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y%m%d %H:%M:%S.%f",
        errors="coerce",
    )

    df["bid"] = pd.to_numeric(
        df["bid"],
        errors="coerce",
    )

    df["ask"] = pd.to_numeric(
        df["ask"],
        errors="coerce",
    )

    malformed_mask = (
        df[
            [
                "symbol",
                "timestamp",
                "bid",
                "ask",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    malformed_rows = int(
        malformed_mask.sum()
    )

    df["spread"] = (
        df["ask"] - df["bid"]
    )

    return df, malformed_rows


# ============================================================
# STATUS
# ============================================================


def determine_status(
    metrics: dict,
) -> tuple[
    str,
    list[str],
    list[str],
]:
    """
    Determine dataset status.

    FAIL:
        malformed rows > 0
        null values > 0
        bid > ask > 0
        negative spread > 0
        duplicate rows > 0
        duplicate timestamps > 0
        out-of-order ticks > 0

    WARNING:
        unusually large spreads
        unusually large gaps
        unusually high zero-spread ratio

    PASS:
        no critical integrity violations
    """

    failures: list[str] = []
    warnings: list[str] = []

    # --------------------------------------------------------
    # CRITICAL FAILURES
    # --------------------------------------------------------

    if metrics["malformed_rows"] > 0:
        failures.append(
            "Malformed rows detected"
        )

    if metrics["null_values"] > 0:
        failures.append(
            "Null values detected"
        )

    if metrics["bid_greater_than_ask"] > 0:
        failures.append(
            "Bid greater than Ask detected"
        )

    if metrics["negative_spread"] > 0:
        failures.append(
            "Negative spreads detected"
        )

    if metrics["duplicate_rows"] > 0:
        failures.append(
            "Duplicate rows detected"
        )

    if metrics["duplicate_timestamps"] > 0:
        failures.append(
            "Duplicate timestamps detected"
        )

    if metrics["out_of_order_ticks"] > 0:
        failures.append(
            "Out-of-order ticks detected"
        )

    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    if metrics["unusually_large_spreads"] > 0:
        warnings.append(
            "Unusually large spreads detected"
        )

    if metrics["unusually_large_gaps"] > 0:
        warnings.append(
            "Unusually large timestamp gaps detected"
        )

    if (
        metrics["zero_spread_ratio"]
        > MAX_ZERO_SPREAD_RATIO
    ):
        warnings.append(
            "Unusually high zero-spread ratio"
        )

    if failures:
        return "FAIL", failures, warnings

    if warnings:
        return "WARNING", failures, warnings

    return "PASS", failures, warnings


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    df: pd.DataFrame,
    malformed_rows: int,
) -> dict:
    """
    Calculate validation metrics.
    """

    row_count = len(df)

    timestamp_delta = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    zero_spread = int(
        (df["spread"] == 0).sum()
    )

    zero_spread_ratio = (
        zero_spread / row_count
        if row_count
        else 0
    )

    quantiles = (
        df["spread"]
        .quantile(
            [
                0.00,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                0.999,
                1.00,
            ]
        )
        .to_dict()
        if row_count
        else {}
    )

    return {
        "row_count": row_count,

        "start_time": (
            df["timestamp"].min()
            if row_count
            else None
        ),

        "end_time": (
            df["timestamp"].max()
            if row_count
            else None
        ),

        # Integrity
        "malformed_rows": malformed_rows,

        "null_values": int(
            df.isnull().sum().sum()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "duplicate_timestamps": int(
            df["timestamp"]
            .duplicated()
            .sum()
        ),

        "out_of_order_ticks": int(
            (timestamp_delta < 0).sum()
        ),

        # Quotes
        "bid_greater_than_ask": int(
            (df["bid"] > df["ask"]).sum()
        ),

        "zero_spread": zero_spread,

        "negative_spread": int(
            (df["spread"] < 0).sum()
        ),

        "positive_spread": int(
            (df["spread"] > 0).sum()
        ),

        # Price
        "bid_min": (
            df["bid"].min()
            if row_count
            else None
        ),

        "bid_max": (
            df["bid"].max()
            if row_count
            else None
        ),

        "ask_min": (
            df["ask"].min()
            if row_count
            else None
        ),

        "ask_max": (
            df["ask"].max()
            if row_count
            else None
        ),

        # Spread
        "spread_min": (
            df["spread"].min()
            if row_count
            else None
        ),

        "spread_mean": (
            df["spread"].mean()
            if row_count
            else None
        ),

        "spread_median": (
            df["spread"].median()
            if row_count
            else None
        ),

        "spread_max": (
            df["spread"].max()
            if row_count
            else None
        ),

        "spread_quantiles": quantiles,

        "unusually_large_spreads": int(
            (
                df["spread"]
                > MAX_NORMAL_SPREAD
            ).sum()
        ),

        # Timing
        "median_tick_interval": (
            timestamp_delta.median()
            if row_count
            else None
        ),

        "mean_tick_interval": (
            timestamp_delta.mean()
            if row_count
            else None
        ),

        "maximum_gap": (
            timestamp_delta.max()
            if row_count
            else None
        ),

        "unusually_large_gaps": int(
            (
                timestamp_delta
                > MAX_NORMAL_GAP_SECONDS
            ).sum()
        ),

        "zero_spread_ratio":
            zero_spread_ratio,
    }


# ============================================================
# SINGLE FILE
# ============================================================


def validate_file(
    csv_path: Path,
) -> ValidationResult:
    """
    Validate one extracted CSV.

    This is the primary low-level validator
    interface.
    """

    df = load_dataset(csv_path)

    df, malformed_rows = prepare_dataset(df)

    metrics = calculate_metrics(
        df,
        malformed_rows,
    )

    status, failures, warnings = (
        determine_status(metrics)
    )

    symbol = (
        str(df["symbol"].iloc[0])
        if len(df)
        else "UNKNOWN"
    )

    return ValidationResult(
        file=csv_path,
        symbol=symbol,

        row_count=metrics["row_count"],
        start_time=metrics["start_time"],
        end_time=metrics["end_time"],

        malformed_rows=metrics[
            "malformed_rows"
        ],

        null_values=metrics[
            "null_values"
        ],

        duplicate_rows=metrics[
            "duplicate_rows"
        ],

        duplicate_timestamps=metrics[
            "duplicate_timestamps"
        ],

        out_of_order_ticks=metrics[
            "out_of_order_ticks"
        ],

        bid_greater_than_ask=metrics[
            "bid_greater_than_ask"
        ],

        zero_spread=metrics[
            "zero_spread"
        ],

        negative_spread=metrics[
            "negative_spread"
        ],

        positive_spread=metrics[
            "positive_spread"
        ],

        bid_min=metrics["bid_min"],
        bid_max=metrics["bid_max"],
        ask_min=metrics["ask_min"],
        ask_max=metrics["ask_max"],

        spread_min=metrics["spread_min"],
        spread_mean=metrics["spread_mean"],
        spread_median=metrics[
            "spread_median"
        ],
        spread_max=metrics["spread_max"],

        spread_quantiles=metrics[
            "spread_quantiles"
        ],

        unusually_large_spreads=metrics[
            "unusually_large_spreads"
        ],

        unusually_large_gaps=metrics[
            "unusually_large_gaps"
        ],

        zero_spread_ratio=metrics[
            "zero_spread_ratio"
        ],

        median_tick_interval=metrics[
            "median_tick_interval"
        ],

        mean_tick_interval=metrics[
            "mean_tick_interval"
        ],

        maximum_gap=metrics[
            "maximum_gap"
        ],

        status=status,
        failures=failures,
        warnings=warnings,
    )


# ============================================================
# DIRECTORY
# ============================================================


def validate_directory(
    symbol: str,
    source: str,
) -> ValidationSummary:
    """
    Validate all extracted CSV files for a
    symbol/source combination.

    This is the primary interface used by main.py.
    """

    directory = (
        EXTRACTED_DIR
        / symbol
        / source
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Extracted data directory does not exist:\n"
            f"{directory}"
        )

    csv_files = sorted(
        directory.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV datasets found in:\n"
            f"{directory}"
        )

    results = [
        validate_file(csv_file)
        for csv_file in csv_files
    ]

    return ValidationSummary(
        symbol=symbol,
        source=source,
        results=results,
    )


# ============================================================
# REPORTING
# ============================================================


def print_result(
    result: ValidationResult,
) -> None:
    """
    Print the detailed human-readable report
    for one dataset.
    """

    print()
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    print()
    print("Dataset")
    print("-------")

    print(
        f"File:                "
        f"{result.file.name}"
    )

    print(
        f"Symbol:              "
        f"{result.symbol}"
    )

    print(
        f"Rows:                "
        f"{result.row_count:,}"
    )

    print(
        f"Start:               "
        f"{result.start_time}"
    )

    print(
        f"End:                 "
        f"{result.end_time}"
    )

    print()
    print("Integrity")
    print("---------")

    print(
        f"Malformed rows:      "
        f"{result.malformed_rows:,}"
    )

    print(
        f"Null values:         "
        f"{result.null_values:,}"
    )

    print(
        f"Duplicate rows:      "
        f"{result.duplicate_rows:,}"
    )

    print(
        f"Duplicate timestamps:"
        f" {result.duplicate_timestamps:,}"
    )

    print(
        f"Out-of-order ticks:  "
        f"{result.out_of_order_ticks:,}"
    )

    print()
    print("Quotes")
    print("------")

    print(
        f"Bid > Ask:           "
        f"{result.bid_greater_than_ask:,}"
    )

    print(
        f"Zero spread:         "
        f"{result.zero_spread:,}"
    )

    print(
        f"Negative spread:     "
        f"{result.negative_spread:,}"
    )

    print(
        f"Zero spread ratio:   "
        f"{result.zero_spread_ratio:.2%}"
    )

    print()
    print("Price")
    print("-----")

    print(
        f"Bid min:              "
        f"{result.bid_min:.5f}"
    )

    print(
        f"Bid max:              "
        f"{result.bid_max:.5f}"
    )

    print(
        f"Ask min:              "
        f"{result.ask_min:.5f}"
    )

    print(
        f"Ask max:              "
        f"{result.ask_max:.5f}"
    )

    print()
    print("Spread")
    print("------")

    print(
        f"Minimum:              "
        f"{result.spread_min:.6f}"
    )

    print(
        f"Mean:                 "
        f"{result.spread_mean:.6f}"
    )

    print(
        f"Median:               "
        f"{result.spread_median:.6f}"
    )

    print(
        f"P99:                  "
        f"{result.spread_quantiles.get(0.99, 0):.6f}"
    )

    print(
        f"P99.9:                "
        f"{result.spread_quantiles.get(0.999, 0):.6f}"
    )

    print(
        f"Maximum:              "
        f"{result.spread_max:.6f}"
    )

    print()
    print("Timing")
    print("------")

    print(
        f"Median interval:      "
        f"{result.median_tick_interval:.3f}s"
    )

    print(
        f"Mean interval:        "
        f"{result.mean_tick_interval:.3f}s"
    )

    print(
        f"Maximum gap:          "
        f"{result.maximum_gap:.3f}s"
    )

    print()
    print("Status")
    print("------")

    print(result.status)

    if result.failures:

        print()
        print("Failures")
        print("--------")

        for failure in result.failures:
            print(f"- {failure}")

    if result.warnings:

        print()
        print("Warnings")
        print("--------")

        for warning in result.warnings:
            print(f"- {warning}")

    print("=" * 70)


def print_summary(
    summary: ValidationSummary,
) -> None:
    """Print directory-level validation summary."""

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for result in summary.results:
        print(
            f"{result.file.name:<35}"
            f"{result.status}"
        )

    print()

    print(
        f"Datasets: {len(summary.results)}"
    )

    print(
        f"Failed:   {len(summary.failed)}"
    )

    print(
        f"Warnings: {len(summary.warnings)}"
    )

    print(
        f"Passed:   "
        f"{sum(r.status == 'PASS' for r in summary.results)}"
    )

    print("=" * 70)


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Validate extracted market data."
    )

    parser.add_argument(
        "--symbol",
        required=True,
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    args = parser.parse_args()

    summary = validate_directory(
        symbol=args.symbol,
        source=args.source,
    )

    for result in summary.results:
        print_result(result)

    print_summary(summary)


if __name__ == "__main__":
    main()