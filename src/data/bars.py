# Create derived OHLC/timeframe datasets.
"""
Market Bar Generation Layer

Converts normalized tick data into time-based market bars.

Input:

    data/normalized/EURUSD/ticks/*.parquet

Output:

    data/processed/EURUSD/1m/*.parquet
    data/processed/EURUSD/5m/*.parquet
    data/processed/EURUSD/15m/*.parquet

The generated bars preserve bid/ask information.

This module does NOT:
    - execute trades
    - calculate strategy signals
    - calculate PnL
    - run backtests
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "normalized"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# TIMEFRAMES
# ============================================================

TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
}


# ============================================================
# TICK READER
# ============================================================


def load_ticks(
    path: Path,
) -> pd.DataFrame:
    """
    Load a normalized Parquet tick dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Normalized dataset not found: {path}"
        )

    df = pd.read_parquet(path)

    required_columns = {
        "symbol",
        "timestamp",
        "bid",
        "ask",
        "spread",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Normalized dataset is missing "
            f"columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


# ============================================================
# BAR GENERATION
# ============================================================


def generate_bars_for_dataframe(
    df: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """
    Generate bid/ask bars from normalized ticks.

    Parameters
    ----------
    df:
        Normalized tick dataframe.

    timeframe:
        1m, 5m, or 15m.
    """

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    rule = TIMEFRAME_MAP[timeframe]

    if df.empty:
        return pd.DataFrame()

    working = df.copy()

    working = working.sort_values(
        "timestamp"
    )

    working = working.set_index(
        "timestamp"
    )

    # --------------------------------------------------------
    # Bid OHLC
    # --------------------------------------------------------

    bid = working["bid"].resample(
        rule
    ).ohlc()

    bid.columns = [
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
    ]

    # --------------------------------------------------------
    # Ask OHLC
    # --------------------------------------------------------

    ask = working["ask"].resample(
        rule
    ).ohlc()

    ask.columns = [
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]

    # --------------------------------------------------------
    # Spread statistics
    # --------------------------------------------------------

    spread = pd.DataFrame(
        {
            "spread_mean": (
                working["spread"]
                .resample(rule)
                .mean()
            ),
            "spread_min": (
                working["spread"]
                .resample(rule)
                .min()
            ),
            "spread_max": (
                working["spread"]
                .resample(rule)
                .max()
            ),
        }
    )
 

    # --------------------------------------------------------
    # Tick count
    # --------------------------------------------------------

    tick_count = (
        working["bid"]
        .resample(rule)
        .count()
        .rename("tick_count")
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    bars = pd.concat(
        [
            bid,
            ask,
            spread,
            tick_count,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Remove completely empty periods
    # --------------------------------------------------------

    price_columns = [
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]

    bars = bars.dropna(
        subset=price_columns,
        how="all",
    )

    # --------------------------------------------------------
    # Symbol
    # --------------------------------------------------------

    symbol = (
        df["symbol"]
        .dropna()
        .iloc[0]
    )

    bars.insert(
        0,
        "symbol",
        symbol,
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    bars = bars.reset_index()

    # --------------------------------------------------------
    # Mid prices
    # --------------------------------------------------------

    bars["mid_open"] = (
        bars["bid_open"]
        + bars["ask_open"]
    ) / 2

    bars["mid_high"] = (
        bars["bid_high"]
        + bars["ask_high"]
    ) / 2

    bars["mid_low"] = (
        bars["bid_low"]
        + bars["ask_low"]
    ) / 2

    bars["mid_close"] = (
        bars["bid_close"]
        + bars["ask_close"]
    ) / 2

    # --------------------------------------------------------
    # Column ordering
    # --------------------------------------------------------

    bars = bars[
        [
            "symbol",
            "timestamp",

            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",

            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",

            "mid_open",
            "mid_high",
            "mid_low",
            "mid_close",

            "spread_mean",
            "spread_min",
            "spread_max",

            "tick_count",
        ]
    ]

    return bars


# ============================================================
# SINGLE FILE
# ============================================================


def generate_bars_for_file(
    source_path: Path,
    destination_path: Path,
    timeframe: str,
) -> Path:
    """
    Generate bars from one normalized Parquet file.
    """

    print(
        f"Generating {timeframe} bars: "
        f"{source_path.name}"
    )

    df = load_ticks(
        source_path
    )

    bars = generate_bars_for_dataframe(
        df,
        timeframe,
    )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bars.to_parquet(
        destination_path,
        index=False,
    )

    print(
        f"  Ticks: {len(df):,}"
    )

    print(
        f"  Bars:  {len(bars):,}"
    )

    print(
        f"  Output: {destination_path}"
    )

    return destination_path


# ============================================================
# DIRECTORY
# ============================================================


def generate_bars(
    symbol: str,
    timeframe: str,
) -> list[Path]:
    """
    Generate bars for all normalized tick datasets.

    Example:

        normalized/EURUSD/ticks/*.parquet

    becomes:

        processed/EURUSD/1m/*.parquet
    """

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    source_dir = (
        NORMALIZED_DIR
        / symbol
        / "ticks"
    )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Normalized directory does not exist: "
            f"{source_dir}"
        )

    parquet_files = sorted(
        source_dir.glob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No normalized Parquet datasets found: "
            f"{source_dir}"
        )

    destination_dir = (
        PROCESSED_DIR
        / symbol
        / timeframe
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs: list[Path] = []

    for parquet_path in parquet_files:

        destination_path = (
            destination_dir
            / parquet_path.name
        )

        output = generate_bars_for_file(
            source_path=parquet_path,
            destination_path=destination_path,
            timeframe=timeframe,
        )

        outputs.append(output)

    return outputs