"""Core market statistics and dataset profiling."""

from __future__ import annotations

import pandas as pd
import numpy as np


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], utc=False)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False)
    else:
        raise ValueError("Dataset must contain timestamp or time.")

    required = {"mid_open", "mid_high", "mid_low", "mid_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    return df.sort_values("time").reset_index(drop=True)


def describe_market(df: pd.DataFrame) -> dict:
    close = df["mid_close"]
    high = df["mid_high"]
    low = df["mid_low"]
    returns = close.pct_change().dropna()

    first = df["time"].iloc[0]
    last = df["time"].iloc[-1]

    return {
        "rows": int(len(df)),
        "start": first.isoformat(sep=" "),
        "end": last.isoformat(sep=" "),
        "price_start": float(close.iloc[0]),
        "price_end": float(close.iloc[-1]),
        "net_price_change_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "high": float(high.max()),
        "low": float(low.min()),
        "mean_bar_range": float((high - low).mean()),
        "median_bar_range": float((high - low).median()),
        "realized_volatility_pct": float(returns.std() * np.sqrt(len(returns)) * 100),
    }


def add_true_range(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["mid_close"].shift(1)
    tr = pd.concat(
        [
            out["mid_high"] - out["mid_low"],
            (out["mid_high"] - prev_close).abs(),
            (out["mid_low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["true_range"] = tr
    return out
