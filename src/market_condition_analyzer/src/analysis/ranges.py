"""Range structure and mean-reversion suitability analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np


def analyze_ranges(
    df: pd.DataFrame, lookback: int = 48
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    out["range_high"] = out["mid_high"].shift(1).rolling(
        lookback, min_periods=lookback
    ).max()
    out["range_low"] = out["mid_low"].shift(1).rolling(
        lookback, min_periods=lookback
    ).min()
    out["range_width"] = out["range_high"] - out["range_low"]
    out["range_width_atr"] = out["range_width"] / out["atr"]

    width = out["range_width"]
    close = out["mid_close"]
    out["range_position"] = (close - out["range_low"]) / width.replace(0, np.nan)

    # How much of the current bar remains inside the previous rolling range.
    out["inside_range"] = (
        (out["mid_high"] <= out["range_high"])
        & (out["mid_low"] >= out["range_low"])
    )

    valid = out["range_width_atr"].dropna()
    if valid.empty:
        raise ValueError("Not enough bars for range analysis.")

    # Strategy-aware bands based on the current strategy's configured range/ATR window.
    out["range_quality"] = pd.cut(
        out["range_width_atr"],
        bins=[-np.inf, 2, 4, 8, 12, np.inf],
        labels=[
            "TOO_TIGHT",
            "NARROW",
            "HEALTHY",
            "WIDE",
            "TOO_WIDE",
        ],
    )

    # A simple reversal proxy: after touching an edge, did price move back toward
    # the middle over the next few bars?
    edge = ((out["range_position"] <= 0.2) | (out["range_position"] >= 0.8))
    future_close = close.shift(-5)
    midpoint = (out["range_high"] + out["range_low"]) / 2
    out["edge_reversion_5bar"] = edge & (
        ((out["range_position"] <= 0.2) & (future_close > close))
        | ((out["range_position"] >= 0.8) & (future_close < close))
    )

    stats = {
        "lookback": lookback,
        "median_range_width": float(width.dropna().median()),
        "median_range_width_atr": float(valid.median()),
        "healthy_range_pct": float((out["range_quality"] == "HEALTHY").mean() * 100),
        "narrow_range_pct": float((out["range_quality"] == "NARROW").mean() * 100),
        "wide_range_pct": float((out["range_quality"] == "WIDE").mean() * 100),
        "too_tight_pct": float((out["range_quality"] == "TOO_TIGHT").mean() * 100),
        "too_wide_pct": float((out["range_quality"] == "TOO_WIDE").mean() * 100),
        "inside_range_pct": float(out["inside_range"].mean() * 100),
        "edge_reversion_proxy_pct": float(out["edge_reversion_5bar"].mean() * 100),
    }
    return out, stats
