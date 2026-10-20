"""Breakout frequency, magnitude and continuation analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np


def analyze_breakouts(
    df: pd.DataFrame, atr_multiple: float = 0.75, continuation_bars: int = 5
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    close = out["mid_close"]

    prev_high = out["range_high"]
    prev_low = out["range_low"]

    out["up_breakout_distance_atr"] = (
        (out["mid_high"] - prev_high) / out["atr"]
    )
    out["down_breakout_distance_atr"] = (
        (prev_low - out["mid_low"]) / out["atr"]
    )

    out["up_breakout"] = out["up_breakout_distance_atr"] >= atr_multiple
    out["down_breakout"] = out["down_breakout_distance_atr"] >= atr_multiple
    out["breakout"] = out["up_breakout"] | out["down_breakout"]

    future_close = close.shift(-continuation_bars)
    out["up_continuation"] = out["up_breakout"] & (
        future_close > close + out["atr"] * atr_multiple
    )
    out["down_continuation"] = out["down_breakout"] & (
        future_close < close - out["atr"] * atr_multiple
    )

    breakout_count = int(out["breakout"].sum())
    continuation_count = int(
        (out["up_continuation"] | out["down_continuation"]).sum()
    )

    stats = {
        "breakout_threshold_atr": atr_multiple,
        "breakout_events": breakout_count,
        "breakout_events_per_1000_bars": float(
            breakout_count / max(len(out), 1) * 1000
        ),
        "up_breakouts": int(out["up_breakout"].sum()),
        "down_breakouts": int(out["down_breakout"].sum()),
        "continuation_events": continuation_count,
        "continuation_rate_pct": float(
            continuation_count / breakout_count * 100
        ) if breakout_count else 0.0,
        "failed_or_noncontinuing_rate_pct": float(
            (breakout_count - continuation_count) / breakout_count * 100
        ) if breakout_count else 0.0,
    }
    return out, stats
