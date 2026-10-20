"""Volatility and ATR analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np


def analyze_volatility(df: pd.DataFrame, atr_period: int = 14) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    tr = out["true_range"]
    out["atr"] = tr.rolling(atr_period, min_periods=atr_period).mean()
    out["atr_pct"] = out["atr"] / out["mid_close"] * 100
    out["bar_range"] = out["mid_high"] - out["mid_low"]
    out["return"] = out["mid_close"].pct_change()

    valid_atr = out["atr"].dropna()
    if valid_atr.empty:
        raise ValueError("Not enough bars for ATR analysis.")

    q1, q2 = valid_atr.quantile([1/3, 2/3])
    out["volatility_regime"] = np.select(
        [out["atr"] <= q1, out["atr"] <= q2],
        ["LOW", "NORMAL"],
        default="HIGH",
    )

    returns = out["return"].dropna()
    stats = {
        "atr_period": atr_period,
        "atr_mean": float(valid_atr.mean()),
        "atr_median": float(valid_atr.median()),
        "atr_min": float(valid_atr.min()),
        "atr_max": float(valid_atr.max()),
        "atr_p33": float(q1),
        "atr_p66": float(q2),
        "realized_volatility_pct": float(returns.std() * np.sqrt(len(returns)) * 100),
        "low_volatility_pct": float((out["volatility_regime"] == "LOW").mean() * 100),
        "normal_volatility_pct": float((out["volatility_regime"] == "NORMAL").mean() * 100),
        "high_volatility_pct": float((out["volatility_regime"] == "HIGH").mean() * 100),
    }
    return out, stats
