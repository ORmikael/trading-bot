"""Composite market-regime classification."""

from __future__ import annotations

import pandas as pd
import numpy as np


def classify_regimes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    adx = out["adx"]
    atr_regime = out["volatility_regime"]
    range_quality = out["range_quality"]
    breakout = out["breakout"]

    regime = np.full(len(out), "TRANSITION", dtype=object)

    strong_trend = adx >= 25
    strong_range = (
        range_quality.isin(["NARROW", "HEALTHY", "WIDE"])
        & (adx < 20)
        & (~breakout)
    )
    breakout_regime = breakout | (adx >= 25) & (
        out["ema_distance_atr"] >= 0.75
    )

    regime[strong_range.fillna(False).to_numpy()] = "RANGE"
    regime[strong_trend.fillna(False).to_numpy()] = "TREND"
    regime[breakout_regime.fillna(False).to_numpy()] = "BREAKOUT"

    out["regime"] = regime

    summary = (
        out.groupby("regime")
        .agg(
            bars=("regime", "size"),
            mean_atr=("atr", "mean"),
            mean_adx=("adx", "mean"),
        )
        .reset_index()
    )
    summary["share_pct"] = summary["bars"] / len(out) * 100

    dominant = summary.sort_values("bars", ascending=False).iloc[0]
    stats = {
        "dominant_regime": str(dominant["regime"]),
        "dominant_regime_pct": float(dominant["share_pct"]),
        "regime_counts": {
            str(r["regime"]): int(r["bars"]) for _, r in summary.iterrows()
        },
    }
    return out, {"summary": summary, **stats}
