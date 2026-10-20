"""Trend-strength and directional persistence analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["mid_high"], df["mid_low"], df["mid_close"]
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )

    tr = df["true_range"]
    atr = tr.rolling(period, min_periods=period).mean()

    plus_di = 100 * plus_dm.rolling(period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period, min_periods=period).mean()


def analyze_trend(df: pd.DataFrame, period: int = 14) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    close = out["mid_close"]

    out["ema_fast"] = close.ewm(span=period, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=period * 3, adjust=False).mean()
    out["ema_distance_atr"] = (
        (out["ema_fast"] - out["ema_slow"]).abs() / out["atr"]
    )

    out["adx"] = _adx(out, period)
    out["direction"] = np.sign(close.diff())
    out["direction_run"] = (
        out["direction"].where(out["direction"] != 0)
        .groupby((out["direction"] != out["direction"].shift()).cumsum())
        .cumcount() + 1
    )

    valid_adx = out["adx"].dropna()
    if valid_adx.empty:
        raise ValueError("Not enough bars for trend analysis.")

    # These are deliberately descriptive bands, not trading signals.
    out["trend_strength"] = pd.cut(
        out["adx"],
        bins=[-np.inf, 15, 25, 40, np.inf],
        labels=["VERY_LOW", "LOW", "MODERATE", "STRONG"],
    )

    net = close.iloc[-1] / close.iloc[0] - 1
    abs_moves = close.diff().abs().sum()
    path_efficiency = abs(close.iloc[-1] - close.iloc[0]) / abs_moves if abs_moves else 0

    stats = {
        "adx_mean": float(valid_adx.mean()),
        "adx_median": float(valid_adx.median()),
        "adx_p90": float(valid_adx.quantile(0.90)),
        "trend_strength_mode": str(out["trend_strength"].mode().iloc[0]),
        "directional_return_pct": float(net * 100),
        "path_efficiency": float(path_efficiency),
        "mean_abs_ema_distance_atr": float(out["ema_distance_atr"].dropna().mean()),
        "strong_trend_pct": float((out["trend_strength"] == "STRONG").mean() * 100),
    }
    return out, stats
