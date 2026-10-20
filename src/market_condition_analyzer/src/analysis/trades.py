"""Trade-level strategy performance analysis."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def load_trades(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "entry_time", "exit_time", "side", "entry", "exit", "lots",
        "gross_pnl", "costs", "pnl", "reason"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trades file missing required columns: {sorted(missing)}")

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["duration_minutes"] = (
        df["exit_time"] - df["entry_time"]
    ).dt.total_seconds() / 60
    return df


def load_metrics(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, header=None, names=["metric", "value"])


def analyze_trades(trades: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = trades.copy()

    # Attach nearest market state at entry.
    state_cols = [
        "time", "atr", "atr_pct", "volatility_regime", "adx",
        "trend_strength", "range_width_atr", "range_position",
        "range_quality", "breakout"
    ]
    available = [c for c in state_cols if c in market.columns]
    state = market[available].sort_values("time")

    out = pd.merge_asof(
        out.sort_values("entry_time"),
        state,
        left_on="entry_time",
        right_on="time",
        direction="backward",
    )
    out = out.drop(columns=["time"], errors="ignore")

    out["win"] = out["pnl"] > 0
    out["loss"] = out["pnl"] < 0
    out["session"] = out["entry_time"].map(
        lambda x: "ASIA" if x.hour < 7
        else "LONDON" if x.hour < 12
        else "LONDON_NY_OVERLAP" if x.hour < 17
        else "NEW_YORK"
    )

    rows = []

    def add_group(label, group_col):
        for key, g in out.groupby(group_col, dropna=False):
            gp = g.loc[g["pnl"] > 0, "pnl"].sum()
            gl = -g.loc[g["pnl"] < 0, "pnl"].sum()
            rows.append({
                "dimension": label,
                "condition": str(key),
                "trades": int(len(g)),
                "wins": int(g["win"].sum()),
                "losses": int(g["loss"].sum()),
                "win_rate_pct": float(g["win"].mean() * 100),
                "net_pnl": float(g["pnl"].sum()),
                "gross_profit": float(gp),
                "gross_loss": float(gl),
                "profit_factor": float(gp / gl) if gl else np.inf,
                "average_trade": float(g["pnl"].mean()),
                "avg_duration_minutes": float(g["duration_minutes"].mean()),
            })

    for label, col in [
        ("exit_reason", "reason"),
        ("side", "side"),
        ("session", "session"),
        ("volatility", "volatility_regime"),
        ("trend_strength", "trend_strength"),
        ("range_quality", "range_quality"),
    ]:
        if col in out.columns:
            add_group(label, col)

    return out, pd.DataFrame(rows)


def condition_performance(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dimensions = [
        "volatility_regime", "trend_strength", "range_quality",
        "session", "reason", "side"
    ]
    for col in dimensions:
        if col not in trades.columns:
            continue
        for key, g in trades.groupby(col, dropna=False):
            gp = g.loc[g["pnl"] > 0, "pnl"].sum()
            gl = -g.loc[g["pnl"] < 0, "pnl"].sum()
            rows.append({
                "condition_type": col,
                "condition": str(key),
                "trades": int(len(g)),
                "win_rate_pct": float(g["pnl"].gt(0).mean() * 100),
                "net_pnl": float(g["pnl"].sum()),
                "profit_factor": float(gp / gl) if gl else np.inf,
                "average_trade": float(g["pnl"].mean()),
                "max_loss": float(g["pnl"].min()),
                "max_win": float(g["pnl"].max()),
            })
    return pd.DataFrame(rows)
