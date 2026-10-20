"""Trading-session context."""

from __future__ import annotations

import pandas as pd


def session_name(ts: pd.Timestamp) -> str:
    hour = ts.hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 17:
        return "LONDON_NY_OVERLAP"
    return "NEW_YORK"


def analyze_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["session"] = out["time"].map(session_name)
    out["bar_return"] = out["mid_close"].pct_change()

    rows = []
    for session, g in out.groupby("session", sort=False):
        rows.append({
            "session": session,
            "bars": int(len(g)),
            "share_pct": float(len(g) / len(out) * 100),
            "mean_bar_return_pct": float(g["bar_return"].mean() * 100),
            "mean_bar_range": float((g["mid_high"] - g["mid_low"]).mean()),
            "total_abs_return_pct": float(g["bar_return"].abs().sum() * 100),
        })

    return out, pd.DataFrame(rows)
