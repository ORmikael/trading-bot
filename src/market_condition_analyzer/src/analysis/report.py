"""Human-readable market-condition report generation."""

from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd


def _fmt(x, digits=2):
    if x is None:
        return "N/A"
    try:
        if math.isnan(float(x)):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _classification(regime: str) -> str:
    return {
        "RANGE": "Predominantly mean-reverting / ranging",
        "TREND": "Predominantly directional / trending",
        "BREAKOUT": "Breakout-dominated",
        "TRANSITION": "Mixed / transitional",
    }.get(regime, "Mixed / transitional")


def generate_report(
    output_dir: str | Path,
    dataset_name: str,
    market_stats: dict,
    volatility_stats: dict,
    trend_stats: dict,
    range_stats: dict,
    breakout_stats: dict,
    regime_stats: dict,
    metrics: dict,
    condition_df: pd.DataFrame,
) -> str:
    output_dir = Path(output_dir)

    dominant = regime_stats["dominant_regime"]
    pf = float(metrics.get("profit_factor", float("nan")))
    ret = float(metrics.get("return_pct", float("nan")))
    wr = float(metrics.get("win_rate_pct", float("nan")))
    dd = float(metrics.get("max_drawdown_pct", float("nan")))

    observations = []

    if ret > 0 and pf > 1:
        observations.append(
            f"The strategy produced positive expectancy over the tested period "
            f"(return {_fmt(ret)}%, profit factor {_fmt(pf, 3)})."
        )
    elif ret < 0 or pf < 1:
        observations.append(
            f"The strategy produced negative aggregate expectancy "
            f"(return {_fmt(ret)}%, profit factor {_fmt(pf, 3)})."
        )

    if wr < 50 and ret > 0:
        observations.append(
            f"Profitability did not require a majority win rate; the strategy won "
            f"{_fmt(wr)}% of closed trade parts, indicating that payoff asymmetry "
            f"and exit behavior mattered more than raw hit rate."
        )

    observations.append(
        f"The dominant measured market regime was {_classification(dominant).lower()}, "
        f"covering {_fmt(regime_stats['dominant_regime_pct'])}% of analyzed bars."
    )

    if breakout_stats["breakout_events"]:
        observations.append(
            f"{breakout_stats['breakout_events']} breakout events were detected at "
            f"the configured {breakout_stats['breakout_threshold_atr']} ATR threshold; "
            f"{_fmt(breakout_stats['continuation_rate_pct'])}% continued by the "
            f"defined continuation test."
        )

    if range_stats["edge_reversion_proxy_pct"] >= 20:
        observations.append(
            f"The edge-reversion proxy was {_fmt(range_stats['edge_reversion_proxy_pct'])}%, "
            f"supporting the presence of recurring mean-reversion behavior."
        )
    else:
        observations.append(
            f"The edge-reversion proxy was only {_fmt(range_stats['edge_reversion_proxy_pct'])}%, "
            f"so range-bound conditions should not be assumed to be consistently stable."
        )

    # Condition-level evidence.
    ranked = condition_df.sort_values("net_pnl", ascending=False).head(3)
    if not ranked.empty:
        labels = [
            f"{r['condition_type']}={r['condition']} ({_fmt(r['net_pnl'], 3)})"
            for _, r in ranked.iterrows()
        ]
        observations.append("Strongest observed condition buckets by net PnL: " + "; ".join(labels) + ".")

    confidence = "LOW"
    if len(condition_df) >= 6 and market_stats["rows"] >= 5000:
        confidence = "MEDIUM"
    if len(condition_df) >= 12 and market_stats["rows"] >= 20000:
        confidence = "HIGH"

    alignment = "UNCERTAIN"
    if ret > 0 and pf > 1 and dominant in {"RANGE", "BREAKOUT"}:
        alignment = "WELL ALIGNED"
    elif ret < 0 and dominant in {"TREND", "BREAKOUT"}:
        alignment = "POTENTIALLY MISALIGNED"
    elif ret < 0:
        alignment = "WEAK ALIGNMENT"

    start = market_stats["start"][:10]
    end = market_stats["end"][:10]

    report = f"""MARKET CONDITION ANALYSIS
=========================

Dataset:
{dataset_name}

Period:
{start} → {end}

MARKET CONTEXT
==============

Market classification:
{_classification(dominant)}

Trend strength:
{trend_stats['trend_strength_mode'].replace('_', ' ').title()}

Volatility:
{('High' if volatility_stats['high_volatility_pct'] > 50 else 'Low' if volatility_stats['low_volatility_pct'] > 50 else 'Moderate')}

Range behavior:
{('Strong' if range_stats['healthy_range_pct'] >= 25 and range_stats['inside_range_pct'] >= 30 else 'Mixed')}

Breakout behavior:
{('High continuation' if breakout_stats['continuation_rate_pct'] >= 50 else 'Low continuation / high failure rate' if breakout_stats['breakout_events'] else 'Few breakout events')}


STRATEGY PERFORMANCE
====================

Return:             {_fmt(ret)}%
Profit factor:      {_fmt(pf, 3)}
Win rate:           {_fmt(wr)}%
Max drawdown:       {_fmt(dd)}%


KEY OBSERVATIONS
================

""" + "\n".join(
        f"{i}. {text}" for i, text in enumerate(observations, 1)
    ) + f"""


REGIME ASSESSMENT
=================

Strategy appears {alignment} with this dataset.

Confidence: {confidence}

IMPORTANT
=========

This report describes statistical relationships observed in the supplied
dataset and backtest. It is not a guarantee of future profitability.
Regime labels are heuristic classifications intended for research and
should be validated across additional datasets before being used as
strategy-selection or risk-control rules.
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def write_summary_json(path: str | Path, payload: dict):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if hasattr(value, "item"):
            return clean(value.item())
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    Path(path).write_text(
        json.dumps(clean(payload), indent=2, default=str),
        encoding="utf-8",
    )
