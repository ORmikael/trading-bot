"""Orchestrator for repeatable market-condition analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .market import load_dataset, describe_market, add_true_range
from .volatility import analyze_volatility
from .trend import analyze_trend
from .ranges import analyze_ranges
from .breakouts import analyze_breakouts
from .sessions import analyze_sessions
from .trades import load_trades, load_metrics, analyze_trades, condition_performance
from .regimes import classify_regimes
from .report import generate_report, write_summary_json


def _metrics_dict(metrics_df: pd.DataFrame) -> dict:
    result = {}
    for _, row in metrics_df.iterrows():
        result[str(row["metric"])] = row["value"]
    return result


def analyze(dataset: str, results_dir: str, output_dir: str | None = None) -> Path:
    dataset_path = Path(dataset)
    results_path = Path(results_dir)

    if output_dir is None:
        output_path = results_path / "analysis"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trades_path = results_path / "trades.csv"
    metrics_path = results_path / "metrics.csv"

    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades.csv: {trades_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")

    df = load_dataset(str(dataset_path))
    df = add_true_range(df)

    market_stats = describe_market(df)

    df, volatility_stats = analyze_volatility(df)
    df, trend_stats = analyze_trend(df)
    df, range_stats = analyze_ranges(df)
    df, breakout_stats = analyze_breakouts(df)
    df, session_df = analyze_sessions(df)
    df, regime_stats = classify_regimes(df)

    trades = load_trades(trades_path)
    metrics_df = load_metrics(metrics_path)
    metrics = _metrics_dict(metrics_df)

    analyzed_trades, trade_analysis_df = analyze_trades(trades, df)
    condition_df = condition_performance(analyzed_trades)

    # Machine-readable outputs.
    pd.DataFrame([market_stats]).to_csv(output_path / "market_summary.csv", index=False)
    pd.DataFrame([volatility_stats]).to_csv(output_path / "volatility.csv", index=False)
    pd.DataFrame([trend_stats]).to_csv(output_path / "trend.csv", index=False)
    pd.DataFrame([range_stats]).to_csv(output_path / "ranges.csv", index=False)
    pd.DataFrame([breakout_stats]).to_csv(output_path / "breakouts.csv", index=False)
    session_df.to_csv(output_path / "sessions.csv", index=False)
    trade_analysis_df.to_csv(output_path / "trade_analysis.csv", index=False)
    regime_stats["summary"].to_csv(output_path / "regime_analysis.csv", index=False)
    condition_df.to_csv(output_path / "condition_performance.csv", index=False)

    payload = {
        "dataset": str(dataset_path),
        "results_dir": str(results_path),
        "market": market_stats,
        "volatility": volatility_stats,
        "trend": trend_stats,
        "ranges": range_stats,
        "breakouts": breakout_stats,
        "regimes": {
            "dominant_regime": regime_stats["dominant_regime"],
            "dominant_regime_pct": regime_stats["dominant_regime_pct"],
            "regime_counts": regime_stats["regime_counts"],
        },
        "strategy_metrics": metrics,
    }
    write_summary_json(output_path / "summary.json", payload)

    report = generate_report(
        output_path,
        dataset_path.name,
        market_stats,
        volatility_stats,
        trend_stats,
        range_stats,
        breakout_stats,
        regime_stats,
        metrics,
        condition_df,
    )

    print("=" * 70)
    print("MARKET CONDITION ANALYSIS")
    print("=" * 70)
    print(f"Dataset:   {dataset_path}")
    print(f"Results:   {results_path}")
    print(f"Analysis:  {output_path}")
    print()
    print(report)
    print()
    print("Generated:")
    for p in sorted(output_path.iterdir()):
        print(f"  {p.name}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze market conditions behind a trading strategy backtest."
    )
    parser.add_argument("--dataset", required=True, help="Processed market Parquet file.")
    parser.add_argument("--results", required=True, help="Backtest result directory containing trades.csv and metrics.csv.")
    parser.add_argument("--output", default=None, help="Optional analysis output directory. Defaults to <results>/analysis.")
    args = parser.parse_args()

    analyze(args.dataset, args.results, args.output)


if __name__ == "__main__":
    main()
