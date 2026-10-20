# Market Condition Analyzer

Independent research tooling for answering:

> What market conditions existed during a backtest, and how did those
> conditions relate to strategy performance?

The analyzer does **not** modify the trading strategy or backtest engine.

## Architecture

```text
src/analysis/
├── __init__.py
├── market.py
├── volatility.py
├── trend.py
├── ranges.py
├── breakouts.py
├── sessions.py
├── trades.py
├── regimes.py
├── report.py
└── market_condition_analyzer.py
```

## Inputs

1. A processed market-data Parquet file, e.g.

```text
data/processed/EURUSD/1m/EURUSD-2025-11.parquet
```

2. A backtest result directory containing:

```text
trades.csv
equity.csv
metrics.csv
```

`equity.csv` is preserved as an accepted result artifact for future equity/drawdown
extensions; the current v0.1 analyzer derives strategy statistics primarily from
`metrics.csv` and `trades.csv`.

## Run

From the project root:

```bash
python -m src.analysis.market_condition_analyzer \
  --dataset data/processed/EURUSD/1m/EURUSD-2025-11.parquet \
  --results data/results/range_pyramid/run-20260910-114857-149339
```

The analyzer writes:

```text
<results>/
└── analysis/
    ├── market_summary.csv
    ├── volatility.csv
    ├── trend.csv
    ├── ranges.csv
    ├── breakouts.csv
    ├── sessions.csv
    ├── trade_analysis.csv
    ├── regime_analysis.csv
    ├── condition_performance.csv
    ├── summary.json
    └── report.md
```

## What it currently analyzes

### Market
- period
- price change
- price extremes
- bar range
- realized volatility

### Volatility
- ATR
- ATR percentile bands
- low/normal/high volatility regimes

### Trend
- ADX
- EMA separation relative to ATR
- directional return
- path efficiency
- trend-strength classification

### Range structure
- rolling support/resistance
- range width
- range width / ATR
- time inside range
- edge-reversion proxy
- strategy-aware range-width categories

### Breakouts
- breakout count
- up/down breakouts
- breakout threshold in ATR
- continuation rate
- non-continuation/failure rate

### Sessions
- Asia
- London
- London/New York overlap
- New York

### Strategy
Each trade is enriched with market state at entry:
- ATR
- volatility regime
- ADX
- trend strength
- range position
- range quality
- breakout state
- session

It then produces condition-level PnL, win rate, profit factor,
average trade, max loss and max win.

### Composite regimes
The current heuristic classifier labels bars as:
- RANGE
- TREND
- BREAKOUT
- TRANSITION

These are research labels, not trading signals.

## Reproducibility

The analyzer is intentionally independent of the strategy code.

Given the same:
- dataset
- backtest result directory
- analyzer version

it should reproduce the same analysis outputs.

Keep the generated `analysis/` directory with the backtest run for auditability.

## Important limitation

This is v0.1.

The report's natural-language conclusions are deliberately conservative.
The regime classifier and thresholds are research heuristics and should not
be treated as validated predictive rules.

The next development stage should test the analyzer across many months before
using its classifications to alter strategy behavior.
