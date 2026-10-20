#!/usr/bin/env bash
set -euo pipefail

python -m src.analysis.market_condition_analyzer \
  --dataset "$1" \
  --results "$2"
