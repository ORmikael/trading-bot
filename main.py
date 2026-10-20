from __future__ import annotations

"""
Trading Bot Pipeline Orchestrator

main.py is responsible for coordinating the trading-data and
backtesting pipeline.

It does NOT contain:
    - data extraction logic
    - validation logic
    - normalization logic
    - bar-generation logic
    - strategy logic
    - backtesting logic
    - execution logic

Those responsibilities belong to their respective modules.

Pipeline:

    Raw archives
        ↓
    Extraction
        ↓
    Validation
        ↓
    Normalization
        ↓
    Bar generation
        ↓
    Backtesting
        ↓
    Results
"""

"""
Trading Bot Pipeline

Application entry point and pipeline orchestrator.

Responsibilities:
    - Parse command-line arguments
    - Initialize the database
    - Create a backtest run
    - Orchestrate extraction
    - Orchestrate validation
    - Orchestrate normalization
    - Orchestrate bar generation
    - Orchestrate backtesting
    - Record run status

Business logic belongs to the individual modules.
"""


import argparse
from datetime import datetime
from pathlib import Path


from src.db import sqlitedb
from src.data.extract import (
    extract_source,
    print_result as print_extraction_result,
)
from src.data.validate import (
    validate_directory,
    print_result as print_validation_result,
    print_summary,)
from src.data.normalize import normalize_directory
from src.data.bars import generate_bars
from src.backtests.range_pyramid_backtester import run_backtest


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"


# ============================================================
# PIPELINE
# ============================================================


def run_pipeline(
    symbol: str,
    source: str,
    strategy: str,
    timeframe: str,
) -> None:
    """
    Execute the complete trading-bot data and backtest pipeline.

    Pipeline:

        Raw Archives
             |
             v
        Extraction
             |
             v
        Validation
             |
             v
        Normalization
             |
             v
        Bar Generation
             |
             v
        Backtest
             |
             v
        Results
    """

    print("=" * 70)
    print("TRADING BOT")
    print("=" * 70)

    print(f"Symbol:    {symbol}")
    print(f"Source:    {source}")
    print(f"Strategy:  {strategy}")
    print(f"Timeframe: {timeframe}")

    # ========================================================
    # DATABASE INITIALIZATION
    # ========================================================

    sqlitedb.initialize_database()

    run_id = sqlitedb.create_run(
        strategy=strategy,
        symbol=symbol,
        source=source,
        timeframe=timeframe,
    )

    print(f"\nRun ID: {run_id}")

    try:

        # ====================================================
        # 1. EXTRACTION
        # ====================================================

        print("\n" + "=" * 70)
        print("1. EXTRACTION")
        print("=" * 70)

        extraction = extract_source(
            symbol=symbol,
            source=source,
        )
        print_extraction_result(extraction)

        # # ====================================================
        # # 2. VALIDATION
        # # ====================================================

        # print("\n" + "=" * 70)
        # print("2. VALIDATION")
        # print("=" * 70)

        # validation_status = validate_directory(
        #     symbol=symbol,
        #     source=source,
        # )

        # if validation_status == "FAIL":

        #     sqlitedb.update_run_status(
        #         run_id,
        #         "VALIDATION_FAILED",
        #         completed_at=datetime.now().isoformat(),
        #     )

        #     raise RuntimeError(
        #         "Dataset validation failed. "
        #         "Pipeline stopped before normalization."
        #     )
        # ----------------------------------------------------
        # 2. Validation
        # ----------------------------------------------------

        # from src.data.validator import (
        #     validate_directory,
        #     print_result as print_validation_result,
        #     print_summary,
        # )

        print("\n" + "=" * 70)
        print("2. VALIDATION")
        print("=" * 70)

        validation = validate_directory(
            symbol=symbol,
            source=source,
        )

        for result in validation.results:
            print_validation_result(result)

        print_summary(validation)

        if not validation.passed:
        
            sqlitedb.update_run_status(
                run_id,
                "VALIDATION_FAILED",
                completed_at=datetime.now().isoformat(),
                notes="Dataset validation failed.",
            )

            raise RuntimeError(
                "Dataset validation failed. "
                "Pipeline stopped before normalization."
            )
        # ====================================================
        # 3. NORMALIZATION
        # ====================================================

        print("\n" + "=" * 70)
        print("3. NORMALIZATION")
        print("=" * 70)

        normalize_directory(
            symbol=symbol,
            source=source,
        )

        # ====================================================
        # 4. BAR GENERATION
        # ====================================================

        print("\n" + "=" * 70)
        print("4. BAR GENERATION")
        print("=" * 70)

        generate_bars(
            symbol=symbol,
            timeframe=timeframe,
        )

        # ====================================================
        # 5. BACKTEST
        # ====================================================

        print("\n" + "=" * 70)
        print("5. BACKTEST")
        print("=" * 70)

        run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            run_id=run_id,
        )

        # ====================================================
        # 6. COMPLETE
        # ====================================================

        results_path = (
            RESULTS_DIR
            / strategy
            / f"run-{run_id}"
        )

        processed_path = (
            PROCESSED_DIR
            / symbol
            / timeframe
        )

        sqlitedb.update_run_status(
            run_id,
            "COMPLETED",
            completed_at=datetime.now().isoformat(),
            data_path=str(processed_path),
            results_path=str(results_path),
        )
        

        print("PIPELINE COMPLETE")
        print("=" * 70)

        print(f"Run ID:        {run_id}")
        print(f"Processed:     {processed_path}")
        print(f"Results:       {results_path}")

    except Exception as exc:

        sqlitedb.update_run_status(
            run_id,
            "FAILED",
            completed_at=datetime.now().isoformat(),
            notes=str(exc),
        )

        print("\n" + "=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)

        print(f"Run ID: {run_id}")
        print(f"Error:  {exc}")

        raise


# ============================================================
# CLI
# ============================================================


def main() -> None:
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description="Trading bot data and backtesting pipeline"
    )

    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Trading symbol",
    )

    parser.add_argument(
        "--source",
        default="TrueFX",
        help="Market-data source",
    )

    parser.add_argument(
        "--strategy",
        default="range_pyramid",
        help="Backtesting strategy",
    )

    parser.add_argument(
        "--timeframe",
        default="1m",
        choices=[
            "1m",
            "5m",
            "15m",
        ],
        help="Bar timeframe",
    )

    args = parser.parse_args()

    run_pipeline(
        symbol=args.symbol,
        source=args.source,
        strategy=args.strategy,
        timeframe=args.timeframe,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()