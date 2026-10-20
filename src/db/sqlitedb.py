"""
SQLite Database Layer

Responsible for:
    - Database initialization
    - Schema creation
    - Backtest run tracking
    - Run status updates
    - Backtest metric storage
    - Database connections

The rest of the application should interact with SQLite
through this module rather than opening database connections
directly.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


# ============================================================
# DATABASE PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DATABASE_PATH = RESULTS_DIR / "backtests.db"


# ============================================================
# CONNECTION
# ============================================================


def get_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZATION
# ============================================================


def initialize_database() -> None:
    """
    Create the SQLite database and required tables.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Backtest runs
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                run_id TEXT NOT NULL UNIQUE,

                strategy TEXT NOT NULL,

                symbol TEXT NOT NULL,

                source TEXT NOT NULL,

                timeframe TEXT NOT NULL,

                started_at TEXT NOT NULL,

                completed_at TEXT,

                status TEXT NOT NULL,

                data_path TEXT,

                results_path TEXT,

                notes TEXT

            )
            """
        )

        # ----------------------------------------------------
        # Backtest metrics
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_metrics (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                run_id TEXT NOT NULL,

                metric TEXT NOT NULL,

                value REAL,

                FOREIGN KEY (run_id)
                    REFERENCES backtest_runs(run_id)

            )
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# BACKTEST RUNS
# ============================================================


def create_run(
    strategy: str,
    symbol: str,
    source: str,
    timeframe: str,
) -> str:
    """
    Create a new backtest run.

    Returns:
        Unique run ID.
    """

    timestamp = datetime.now()

    run_id = timestamp.strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO backtest_runs (

                run_id,
                strategy,
                symbol,
                source,
                timeframe,
                started_at,
                status

            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                strategy,
                symbol,
                source,
                timeframe,
                timestamp.isoformat(),
                "INITIALIZING",
            ),
        )

        connection.commit()

    finally:

        connection.close()

    return run_id


def update_run_status(
    run_id: str,
    status: str,
    *,
    completed_at: str | None = None,
    data_path: str | None = None,
    results_path: str | None = None,
    notes: str | None = None,
) -> None:
    """
    Update the state of a backtest run.
    """

    connection = get_connection()

    try:

        connection.execute(
            """
            UPDATE backtest_runs

            SET
                status = ?,
                completed_at = ?,
                data_path = ?,
                results_path = ?,
                notes = ?

            WHERE run_id = ?
            """,
            (
                status,
                completed_at,
                data_path,
                results_path,
                notes,
                run_id,
            ),
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# METRICS
# ============================================================


def store_metric(
    run_id: str,
    metric: str,
    value: float | None,
) -> None:
    """
    Store one backtest metric.
    """

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO backtest_metrics (
                run_id,
                metric,
                value
            )
            VALUES (?, ?, ?)
            """,
            (
                run_id,
                metric,
                value,
            ),
        )

        connection.commit()

    finally:

        connection.close()


def store_metrics(
    run_id: str,
    metrics: dict[str, float | None],
) -> None:
    """
    Store multiple backtest metrics.
    """

    connection = get_connection()

    try:

        connection.executemany(
            """
            INSERT INTO backtest_metrics (
                run_id,
                metric,
                value
            )
            VALUES (?, ?, ?)
            """,
            [
                (
                    run_id,
                    metric,
                    value,
                )
                for metric, value in metrics.items()
            ],
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# QUERY FUNCTIONS
# ============================================================


def get_run(
    run_id: str,
) -> sqlite3.Row | None:
    """
    Retrieve a backtest run by run ID.
    """

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT *
            FROM backtest_runs
            WHERE run_id = ?
            """,
            (run_id,),
        )

        return cursor.fetchone()

    finally:

        connection.close()


def get_run_metrics(
    run_id: str,
) -> list[sqlite3.Row]:
    """
    Retrieve all metrics belonging to a backtest run.
    """

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT
                metric,
                value
            FROM backtest_metrics
            WHERE run_id = ?
            ORDER BY metric
            """,
            (run_id,),
        )

        return cursor.fetchall()

    finally:

        connection.close()


def get_recent_runs(
    limit: int = 10,
) -> list[sqlite3.Row]:
    """
    Retrieve the most recent backtest runs.
    """

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            SELECT *
            FROM backtest_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        return cursor.fetchall()

    finally:

        connection.close()
