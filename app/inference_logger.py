"""
inference_logger.py — SQLite-backed logger for fraud prediction requests.

Every call to /predict is recorded here so you can:
  • Audit predictions after the fact.
  • Track model performance over time (fraud_rate drift, probability shift).
  • Replay or re-score historical transactions.

Design choices
--------------
• Raw sqlite3 — no ORM, no migrations framework.  The schema is tiny and
  unlikely to change frequently; sql3 is in the Python stdlib so there are
  zero extra dependencies.
• Thread-safety — sqlite3 connections are not thread-safe across threads.
  We open a short-lived connection per call (check_same_thread=False is NOT
  used).  This is slightly slower than a persistent connection but is correct
  under FastAPI's async/threading model without a connection pool.
• DB path — configurable via the PREDICTIONS_DB_PATH environment variable so
  different environments (dev, staging, prod) can point to different files or
  a shared network mount.
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH: str = os.getenv("PREDICTIONS_DB_PATH", "predictions.db")

# DDL executed once at module import time (if the table does not already exist)
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    TEXT      PRIMARY KEY,
    timestamp        TEXT      NOT NULL,
    amount           REAL      NOT NULL,
    fraud_probability REAL     NOT NULL,
    is_fraud         INTEGER   NOT NULL,   -- SQLite has no BOOLEAN; 1=True, 0=False
    model_version    TEXT      NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_connection() -> sqlite3.Connection:
    """
    Open and return a new SQLite connection to DB_PATH.

    `detect_types` enables automatic conversion of SQLite TEXT columns back
    to Python objects when `PARSE_DECLTYPES` is set (not used here, reserved
    for future TIMESTAMP columns).

    Returns
    -------
    sqlite3.Connection
        Caller is responsible for closing the connection (use `with` block).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts (col access by name)
    return conn


def _init_db() -> None:
    """
    Create the `predictions` table if it does not already exist.

    Called automatically at module import time so the file and schema are
    ready before the first request arrives.  Safe to call repeatedly — the
    `IF NOT EXISTS` guard makes it idempotent.
    """
    try:
        with _get_connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        logger.info("Prediction database initialised at '%s'.", DB_PATH)
    except sqlite3.Error as exc:
        logger.error("Failed to initialise prediction database: %s", exc)
        # Don't re-raise — a DB init failure should not crash the whole app
        # on import.  Errors will surface on the first write attempt.


# Initialise the schema as soon as this module is imported.
_init_db()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_prediction(
    *,
    prediction_id: str,
    amount: float,
    fraud_probability: float,
    is_fraud: bool,
    model_version: str,
    timestamp: str | None = None,
    # Extra kwargs forwarded from main.py (e.g. `transaction` dict) are
    # silently accepted and ignored so the call site doesn't need to change
    # when this signature evolves.
    **_ignored: Any,
) -> None:
    """
    Insert one prediction record into the `predictions` table.

    Parameters
    ----------
    prediction_id : str
        UUID-4 string uniquely identifying this prediction (from PredictionOutput).
    amount : float
        Transaction amount forwarded from TransactionInput.
    fraud_probability : float
        Model's estimated fraud probability in [0, 1].
    is_fraud : bool
        Hard classification result (True = fraud).
    model_version : str
        MLflow model version that produced this prediction.
    timestamp : str, optional
        ISO-8601 UTC string.  Defaults to *now* if not supplied.
    **_ignored
        Any extra keyword arguments (e.g. ``transaction`` dict from main.py)
        are accepted and silently discarded for forward compatibility.

    Returns
    -------
    None
        Errors are logged but not re-raised so a logging failure never kills
        an otherwise successful prediction response.
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    sql = """
        INSERT OR IGNORE INTO predictions
            (prediction_id, timestamp, amount, fraud_probability, is_fraud, model_version)
        VALUES
            (?, ?, ?, ?, ?, ?)
    """
    # `OR IGNORE` silently skips duplicate prediction_ids (idempotent retries).

    try:
        with _get_connection() as conn:
            conn.execute(
                sql,
                (
                    prediction_id,
                    ts,
                    amount,
                    fraud_probability,
                    int(is_fraud),
                    model_version,
                ),
            )
            conn.commit()
        logger.debug(
            "Logged prediction_id=%s  is_fraud=%s  prob=%.4f",
            prediction_id,
            is_fraud,
            fraud_probability,
        )
    except sqlite3.Error as exc:
        # Log the failure but never let a DB write error bubble up to the
        # caller — the prediction itself has already succeeded.
        logger.error("Failed to log prediction %s: %s", prediction_id, exc)


def get_recent_predictions(n: int = 100) -> list[dict[str, Any]]:
    """
    Return the most recent *n* predictions ordered newest-first.

    Parameters
    ----------
    n : int, optional
        Maximum number of rows to return (default 100).

    Returns
    -------
    list[dict[str, Any]]
        Each element is a dict with keys:
        ``prediction_id``, ``timestamp``, ``amount``,
        ``fraud_probability``, ``is_fraud`` (bool), ``model_version``.
        Returns an empty list if the database is unavailable.
    """
    sql = """
        SELECT prediction_id, timestamp, amount, fraud_probability,
               is_fraud, model_version
        FROM   predictions
        ORDER  BY timestamp DESC
        LIMIT  ?
    """
    try:
        with _get_connection() as conn:
            rows = conn.execute(sql, (n,)).fetchall()
        return [
            {
                "prediction_id": row["prediction_id"],
                "timestamp": row["timestamp"],
                "amount": row["amount"],
                "fraud_probability": row["fraud_probability"],
                "is_fraud": bool(row["is_fraud"]),
                "model_version": row["model_version"],
            }
            for row in rows
        ]
    except sqlite3.Error as exc:
        logger.error("get_recent_predictions failed: %s", exc)
        return []


def get_prediction_stats(n: int = 100) -> dict[str, Any]:
    """
    Compute aggregate statistics over the most recent *n* predictions.

    Parameters
    ----------
    n : int, optional
        Window size (default 100).

    Returns
    -------
    dict[str, Any]
        ``count``           — number of predictions in the window.
        ``fraud_rate``      — fraction classified as fraud (None if no rows).
        ``avg_probability`` — mean fraud_probability (None if no rows).
    """
    sql = """
        SELECT
            COUNT(*)                        AS count,
            AVG(is_fraud)                   AS fraud_rate,
            AVG(fraud_probability)          AS avg_probability
        FROM (
            SELECT is_fraud, fraud_probability
            FROM   predictions
            ORDER  BY timestamp DESC
            LIMIT  ?
        )
    """
    try:
        with _get_connection() as conn:
            row = conn.execute(sql, (n,)).fetchone()

        count = row["count"] if row else 0
        return {
            "count": count,
            "fraud_rate": round(row["fraud_rate"], 4)
            if row and row["fraud_rate"] is not None
            else None,
            "avg_probability": round(row["avg_probability"], 4)
            if row and row["avg_probability"] is not None
            else None,
        }
    except sqlite3.Error as exc:
        logger.error("get_prediction_stats failed: %s", exc)
        return {"count": 0, "fraud_rate": None, "avg_probability": None}
