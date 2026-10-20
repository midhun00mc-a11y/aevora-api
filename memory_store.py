"""Small SQLite store for explicit demo-customer preferences and fit feedback."""

import os
import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


def database_path() -> Path:
    return Path(os.getenv("AEVORA_DB_PATH", "aevora_memory.sqlite3"))


def _connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with _connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS customer_preferences (
                customer_id TEXT PRIMARY KEY,
                usual_size TEXT,
                fit_preference TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fit_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                recommended_size TEXT NOT NULL,
                sentiment TEXT NOT NULL CHECK (sentiment IN ('too_tight', 'too_loose', 'fits_well')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_feedback_customer_product ON fit_feedback(customer_id, product_id, id DESC)")


def get_preferences(customer_id: str) -> Optional[dict]:
    initialize_database()
    with _connection() as connection:
        row = connection.execute(
            "SELECT usual_size, fit_preference FROM customer_preferences WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
    if row is None:
        return None
    return {"usual_size": row["usual_size"], "fit_preference": row["fit_preference"]}


def save_preferences(customer_id: str, usual_size: Optional[str], fit_preference: Optional[str]) -> dict:
    initialize_database()
    current = get_preferences(customer_id) or {"usual_size": None, "fit_preference": None}
    usual_size = usual_size if usual_size is not None else current["usual_size"]
    fit_preference = fit_preference if fit_preference is not None else current["fit_preference"]
    with _connection() as connection:
        connection.execute("""
            INSERT INTO customer_preferences(customer_id, usual_size, fit_preference, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(customer_id) DO UPDATE SET
                usual_size = excluded.usual_size,
                fit_preference = excluded.fit_preference,
                updated_at = CURRENT_TIMESTAMP
        """, (customer_id, usual_size, fit_preference))
    return {"usual_size": usual_size, "fit_preference": fit_preference}


def delete_preferences(customer_id: str) -> bool:
    initialize_database()
    with _connection() as connection:
        cursor = connection.execute("DELETE FROM customer_preferences WHERE customer_id = ?", (customer_id,))
    return cursor.rowcount > 0


def record_feedback(customer_id: str, product_id: str, recommended_size: str, sentiment: str) -> None:
    initialize_database()
    with _connection() as connection:
        connection.execute("""
            INSERT INTO fit_feedback(customer_id, product_id, recommended_size, sentiment)
            VALUES (?, ?, ?, ?)
        """, (customer_id, product_id, recommended_size, sentiment))


def get_feedback_summary(customer_id: str, product_id: Optional[str] = None) -> dict:
    initialize_database()
    with _connection() as connection:
        if product_id:
            count = connection.execute(
                "SELECT COUNT(*) FROM fit_feedback WHERE customer_id = ? AND product_id = ?",
                (customer_id, product_id),
            ).fetchone()[0]
            latest = connection.execute(
                "SELECT sentiment FROM fit_feedback WHERE customer_id = ? AND product_id = ? ORDER BY id DESC LIMIT 1",
                (customer_id, product_id),
            ).fetchone()
        else:
            count = connection.execute(
                "SELECT COUNT(*) FROM fit_feedback WHERE customer_id = ?", (customer_id,)
            ).fetchone()[0]
            latest = connection.execute(
                "SELECT sentiment FROM fit_feedback WHERE customer_id = ? ORDER BY id DESC LIMIT 1",
                (customer_id,),
            ).fetchone()
    return {"feedback_count": count, "latest_feedback": latest["sentiment"] if latest else None}


def delete_customer_memory(customer_id: str) -> dict:
    initialize_database()
    with _connection() as connection:
        preferences = connection.execute("DELETE FROM customer_preferences WHERE customer_id = ?", (customer_id,)).rowcount
        feedback = connection.execute("DELETE FROM fit_feedback WHERE customer_id = ?", (customer_id,)).rowcount
    return {"preferences_deleted": preferences, "feedback_deleted": feedback}
