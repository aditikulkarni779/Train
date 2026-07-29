"""Shared psycopg2 connection helper for Python services writing to PostgreSQL.

Wrapped in a circuit breaker (D3): once Postgres fails repeatedly the breaker
opens and get_conn() fails fast (CircuitOpen) instead of every caller blocking on
a connect timeout, then probes for recovery after the reset window.
"""
import os
import psycopg2
import psycopg2.extras

from circuit_breaker import CircuitBreaker

_conn = None

_DB_BREAKER = CircuitBreaker(
    "postgres",
    failure_threshold=int(os.environ.get("DB_CB_THRESHOLD", "5")),
    reset_timeout_s=float(os.environ.get("DB_CB_RESET", "30")),
)


def _connect():
    return psycopg2.connect(
        os.environ["DATABASE_URL"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def get_conn():
    """Cached module-level connection (reconnects if closed), through the breaker."""
    global _conn
    if _conn is None or _conn.closed:
        _conn = _DB_BREAKER.call(_connect)   # raises CircuitOpen when breaker is OPEN
    return _conn


def connect():
    """Fresh connection through the breaker — for services that manage their own
    per-request connect/close lifecycle. Raises CircuitOpen when breaker is OPEN."""
    return _DB_BREAKER.call(_connect)
