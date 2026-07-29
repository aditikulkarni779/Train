"""D3 wiring — circuit breakers on the Postgres + Cloudinary clients."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402
import db_client  # noqa: E402
import cloudinary_client  # noqa: E402
from circuit_breaker import CircuitBreaker, CircuitOpen  # noqa: E402


def _fresh(name):
    return CircuitBreaker(name, failure_threshold=2, reset_timeout_s=30, clock=lambda: 0.0)


def test_db_breaker_opens_and_fails_fast(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(db_client, "_conn", None)
    br = _fresh("postgres")
    monkeypatch.setattr(db_client, "_DB_BREAKER", br)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise ConnectionError("db down")

    monkeypatch.setattr(db_client.psycopg2, "connect", boom)

    for _ in range(2):                       # threshold=2 -> opens
        with pytest.raises(ConnectionError):
            db_client.get_conn()
    assert br.state.value == "open"

    before = calls["n"]
    with pytest.raises(CircuitOpen):
        db_client.get_conn()                 # fast-fail, real connect NOT called
    assert calls["n"] == before


def test_cloudinary_breaker_opens_and_fails_fast(monkeypatch):
    br = _fresh("cloudinary")
    monkeypatch.setattr(cloudinary_client, "_CLOUDINARY_BREAKER", br)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise ConnectionError("cloudinary 500")

    monkeypatch.setattr(cloudinary_client.cloudinary.uploader, "upload", boom)

    for _ in range(2):
        with pytest.raises(ConnectionError):
            cloudinary_client.upload_frame("x.jpg", "folder")
    assert br.state.value == "open"

    before = calls["n"]
    with pytest.raises(CircuitOpen):
        cloudinary_client.upload_frame("x.jpg", "folder")
    assert calls["n"] == before             # hammering stopped while OPEN


def test_db_breaker_recovers_after_reset(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(db_client, "_conn", None)
    t = {"now": 0.0}
    br = CircuitBreaker("postgres", failure_threshold=1, reset_timeout_s=10, clock=lambda: t["now"])
    monkeypatch.setattr(db_client, "_DB_BREAKER", br)

    state = {"fail": True}

    class _FakeConn:
        closed = False

    def maybe(*a, **k):
        if state["fail"]:
            raise ConnectionError("db down")
        return _FakeConn()

    monkeypatch.setattr(db_client.psycopg2, "connect", maybe)

    with pytest.raises(ConnectionError):
        db_client.get_conn()
    assert br.state.value == "open"

    t["now"] = 11                            # past reset window -> half-open allowed
    state["fail"] = False
    conn = db_client.get_conn()              # probe succeeds -> closed
    assert conn is not None
    assert br.state.value == "closed"
