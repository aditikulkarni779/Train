"""D3 — circuit breaker state machine + fail-fast + fallback tests."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402
from circuit_breaker import CircuitBreaker, CircuitOpen, CBState  # noqa: E402


def _flaky(fail: bool):
    def fn():
        if fail:
            raise ConnectionError("dep down")
        return "ok"
    return fn


def test_opens_after_threshold_failures():
    cb = CircuitBreaker("cloudinary", failure_threshold=3, reset_timeout_s=30,
                        clock=lambda: 0.0)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            cb.call(_flaky(True))
    assert cb.state == CBState.OPEN


def test_open_fails_fast_without_calling():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ConnectionError("dep down")

    t = {"now": 0.0}
    cb = CircuitBreaker("pg", failure_threshold=1, reset_timeout_s=30, clock=lambda: t["now"])
    with pytest.raises(ConnectionError):
        cb.call(fn)                 # trips OPEN
    assert cb.state == CBState.OPEN
    before = calls["n"]
    with pytest.raises(CircuitOpen):
        cb.call(fn)                 # rejected fast, fn NOT invoked
    assert calls["n"] == before


def test_half_open_then_close_on_success():
    t = {"now": 0.0}
    cb = CircuitBreaker("pg", failure_threshold=1, reset_timeout_s=10, clock=lambda: t["now"])
    with pytest.raises(ConnectionError):
        cb.call(_flaky(True))
    assert cb.state == CBState.OPEN
    t["now"] = 11                   # past reset timeout
    assert cb.allow() is True       # -> HALF_OPEN
    assert cb.state == CBState.HALF_OPEN
    assert cb.call(_flaky(False)) == "ok"
    assert cb.state == CBState.CLOSED


def test_half_open_failure_reopens():
    t = {"now": 0.0}
    cb = CircuitBreaker("pg", failure_threshold=1, reset_timeout_s=10, clock=lambda: t["now"])
    with pytest.raises(ConnectionError):
        cb.call(_flaky(True))
    t["now"] = 11
    cb.allow()                      # HALF_OPEN
    with pytest.raises(ConnectionError):
        cb.call(_flaky(True))       # trial fails -> OPEN again
    assert cb.state == CBState.OPEN


def test_success_resets_failure_count():
    cb = CircuitBreaker("pg", failure_threshold=3, clock=lambda: 0.0)
    with pytest.raises(ConnectionError):
        cb.call(_flaky(True))
    with pytest.raises(ConnectionError):
        cb.call(_flaky(True))
    cb.call(_flaky(False))          # success resets streak
    assert cb.failures == 0
    assert cb.state == CBState.CLOSED


def test_fallback_buffers_event_during_outage():
    # dependency outage must NOT lose the event: fallback buffers it
    buffer = []
    t = {"now": 0.0}
    cb = CircuitBreaker("cloudinary", failure_threshold=1, reset_timeout_s=120,
                        clock=lambda: t["now"])

    def upload():
        raise ConnectionError("cloudinary 500")

    def buffer_it():
        buffer.append("event")
        return "buffered"

    # first call fails (real error) but is buffered
    assert cb.call_with_fallback(upload, buffer_it) == "buffered"
    # now OPEN -> still buffered, fails fast, nothing lost
    for _ in range(5):
        assert cb.call_with_fallback(upload, buffer_it) == "buffered"
    assert len(buffer) == 6
    assert cb.state == CBState.OPEN


def test_transitions_recorded():
    t = {"now": 0.0}
    cb = CircuitBreaker("pg", failure_threshold=1, reset_timeout_s=5, clock=lambda: t["now"])
    with pytest.raises(ConnectionError):
        cb.call(_flaky(True))       # CLOSED->OPEN
    t["now"] = 6
    cb.allow()                      # OPEN->HALF_OPEN
    cb.call(_flaky(False))          # HALF_OPEN->CLOSED
    assert (CBState.CLOSED, CBState.OPEN) in cb.transitions
    assert (CBState.OPEN, CBState.HALF_OPEN) in cb.transitions
    assert (CBState.HALF_OPEN, CBState.CLOSED) in cb.transitions
