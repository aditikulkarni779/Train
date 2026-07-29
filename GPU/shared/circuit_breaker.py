"""
Circuit Breaker (D3) — protects external dependencies (Cloudinary, Postgres, SMTP)
so a failing dependency fails fast and degrades gracefully instead of cascading
hangs through the pipeline.

States:
    CLOSED     calls pass through; N consecutive failures -> OPEN
    OPEN       calls fail fast (CircuitOpen); after reset_timeout -> HALF_OPEN
    HALF_OPEN  one trial call; success -> CLOSED, failure -> OPEN

`call_with_fallback` lets callers buffer/skip on OPEN so confirmed events are not
lost during an outage (e.g. queue the write, return a placeholder).

Pure logic, clock injected — unit-tested without real dependencies.
"""
import logging
from enum import Enum

logger = logging.getLogger("circuit_breaker")


class CBState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(Exception):
    """Raised when a call is rejected because the breaker is OPEN."""


class CircuitBreaker:
    def __init__(self, name: str, *, failure_threshold: int = 5,
                 reset_timeout_s: float = 30.0, clock=None):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        import time as _t
        self.clock = clock or (lambda: _t.monotonic())
        self.state = CBState.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self.transitions: list[tuple] = []   # (from, to) history for observability

    def _set(self, to: CBState):
        if to != self.state:
            self.transitions.append((self.state, to))
            logger.info('{"event":"cb_transition","name":"%s","from":"%s","to":"%s"}',
                        self.name, self.state.value, to.value)
            self.state = to

    def _maybe_half_open(self, now: float):
        if self.state == CBState.OPEN and (now - self.opened_at) >= self.reset_timeout_s:
            self._set(CBState.HALF_OPEN)

    def allow(self) -> bool:
        """True if a call may proceed right now."""
        now = self.clock()
        self._maybe_half_open(now)
        return self.state != CBState.OPEN

    def _on_success(self):
        self.failures = 0
        if self.state in (CBState.HALF_OPEN, CBState.OPEN):
            self._set(CBState.CLOSED)

    def _on_failure(self):
        if self.state == CBState.HALF_OPEN:
            self.opened_at = self.clock()
            self._set(CBState.OPEN)
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = self.clock()
            self._set(CBState.OPEN)

    def call(self, fn, *args, **kwargs):
        """Run fn through the breaker. Raises CircuitOpen if OPEN; re-raises fn errors."""
        if not self.allow():
            raise CircuitOpen(f"{self.name} circuit open")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def call_with_fallback(self, fn, fallback, *args, **kwargs):
        """Like call(), but on OPEN or failure returns fallback(*args) instead of
        raising — use to buffer writes / serve degraded and never lose an event."""
        try:
            return self.call(fn, *args, **kwargs)
        except (CircuitOpen, Exception):  # noqa: BLE001 - degrade on any failure
            return fallback(*args, **kwargs)
