"""Per-camera bounded ring buffer: FrameCaptureManager producer, PreprocessingManager consumer.

Bounded + drop-oldest-with-counter (never silent, never unbounded) — mirrors the
ADS queue design rules (Ch. 4.4.1) and the existing repo's own bounded-queue
precedent (backend/src/queue — Redis Streams with maxlen/DLQ).
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Optional

from core.logging_config import get_stage_logger
from core.models import FrameMessage

log = get_stage_logger("ring_buffer")


class RingBufferManager:
    def __init__(self, camera_name: str, max_size: int):
        self._camera_name = camera_name
        self._max_size = max_size
        self._buffer: deque[FrameMessage] = deque()
        self._lock = threading.Condition()
        self._dropped_count = 0
        self._closed = False

    def put(self, frame: FrameMessage) -> None:
        with self._lock:
            if len(self._buffer) >= self._max_size:
                self._buffer.popleft()
                self._dropped_count += 1
                log.warning(
                    "ring buffer overflow — dropping oldest frame",
                    extra={"extra_fields": {
                        "camera": self._camera_name,
                        "dropped_total": self._dropped_count,
                    }},
                )
            self._buffer.append(frame)
            self._lock.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[FrameMessage]:
        with self._lock:
            if not self._buffer and not self._closed:
                self._lock.wait(timeout=timeout)
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._buffer)
