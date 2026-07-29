"""Thread-safe, timestamp-sorted store per camera — the "Processed Frame Buffer".

FrameAssignmentManager queries this with a timestamp interval
(EventStart <= ts <= EventEnd); insertion order from parallel preprocessing
workers is not capture order, so frames are kept sorted by timestamp via
bisect rather than relying on append order.
"""
from __future__ import annotations

import bisect
import threading
from typing import Dict, List

from core.models import ProcessedFrame


class ProcessedFrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._by_camera: Dict[str, List[ProcessedFrame]] = {}
        self._timestamps: Dict[str, List[float]] = {}

    def add(self, frame: ProcessedFrame) -> None:
        with self._lock:
            ts_list = self._timestamps.setdefault(frame.camera_name, [])
            frames = self._by_camera.setdefault(frame.camera_name, [])
            idx = bisect.bisect_left(ts_list, frame.timestamp_ms)
            ts_list.insert(idx, frame.timestamp_ms)
            frames.insert(idx, frame)

    def query_range(self, camera_name: str, start_ts: float, end_ts: float) -> List[ProcessedFrame]:
        with self._lock:
            ts_list = self._timestamps.get(camera_name, [])
            frames = self._by_camera.get(camera_name, [])
            lo = bisect.bisect_left(ts_list, start_ts)
            hi = bisect.bisect_right(ts_list, end_ts)
            return list(frames[lo:hi])

    def latest_timestamp(self, camera_name: str) -> float:
        with self._lock:
            ts_list = self._timestamps.get(camera_name, [])
            return ts_list[-1] if ts_list else 0.0

    def count(self, camera_name: str) -> int:
        with self._lock:
            return len(self._by_camera.get(camera_name, []))
