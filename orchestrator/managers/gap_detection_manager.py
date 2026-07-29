"""Clusters raw gap-box candidates (from OCRManager's shared YOLO call) into
coach-boundary events.

Same clustering shape as the proven services/sync_engine/engine.py
cluster_gaps() — consecutive detections within a radius are the same
physical gap seen across multiple frames — translated from trigger_id radius
to a timestamp window, since Phase-1 has no encoder/trigger_id, only capture
timestamps.

Gap Detection *owns* the boundary decision (rule 2): OCRManager only reports
candidates, this manager decides when a cluster is final and tells
EventManager to close/open an Event window.
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional

from core.logging_config import get_stage_logger
from core.models import GapCandidate

log = get_stage_logger("gap_detection")


class GapDetectionManager:
    def __init__(
        self,
        confidence_threshold: float,
        cluster_window_ms: float,
        on_boundary: Callable[[float, float], None],
    ):
        self._confidence_threshold = confidence_threshold
        self._cluster_window_ms = cluster_window_ms
        self._on_boundary = on_boundary
        self._lock = threading.Lock()
        self._pending: List[GapCandidate] = []
        self._boundaries_emitted = 0
        self._candidates_seen = 0
        self._candidates_rejected = 0

    @property
    def stats(self) -> dict:
        return {
            "candidates_seen": self._candidates_seen,
            "candidates_rejected_low_confidence": self._candidates_rejected,
            "boundaries_emitted": self._boundaries_emitted,
        }

    def submit_candidate(self, candidate: GapCandidate) -> None:
        self._candidates_seen += 1
        log.debug(
            "gap candidate received",
            extra={"extra_fields": {
                "camera": candidate.camera_name, "timestamp_ms": candidate.timestamp_ms,
                "confidence": candidate.confidence, "threshold": self._confidence_threshold,
                "accepted": candidate.confidence >= self._confidence_threshold,
            }},
        )
        if candidate.confidence < self._confidence_threshold:
            self._candidates_rejected += 1
            return

        with self._lock:
            if not self._pending:
                self._pending.append(candidate)
                return

            if candidate.timestamp_ms - self._pending[-1].timestamp_ms <= self._cluster_window_ms:
                self._pending.append(candidate)
                return

            cluster = self._pending
            self._pending = [candidate]

        self._finalize_cluster(cluster)

    def flush(self) -> None:
        """Call once the OCR camera stream has ended — finalizes any open cluster."""
        with self._lock:
            cluster = self._pending
            self._pending = []
        if cluster:
            self._finalize_cluster(cluster)

    def _finalize_cluster(self, cluster: List[GapCandidate]) -> None:
        best = max(cluster, key=lambda c: c.confidence)
        self._boundaries_emitted += 1
        log.info(
            "gap boundary detected",
            extra={"extra_fields": {
                "boundary_ts_ms": best.timestamp_ms, "confidence": best.confidence,
                "cluster_size": len(cluster),
            }},
        )
        try:
            self._on_boundary(best.timestamp_ms, best.confidence)
        except Exception:
            log.exception("EventManager boundary callback raised — boundary logged but not applied")
