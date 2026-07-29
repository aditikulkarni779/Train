"""Creates and owns Event windows. Gap Detection is the only thing that
closes a window and opens the next (rule 2); OCR only ever enriches an
existing Event's metadata (rule 3 — coach_number is never a key, `event_id`
like EVT_000001 is the only identifier used downstream).

Frame assignment / stitching / detection / storage for a just-bounded event
run through `on_event_bounded`, off the calling thread, so a slow downstream
chain never stalls gap detection for the next coach.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

from core.logging_config import get_stage_logger
from core.models import EventStatus, EventWindow, OcrObservation

log = get_stage_logger("event_manager")


class EventManager:
    def __init__(
        self,
        session_start_ts_ms: float,
        on_event_bounded: Callable[[EventWindow], None],
        on_event_updated: Optional[Callable[[EventWindow], None]] = None,
        on_identity_resolved: Optional[Callable[[EventWindow], None]] = None,
        late_ocr_grace_period_ms: float = 5000.0,
    ):
        self._on_event_bounded = on_event_bounded
        self._on_event_updated = on_event_updated
        self._on_identity_resolved = on_identity_resolved
        self._late_ocr_grace_period_ms = late_ocr_grace_period_ms
        self._lock = threading.Lock()
        self._events: Dict[str, EventWindow] = {}
        self._ordered_ids: List[str] = []
        self._next_index = 1
        self._sequence = 0

        self._current: EventWindow = self._open_new_window(session_start_ts_ms)

    def _next_event_id(self) -> str:
        self._sequence += 1
        return f"EVT_{self._sequence:06d}"

    def _open_new_window(self, start_ts_ms: float) -> EventWindow:
        event = EventWindow(
            event_id=self._next_event_id(),
            coach_index=self._next_index,
            start_ts_ms=start_ts_ms,
        )
        self._next_index += 1
        self._events[event.event_id] = event
        self._ordered_ids.append(event.event_id)
        log.info(
            "event window opened",
            extra={"extra_fields": {"event_id": event.event_id, "start_ts_ms": start_ts_ms}},
        )
        return event

    def on_gap_boundary(self, boundary_ts_ms: float, confidence: float) -> None:
        with self._lock:
            closed = self._current
            closed.end_ts_ms = boundary_ts_ms
            closed.status = EventStatus.BOUNDED
            self._current = self._open_new_window(boundary_ts_ms)

        log.info(
            "event window bounded",
            extra={"extra_fields": {
                "event_id": closed.event_id, "start_ts_ms": closed.start_ts_ms,
                "end_ts_ms": closed.end_ts_ms, "boundary_confidence": confidence,
            }},
        )
        self._safe(self._on_event_bounded, closed)

    def close_final_window(self, end_ts_ms: float) -> None:
        """Called at end-of-stream — the last open window has no gap boundary
        after it, so it is closed on stream end instead."""
        with self._lock:
            closed = self._current
            if closed.status != EventStatus.OPEN:
                return
            closed.end_ts_ms = end_ts_ms
            closed.status = EventStatus.BOUNDED
            closed.warnings.append("closed_by_stream_end_not_gap_boundary")
        log.info(
            "final event window closed at stream end (no trailing gap boundary)",
            extra={"extra_fields": {"event_id": closed.event_id}},
        )
        self._safe(self._on_event_bounded, closed)

    def enrich_with_ocr(self, observation: OcrObservation) -> None:
        with self._lock:
            target = self._find_event_for_timestamp(observation.timestamp_ms)

        if target is None:
            log.warning(
                "OCR observation matches no known event window — dropped",
                extra={"extra_fields": {"timestamp_ms": observation.timestamp_ms}},
            )
            return

        if not observation.coach_number:
            return  # nothing to attach; identity stays pending/unresolved

        with self._lock:
            is_late = target.status != EventStatus.OPEN and target.status != EventStatus.BOUNDED
            updated = observation.confidence > target.ocr_confidence
            if updated:
                target.coach_number = observation.coach_number
                target.ocr_confidence = observation.confidence
                target.identity_state = "resolved"

        if not updated:
            return  # a higher-confidence result already won; nothing changed

        log.info(
            "event metadata enriched from OCR" + (" (late)" if is_late else ""),
            extra={"extra_fields": {
                "event_id": target.event_id, "coach_number": target.coach_number,
                "confidence": target.ocr_confidence,
            }},
        )
        if self._on_identity_resolved:
            self._safe(self._on_identity_resolved, target)
        if is_late and self._on_event_updated:
            self._safe(self._on_event_updated, target)

    def mark_unresolved_if_no_identity(self, event: EventWindow) -> None:
        if event.coach_number is None:
            event.identity_state = "unresolved"
            event.warnings.append("identity_unresolved_ocr_missing_or_failed")
            log.warning(
                "event has no resolved coach identity — flagged identity_unresolved",
                extra={"extra_fields": {"event_id": event.event_id}},
            )

    def _find_event_for_timestamp(self, ts_ms: float) -> Optional[EventWindow]:
        for event_id in reversed(self._ordered_ids):
            event = self._events[event_id]
            if event.contains(ts_ms):
                return event
        return None

    @staticmethod
    def _safe(fn, arg) -> None:
        try:
            fn(arg)
        except Exception:
            log.exception("event callback raised — continuing")
