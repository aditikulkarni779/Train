from core.models import EventStatus, OcrObservation
from managers.event_manager import EventManager


def test_first_event_opens_at_session_start():
    mgr = EventManager(session_start_ts_ms=0.0, on_event_bounded=lambda e: None)
    assert mgr._current.event_id == "EVT_000001"
    assert mgr._current.start_ts_ms == 0.0
    assert mgr._current.status == EventStatus.OPEN


def test_gap_boundary_closes_current_and_opens_next_with_sequential_ids():
    bounded = []
    mgr = EventManager(session_start_ts_ms=0.0, on_event_bounded=bounded.append)
    mgr.on_gap_boundary(1000.0, confidence=0.9)

    assert len(bounded) == 1
    closed = bounded[0]
    assert closed.event_id == "EVT_000001"
    assert closed.end_ts_ms == 1000.0
    assert closed.status == EventStatus.BOUNDED

    assert mgr._current.event_id == "EVT_000002"
    assert mgr._current.start_ts_ms == 1000.0


def test_ocr_enriches_the_event_whose_interval_contains_the_timestamp():
    mgr = EventManager(session_start_ts_ms=0.0, on_event_bounded=lambda e: None)
    mgr.on_gap_boundary(1000.0, confidence=0.9)  # EVT_000001: [0, 1000]

    mgr.enrich_with_ocr(OcrObservation(
        timestamp_ms=500.0, coach_number="12345", confidence=0.8, pass_used=1, roi_used=True,
    ))

    evt1 = mgr._events["EVT_000001"]
    assert evt1.coach_number == "12345"
    assert evt1.identity_state == "resolved"


def test_lower_confidence_ocr_does_not_overwrite_a_better_result():
    mgr = EventManager(session_start_ts_ms=0.0, on_event_bounded=lambda e: None)
    mgr.on_gap_boundary(1000.0, confidence=0.9)

    mgr.enrich_with_ocr(OcrObservation(500.0, "12345", 0.9, 1, True))
    mgr.enrich_with_ocr(OcrObservation(600.0, "99999", 0.2, 1, True))

    assert mgr._events["EVT_000001"].coach_number == "12345"


def test_ocr_outside_any_event_interval_is_dropped_without_crashing():
    mgr = EventManager(session_start_ts_ms=1000.0, on_event_bounded=lambda e: None)
    mgr.enrich_with_ocr(OcrObservation(500.0, "12345", 0.9, 1, True))  # before session start
    assert mgr._events["EVT_000001"].coach_number is None


def test_mark_unresolved_if_no_identity_flags_missing_coach_number():
    mgr = EventManager(session_start_ts_ms=0.0, on_event_bounded=lambda e: None)
    mgr.on_gap_boundary(1000.0, confidence=0.9)
    evt1 = mgr._events["EVT_000001"]

    mgr.mark_unresolved_if_no_identity(evt1)
    assert evt1.identity_state == "unresolved"
    assert "identity_unresolved_ocr_missing_or_failed" in evt1.warnings


def test_close_final_window_only_applies_to_still_open_window():
    mgr = EventManager(session_start_ts_ms=0.0, on_event_bounded=lambda e: None)
    mgr.on_gap_boundary(1000.0, confidence=0.9)
    mgr.close_final_window(2000.0)

    evt2 = mgr._events["EVT_000002"]
    assert evt2.end_ts_ms == 2000.0
    assert "closed_by_stream_end_not_gap_boundary" in evt2.warnings
