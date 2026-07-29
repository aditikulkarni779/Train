from core.models import OcrObservation
from managers.event_manager import EventManager


def test_on_identity_resolved_fires_once_per_higher_confidence_result():
    resolved = []
    mgr = EventManager(session_start_ts_ms=0.0, on_event_bounded=lambda e: None,
                        on_identity_resolved=resolved.append)
    mgr.on_gap_boundary(1000.0, confidence=0.9)  # EVT_000001: [0, 1000]

    mgr.enrich_with_ocr(OcrObservation(500.0, "12345", 0.7, 1, True))
    assert len(resolved) == 1
    assert resolved[0].coach_number == "12345"

    mgr.enrich_with_ocr(OcrObservation(600.0, "99999", 0.2, 1, True))  # lower confidence
    assert len(resolved) == 1  # no second callback — nothing changed


def test_on_identity_resolved_not_called_when_coach_number_missing():
    resolved = []
    mgr = EventManager(session_start_ts_ms=0.0, on_event_bounded=lambda e: None,
                        on_identity_resolved=resolved.append)
    mgr.enrich_with_ocr(OcrObservation(500.0, None, 0.0, 0, False))
    assert resolved == []
