from core.models import GapCandidate
from managers.gap_detection_manager import GapDetectionManager


def _candidate(ts, conf=0.9):
    return GapCandidate(camera_name="ocr_cam", timestamp_ms=ts, confidence=conf, bbox_xyxy=[0, 0, 10, 10])


def test_low_confidence_candidates_are_rejected():
    boundaries = []
    mgr = GapDetectionManager(0.5, 800.0, on_boundary=lambda ts, c: boundaries.append(ts))
    mgr.submit_candidate(_candidate(100, conf=0.1))
    mgr.flush()
    assert boundaries == []
    assert mgr.stats["candidates_rejected_low_confidence"] == 1


def test_consecutive_candidates_within_window_cluster_into_one_boundary():
    boundaries = []
    mgr = GapDetectionManager(0.4, 800.0, on_boundary=lambda ts, c: boundaries.append((ts, c)))
    mgr.submit_candidate(_candidate(1000, conf=0.6))
    mgr.submit_candidate(_candidate(1200, conf=0.9))   # within 800ms — same cluster
    mgr.submit_candidate(_candidate(1600, conf=0.5))   # within 800ms of 1200 — same cluster
    mgr.submit_candidate(_candidate(3000, conf=0.7))   # >800ms after 1600 — new cluster, closes first
    assert len(boundaries) == 1
    assert boundaries[0] == (1200, 0.9)  # highest-confidence candidate in the cluster wins
    assert mgr.stats["boundaries_emitted"] == 1


def test_flush_finalizes_a_pending_cluster_at_stream_end():
    boundaries = []
    mgr = GapDetectionManager(0.4, 800.0, on_boundary=lambda ts, c: boundaries.append(ts))
    mgr.submit_candidate(_candidate(500, conf=0.8))
    assert boundaries == []
    mgr.flush()
    assert boundaries == [500]


def test_far_apart_candidates_form_separate_boundaries():
    boundaries = []
    mgr = GapDetectionManager(0.4, 800.0, on_boundary=lambda ts, c: boundaries.append(ts))
    mgr.submit_candidate(_candidate(1000))
    mgr.submit_candidate(_candidate(5000))
    mgr.submit_candidate(_candidate(9000))
    mgr.flush()
    assert boundaries == [1000, 5000, 9000]
