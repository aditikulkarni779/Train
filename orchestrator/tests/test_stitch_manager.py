import numpy as np

from core.models import FrameMessage, ProcessedFrame
from managers.stitch_manager import _pair_frames, StitchManager
from managers.frame_assignment_manager import EventFrames


def _frame(camera, ts, h=100, w=80):
    source = FrameMessage(camera_name=camera, sequence_number=int(ts), timestamp_ms=ts,
                           image=np.full((h, w, 3), 200, dtype="uint8"))
    return ProcessedFrame(source=source, image=source.image, blur_score=100.0,
                           quality_score=0.9, brightness_mean=120.0)


def test_pair_frames_matches_nearest_within_epsilon():
    top = [_frame("cam1", 1000), _frame("cam1", 2000)]
    bottom = [_frame("cam2", 1050), _frame("cam2", 2900)]  # 2900 outside epsilon of 2000
    pairs = _pair_frames(top, bottom, epsilon_ms=150.0)
    assert len(pairs) == 1
    assert pairs[0][0].timestamp_ms == 1000
    assert pairs[0][1].timestamp_ms == 1050


def test_pair_frames_does_not_reuse_the_same_bottom_frame_twice():
    top = [_frame("cam1", 1000), _frame("cam1", 1010)]
    bottom = [_frame("cam2", 1005)]
    pairs = _pair_frames(top, bottom, epsilon_ms=150.0)
    assert len(pairs) == 1


def test_stitch_event_returns_empty_when_one_camera_has_no_frames():
    mgr = StitchManager("cam1", "cam2", pair_epsilon_ms=150.0)
    event_frames = EventFrames(event_id="EVT_000001", frames_by_camera={
        "cam1": [_frame("cam1", 1000)], "cam2": [],
    })
    assert mgr.stitch_event(event_frames) == []


def test_stitch_event_produces_calibrated_composite_of_expected_height():
    mgr = StitchManager("cam1", "cam2", pair_epsilon_ms=150.0, overlap_ratio=0.15)
    event_frames = EventFrames(event_id="EVT_000001", frames_by_camera={
        "cam1": [_frame("cam1", 1000, h=100, w=80)],
        "cam2": [_frame("cam2", 1010, h=100, w=80)],
    })
    results = mgr.stitch_event(event_frames)
    assert len(results) == 1
    assert results[0].method == "calibrated"
    overlap_px = round(0.15 * 100)
    assert results[0].image.shape == (2 * 100 - overlap_px, 80, 3)


def test_stitch_event_labels_method_when_feature_refinement_enabled():
    mgr = StitchManager("cam1", "cam2", pair_epsilon_ms=150.0, use_feature_refinement=True)
    event_frames = EventFrames(event_id="EVT_000001", frames_by_camera={
        "cam1": [_frame("cam1", 1000)], "cam2": [_frame("cam2", 1010)],
    })
    results = mgr.stitch_event(event_frames)
    assert results[0].method == "calibrated_features"


def test_stitch_event_skips_pair_on_size_mismatch_without_crashing():
    mgr = StitchManager("cam1", "cam2", pair_epsilon_ms=150.0)
    event_frames = EventFrames(event_id="EVT_000001", frames_by_camera={
        "cam1": [_frame("cam1", 1000, h=100, w=80)],
        "cam2": [_frame("cam2", 1010, h=90, w=80)],   # mismatched height
    })
    assert mgr.stitch_event(event_frames) == []
