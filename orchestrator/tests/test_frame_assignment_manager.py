import numpy as np

from core.models import EventStatus, EventWindow, FrameMessage, ProcessedFrame
from managers.frame_assignment_manager import FrameAssignmentManager
from managers.processed_frame_store import ProcessedFrameStore


def _frame(camera, seq, ts):
    source = FrameMessage(camera_name=camera, sequence_number=seq, timestamp_ms=ts,
                           image=np.zeros((4, 4, 3), dtype="uint8"))
    return ProcessedFrame(source=source, image=source.image, blur_score=100.0,
                           quality_score=0.9, brightness_mean=120.0)


def test_frames_within_closed_interval_are_assigned_inclusive_both_ends():
    store = ProcessedFrameStore()
    for ts in [900, 1000, 1500, 2000, 2100]:
        store.add(_frame("cam1", ts, ts))

    event = EventWindow(event_id="EVT_000001", coach_index=1, start_ts_ms=1000, end_ts_ms=2000)
    mgr = FrameAssignmentManager(store, ["cam1"])
    result = mgr.assign(event)

    assigned_ts = sorted(f.timestamp_ms for f in result.frames_by_camera["cam1"])
    assert assigned_ts == [1000, 1500, 2000]   # 900 and 2100 excluded — interval is closed, not open-ended


def test_empty_camera_marks_event_data_unavailable_but_does_not_raise():
    store = ProcessedFrameStore()
    event = EventWindow(event_id="EVT_000002", coach_index=2, start_ts_ms=0, end_ts_ms=100)
    mgr = FrameAssignmentManager(store, ["cam1", "cam2"])
    result = mgr.assign(event)

    assert result.total() == 0
    assert "no_frames_assigned:cam1" in event.warnings
    assert "no_frames_assigned:cam2" in event.warnings


def test_open_event_end_defaults_to_infinity():
    store = ProcessedFrameStore()
    store.add(_frame("cam1", 1, 5000))
    event = EventWindow(event_id="EVT_000003", coach_index=3, start_ts_ms=0, end_ts_ms=None,
                         status=EventStatus.OPEN)
    mgr = FrameAssignmentManager(store, ["cam1"])
    result = mgr.assign(event)
    assert result.count("cam1") == 1
