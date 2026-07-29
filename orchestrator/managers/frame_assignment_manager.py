"""Assigns Cam1/Cam2 processed frames to a bounded Event window using a
closed timestamp interval — EventStart <= ts <= EventEnd, never equality
(brief rule 5), same semantics as sync_engine.assign_frames_to_coaches but
timestamp-keyed instead of trigger_id-keyed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.logging_config import get_stage_logger
from core.models import EventWindow, ProcessedFrame
from managers.processed_frame_store import ProcessedFrameStore

log = get_stage_logger("frame_assignment")


@dataclass
class EventFrames:
    event_id: str
    frames_by_camera: Dict[str, List[ProcessedFrame]] = field(default_factory=dict)

    def count(self, camera_name: str) -> int:
        return len(self.frames_by_camera.get(camera_name, []))

    def total(self) -> int:
        return sum(len(v) for v in self.frames_by_camera.values())


class FrameAssignmentManager:
    def __init__(self, store: ProcessedFrameStore, component_camera_names: List[str]):
        self._store = store
        self._component_camera_names = component_camera_names

    def assign(self, event: EventWindow) -> EventFrames:
        end_ts = event.end_ts_ms if event.end_ts_ms is not None else float("inf")
        result = EventFrames(event_id=event.event_id)

        for camera_name in self._component_camera_names:
            frames = self._store.query_range(camera_name, event.start_ts_ms, end_ts)
            result.frames_by_camera[camera_name] = frames
            if not frames:
                event.warnings.append(f"no_frames_assigned:{camera_name}")
                log.warning(
                    "no frames found for camera in event window — marking data_unavailable",
                    extra={"extra_fields": {
                        "event_id": event.event_id, "camera": camera_name,
                        "start_ts_ms": event.start_ts_ms, "end_ts_ms": event.end_ts_ms,
                    }},
                )

        log.info(
            "frames assigned",
            extra={"extra_fields": {
                "event_id": event.event_id,
                "counts": {c: len(f) for c, f in result.frames_by_camera.items()},
            }},
        )
        return result
