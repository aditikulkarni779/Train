"""Shared data contracts passed between managers.

Kept as plain dataclasses (no ORM/DB coupling) so the orchestrator stays a
standalone process — it does not touch backend/Prisma or Postgres.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class CameraType(str, Enum):
    OCR = "ocr"
    COMPONENT = "component"


class EventStatus(str, Enum):
    OPEN = "OPEN"                  # window started, end boundary not yet known
    BOUNDED = "BOUNDED"            # gap boundary found, window closed
    IDENTITY_PENDING = "IDENTITY_PENDING"
    ASSIGNED = "ASSIGNED"          # frames assigned from ring/processed buffers
    STITCHED = "STITCHED"
    DETECTED = "DETECTED"
    STORED = "STORED"
    FAILED_PARTIAL = "FAILED_PARTIAL"  # completed with a degraded/missing stage


@dataclass
class CameraConfig:
    name: str
    source: str                 # video file path (Phase-1) or device index / RTSP url (future)
    camera_type: CameraType
    target_fps: float = 5.0
    # Per-camera resize override — falls back to PipelineConfig.resize_width/height
    # when unset. Needed because the OCR camera and the component cameras can be
    # (and in practice are) wildly different native resolutions/aspect ratios; one
    # global resize target would distort whichever camera doesn't match it.
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None


@dataclass
class FrameMessage:
    """One captured frame, timestamped at the moment it is read off the source.

    timestamp_ms is a synthetic monotonic capture clock (session start + n/fps),
    standing in for the ADS's PTP-disciplined hardware timestamp (Ch. 3.3.2) —
    real camera hardware is not available in this phase. This is a scoped
    stand-in, not a redesign: the interval-assignment logic downstream is
    timestamp-based either way, so swapping in real PTP timestamps later is a
    FrameCaptureManager-only change.
    """
    camera_name: str
    sequence_number: int
    timestamp_ms: float
    image: np.ndarray
    frame_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    captured_wall_time: float = field(default_factory=time.time)


@dataclass
class ProcessedFrame:
    source: FrameMessage
    image: np.ndarray
    blur_score: float
    quality_score: float
    brightness_mean: float

    @property
    def camera_name(self) -> str:
        return self.source.camera_name

    @property
    def timestamp_ms(self) -> float:
        return self.source.timestamp_ms

    @property
    def frame_id(self) -> str:
        return self.source.frame_id


@dataclass
class GapCandidate:
    camera_name: str
    timestamp_ms: float
    confidence: float
    bbox_xyxy: list


@dataclass
class OcrObservation:
    timestamp_ms: float
    coach_number: Optional[str]
    confidence: float
    pass_used: int
    roi_used: bool


@dataclass
class EventWindow:
    event_id: str
    coach_index: int
    start_ts_ms: float
    end_ts_ms: Optional[float] = None
    status: EventStatus = EventStatus.OPEN
    coach_number: Optional[str] = None
    ocr_confidence: float = 0.0
    identity_state: str = "pending"     # pending | resolved | unresolved
    warnings: list = field(default_factory=list)

    def contains(self, timestamp_ms: float) -> bool:
        """Interval membership — ADS rule: EventStart <= ts <= EventEnd, never equality."""
        if self.end_ts_ms is None:
            return timestamp_ms >= self.start_ts_ms
        return self.start_ts_ms <= timestamp_ms <= self.end_ts_ms
