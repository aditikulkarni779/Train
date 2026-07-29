"""Tier-1 capture stand-in: one thread per camera reads a video file with
OpenCV and pushes timestamped frames into that camera's RingBufferManager.

Real hardware (PTP/encoder/GigE Vision, ADS Ch. 3) is out of scope for this
phase — decision was to simulate live feed from pre-recorded video per camera,
reusing the same VideoCapture-driven extraction pattern already proven in
services/frame_extractor/server.py, restructured to run in-process instead of
polling Postgres.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, Optional

import cv2

from core.logging_config import get_stage_logger
from core.models import CameraConfig, FrameMessage
from managers.ring_buffer_manager import RingBufferManager

log = get_stage_logger("frame_capture")


class CameraDisconnectedError(RuntimeError):
    pass


class FrameCaptureManager:
    def __init__(
        self,
        camera: CameraConfig,
        ring_buffer: RingBufferManager,
        session_start_ts_ms: float,
        on_camera_finished: Optional[Callable[[str], None]] = None,
    ):
        self._camera = camera
        self._ring_buffer = ring_buffer
        self._session_start_ts_ms = session_start_ts_ms
        self._on_finished = on_camera_finished
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._frames_captured = 0

    @property
    def frames_captured(self) -> int:
        return self._frames_captured

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self._camera.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        cap = cv2.VideoCapture(self._camera.source)
        if not cap.isOpened():
            log.error(
                "camera disconnected / cannot open source",
                extra={"extra_fields": {"camera": self._camera.name, "source": self._camera.source}},
            )
            self._ring_buffer.close()
            if self._on_finished:
                self._on_finished(self._camera.name)
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or self._camera.target_fps
        frame_interval = max(1, round(source_fps / self._camera.target_fps))
        raw_index = 0

        log.info(
            "capture started",
            extra={"extra_fields": {
                "camera": self._camera.name, "source_fps": source_fps,
                "target_fps": self._camera.target_fps, "sample_every": frame_interval,
            }},
        )

        try:
            while not self._stop_flag.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                if raw_index % frame_interval == 0:
                    timestamp_ms = self._session_start_ts_ms + (
                        self._frames_captured * (1000.0 / self._camera.target_fps)
                    )
                    frame = FrameMessage(
                        camera_name=self._camera.name,
                        sequence_number=self._frames_captured,
                        timestamp_ms=timestamp_ms,
                        image=image,
                    )
                    self._ring_buffer.put(frame)
                    self._frames_captured += 1
                raw_index += 1
        except Exception:
            log.exception(
                "capture loop crashed",
                extra={"extra_fields": {"camera": self._camera.name}},
            )
        finally:
            cap.release()
            self._ring_buffer.close()
            log.info(
                "capture finished",
                extra={"extra_fields": {
                    "camera": self._camera.name, "frames_captured": self._frames_captured,
                }},
            )
            if self._on_finished:
                self._on_finished(self._camera.name)
