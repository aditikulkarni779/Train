"""Generates the Event folder tree, event.json, metadata.json, OCR.json,
per-camera frames, stitched images, and detection.json.

Folder naming: the session folder starts as `session_name` (config) and the
event folder starts as `EVT_NNNNNN` (its stable identity), because neither
train number nor coach number is known yet at creation time — OCR resolves
later, sometimes after the folder is already written to. When OCR resolves:
  - the event folder is renamed on disk to the coach number
  - the FIRST coach number resolved in the whole session also renames the
    session folder (proxy for "train number" — there is no separate
    locomotive-only detector; the same OCR signal is reused for both,
    documented as a known limitation in README.md)

`event_id` is never lost even after rename — it stays inside event.json /
metadata.json, so traceability holds regardless of what the folder is
currently called.

All path lookups AND all disk writes for a given event go through
`self._lock` (RLock — same thread can re-enter, e.g. event_dir() called from
inside save_event()). This is deliberately coarse-grained: a rename must
never land between two write calls of the same save_event(), and save_event
for one event must never race a rename of a *different* event that happens
to share the session-level rename. Correctness over throughput here — disk
I/O was never the bottleneck.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Dict, List, Optional

import cv2

from core.logging_config import get_stage_logger
from core.models import EventWindow
from managers.detection_manager import DetectionResult
from managers.frame_assignment_manager import EventFrames
from managers.stitch_manager import StitchedImage

log = get_stage_logger("storage_manager")

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_\-]")


def _sanitize(value: str) -> str:
    """OCR output is already digit-only (train_number_filter.py regex), but
    this is the folder-naming boundary — defend it independently."""
    return _UNSAFE_CHARS.sub("_", str(value)) or "UNKNOWN"


class StorageFailure(RuntimeError):
    pass


class StorageManager:
    def __init__(self, session_name: str, output_root: str):
        self._output_root = output_root
        self._lock = threading.RLock()
        self._session_dir = os.path.join(output_root, session_name)
        self._session_train_resolved = False
        self._event_dirs: Dict[str, str] = {}
        os.makedirs(self._session_dir, exist_ok=True)

    def event_dir(self, event_id: str) -> str:
        with self._lock:
            if event_id not in self._event_dirs:
                self._event_dirs[event_id] = os.path.join(self._session_dir, event_id)
            return self._event_dirs[event_id]

    # ── Rename hooks — called by PipelineCoordinator when OCR resolves ─────

    def rename_session_for_train_number(self, train_number: str) -> None:
        """First-ever resolved coach number in the session becomes the train
        label. Only fires once — later resolutions never rename the session
        again, even if a later coach's number would also have qualified."""
        with self._lock:
            if self._session_train_resolved:
                return
            self._session_train_resolved = True

            new_dir = os.path.join(self._output_root, _sanitize(train_number))
            if new_dir == self._session_dir:
                return

            new_dir = self._dedupe_path(new_dir)
            try:
                if os.path.isdir(self._session_dir):
                    os.rename(self._session_dir, new_dir)
                else:
                    os.makedirs(new_dir, exist_ok=True)
                old_dir = self._session_dir
                self._session_dir = new_dir
                for event_id, path in list(self._event_dirs.items()):
                    rel = os.path.relpath(path, old_dir)
                    self._event_dirs[event_id] = os.path.normpath(os.path.join(new_dir, rel))
                log.info(
                    "session folder renamed to train number",
                    extra={"extra_fields": {"train_number": train_number, "new_dir": new_dir}},
                )
            except Exception:
                log.exception(
                    "session rename failed — continuing under original session name",
                    extra={"extra_fields": {"train_number": train_number}},
                )
                self._session_train_resolved = False  # allow a later attempt

    def rename_event_for_coach_number(self, event_id: str, coach_number: str) -> None:
        with self._lock:
            old_dir = self.event_dir(event_id)
            new_dir = os.path.join(self._session_dir, _sanitize(coach_number))
            if new_dir == old_dir:
                return
            new_dir = self._dedupe_path(new_dir, exclude=old_dir)

            try:
                if os.path.isdir(old_dir):
                    os.rename(old_dir, new_dir)
                self._event_dirs[event_id] = new_dir
                log.info(
                    "event folder renamed to coach number",
                    extra={"extra_fields": {
                        "event_id": event_id, "coach_number": coach_number, "new_dir": new_dir,
                    }},
                )
            except Exception:
                log.exception(
                    "event rename failed — continuing under previous folder name",
                    extra={"extra_fields": {"event_id": event_id, "coach_number": coach_number}},
                )

    @staticmethod
    def _dedupe_path(path: str, exclude: Optional[str] = None) -> str:
        """Two coaches misread (or genuinely sharing) the same number must
        never silently collide — append a numeric suffix instead of
        overwriting one coach's folder with another's."""
        if not os.path.exists(path) or path == exclude:
            return path
        base = path
        n = 2
        candidate = f"{base}_dup{n}"
        while os.path.exists(candidate):
            n += 1
            candidate = f"{base}_dup{n}"
        log.warning(
            "folder name collision — deduplicated",
            extra={"extra_fields": {"requested": path, "assigned": candidate}},
        )
        return candidate

    # ── Writes ──────────────────────────────────────────────────────────────

    def save_event(
        self,
        event: EventWindow,
        event_frames: EventFrames,
        stitched_images: List[StitchedImage],
        detection_results: List[DetectionResult],
        processing_time_s: float,
    ) -> bool:
        with self._lock:
            event_dir = self.event_dir(event.event_id)
            try:
                os.makedirs(event_dir, exist_ok=True)
                for camera_name, frames in event_frames.frames_by_camera.items():
                    cam_dir = os.path.join(event_dir, camera_name.capitalize())
                    os.makedirs(cam_dir, exist_ok=True)
                    for frame in frames:
                        path = os.path.join(cam_dir, f"frame_{frame.source.sequence_number:06d}.jpg")
                        cv2.imwrite(path, frame.image)

                stitched_dir = os.path.join(event_dir, "Stitched")
                os.makedirs(stitched_dir, exist_ok=True)
                for i, stitched in enumerate(stitched_images):
                    path = os.path.join(stitched_dir, f"stitched_{i:04d}.jpg")
                    cv2.imwrite(path, stitched.image)

                detection_dir = os.path.join(event_dir, "Detection")
                os.makedirs(detection_dir, exist_ok=True)
                detection_payload = {
                    "event_id": event.event_id,
                    "detection_count": sum(len(r.detections) for r in detection_results),
                    "frames": [
                        {"timestamp_ms": r.timestamp_ms, "detections": r.detections}
                        for r in detection_results
                    ],
                }
                with open(os.path.join(detection_dir, "detection.json"), "w", encoding="utf-8") as f:
                    json.dump(detection_payload, f, indent=2)

                self._write_event_json(event_dir, event)
                self._write_ocr_json(event_dir, event)
                self._write_metadata_json(event_dir, event, event_frames, stitched_images,
                                           detection_results, processing_time_s)
                return True
            except Exception:
                log.exception(
                    "storage failure for event — folder may be incomplete",
                    extra={"extra_fields": {"event_id": event.event_id}},
                )
                event.warnings.append("storage_failure")
                return False

    def update_metadata(self, event: EventWindow) -> None:
        """Reconciles late-arriving OCR into an already-stored event."""
        with self._lock:
            event_dir = self.event_dir(event.event_id)
            if not os.path.isdir(event_dir):
                log.warning(
                    "late OCR update skipped — event folder does not exist yet",
                    extra={"extra_fields": {"event_id": event.event_id}},
                )
                return
            try:
                self._write_ocr_json(event_dir, event)
                metadata_path = os.path.join(event_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    metadata["coach_number"] = event.coach_number
                    metadata["ocr_status"] = event.identity_state
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)
                log.info(
                    "late OCR reconciled into stored event",
                    extra={"extra_fields": {"event_id": event.event_id, "coach_number": event.coach_number}},
                )
            except Exception:
                log.exception(
                    "failed to patch metadata for late OCR",
                    extra={"extra_fields": {"event_id": event.event_id}},
                )

    @staticmethod
    def _write_event_json(event_dir: str, event: EventWindow) -> None:
        payload = {
            "event_id": event.event_id,
            "coach_index": event.coach_index,
            "start_ts_ms": event.start_ts_ms,
            "end_ts_ms": event.end_ts_ms,
            "status": event.status.value,
            "warnings": event.warnings,
        }
        with open(os.path.join(event_dir, "event.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _write_ocr_json(event_dir: str, event: EventWindow) -> None:
        payload = {
            "event_id": event.event_id,
            "coach_number": event.coach_number,
            "ocr_confidence": event.ocr_confidence,
            "identity_state": event.identity_state,
        }
        with open(os.path.join(event_dir, "OCR.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _write_metadata_json(
        event_dir: str,
        event: EventWindow,
        event_frames: EventFrames,
        stitched_images: List[StitchedImage],
        detection_results: List[DetectionResult],
        processing_time_s: float,
    ) -> None:
        payload = {
            "event_id": event.event_id,
            "start_timestamp_ms": event.start_ts_ms,
            "end_timestamp_ms": event.end_ts_ms,
            "ocr_status": event.identity_state,
            "coach_number": event.coach_number,
            "frame_count": event_frames.total(),
            "camera_frame_counts": {c: len(f) for c, f in event_frames.frames_by_camera.items()},
            "stitched_image_count": len(stitched_images),
            "detection_count": sum(len(r.detections) for r in detection_results),
            "processing_time_s": round(processing_time_s, 3),
            "warnings": event.warnings,
            "generated_at": time.time(),
        }
        with open(os.path.join(event_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
