"""Wires every manager together and owns the pipeline lifecycle:
start all cameras → run until every video source ends → drain in-flight
events → close the final (trailing) event window → wait for all event
finalization to complete → shutdown cleanly.

This is the only place that knows about every manager; individual managers
never import each other directly (SOLID — dependency direction flows
through this coordinator, not between managers).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from core.config_loader import PipelineConfig
from core.logging_config import get_stage_logger
from core.models import CameraType, EventStatus, EventWindow, GapCandidate, OcrObservation
from managers.detection_manager import DetectionManager
from managers.event_manager import EventManager
from managers.frame_assignment_manager import FrameAssignmentManager
from managers.frame_capture_manager import FrameCaptureManager
from managers.gap_detection_manager import GapDetectionManager
from managers.ocr_manager import OCRManager
from managers.preprocessing_manager import PreprocessingManager
from managers.processed_frame_store import ProcessedFrameStore
from managers.ring_buffer_manager import RingBufferManager
from managers.stitch_manager import StitchManager
from managers.storage_manager import StorageManager
from services.yolo_client import YoloClient
from visualization import VisualizationState, Visualizer

log = get_stage_logger("pipeline_coordinator")


class PipelineCoordinator:
    def __init__(self, config: PipelineConfig):
        self._config = config
        self._store = ProcessedFrameStore()
        self._yolo = YoloClient(config.yolo_service_url, config.yolo_request_timeout_s)
        self._viz_state = VisualizationState()
        self._visualizer = Visualizer(self._viz_state, config.visualization_refresh_ms) \
            if config.enable_visualization else None

        self._storage = StorageManager(config.session_name, config.output_folder)
        component_cameras = [c.name for c in config.cameras if c.camera_type == CameraType.COMPONENT]
        self._frame_assignment = FrameAssignmentManager(self._store, component_cameras)
        self._stitcher = StitchManager(
            config.stitch_camera_top, config.stitch_camera_bottom, config.stitch_pair_epsilon_ms,
            overlap_ratio=config.stitch_overlap_ratio,
            use_feature_refinement=config.stitch_use_feature_refinement,
        )
        self._detector = DetectionManager(self._yolo, config.detection_confidence_threshold)
        self._finalize_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="event-finalize")

        self._session_start_ts_ms = time.time() * 1000.0
        self._event_manager = EventManager(
            session_start_ts_ms=self._session_start_ts_ms,
            on_event_bounded=self._on_event_bounded,
            on_event_updated=self._on_event_updated,
            on_identity_resolved=self._on_identity_resolved,
            late_ocr_grace_period_ms=config.late_ocr_grace_period_ms,
        )
        self._gap_detector = GapDetectionManager(
            confidence_threshold=config.gap_confidence_threshold,
            cluster_window_ms=config.gap_cluster_window_ms,
            on_boundary=self._event_manager.on_gap_boundary,
        )

        ocr_camera = next(c for c in config.cameras if c.camera_type == CameraType.OCR)
        self._ocr_manager = OCRManager(
            camera_name=ocr_camera.name,
            store=self._store,
            yolo_client=self._yolo,
            on_gap_candidate=self._on_gap_candidate,
            on_ocr_result=self._on_ocr_result,
            sample_every_n=config.ocr_camera_sample_every_n,
        )

        self._ring_buffers: Dict[str, RingBufferManager] = {}
        self._captures: Dict[str, FrameCaptureManager] = {}
        self._preprocessors: Dict[str, PreprocessingManager] = {}
        self._cameras_remaining = set(c.name for c in config.cameras)
        self._all_finished = threading.Event()

        for camera in config.cameras:
            ring_buffer = RingBufferManager(camera.name, config.ring_buffer_size)
            capture = FrameCaptureManager(
                camera, ring_buffer, self._session_start_ts_ms,
                on_camera_finished=self._on_camera_finished,
            )
            preprocessor = PreprocessingManager(
                ring_buffer, self._store,
                camera.resize_width or config.resize_width,
                camera.resize_height or config.resize_height,
                config.blur_score_threshold,
                config.preprocessing_thread_count,
            )
            self._ring_buffers[camera.name] = ring_buffer
            self._captures[camera.name] = capture
            self._preprocessors[camera.name] = preprocessor

        self._events_completed = 0
        self._events_failed_partial = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("pipeline starting", extra={"extra_fields": {
            "session": self._config.session_name, "cameras": list(self._ring_buffers.keys()),
        }})

        if self._visualizer:
            self._visualizer.start()

        for preprocessor in self._preprocessors.values():
            preprocessor.start()
        self._ocr_manager.start()
        for capture in self._captures.values():
            capture.start()

        self._all_finished.wait()

        for capture in self._captures.values():
            capture.join(timeout=5.0)
        for preprocessor in self._preprocessors.values():
            preprocessor.stop_and_wait(timeout=30.0)
        self._ocr_manager.stop_and_wait(timeout=10.0)

        self._gap_detector.flush()
        last_ts = max(
            (self._store.latest_timestamp(name) for name in self._ring_buffers),
            default=self._session_start_ts_ms,
        )
        self._event_manager.close_final_window(last_ts)

        self._finalize_pool.shutdown(wait=True)

        log.info("pipeline finished", extra={"extra_fields": {
            "events_completed": self._events_completed,
            "events_failed_partial": self._events_failed_partial,
            "gap_detection_stats": self._gap_detector.stats,
            "ocr_stats": self._ocr_manager.stats,
        }})
        if self._visualizer:
            self._visualizer.stop()

    def _on_camera_finished(self, camera_name: str) -> None:
        self._cameras_remaining.discard(camera_name)
        if not self._cameras_remaining:
            self._all_finished.set()

    # ── Callback wiring ───────────────────────────────────────────────────

    def _on_gap_candidate(self, candidate: GapCandidate) -> None:
        self._gap_detector.submit_candidate(candidate)

    def _on_ocr_result(self, observation: OcrObservation) -> None:
        self._event_manager.enrich_with_ocr(observation)

    def _on_event_updated(self, event: EventWindow) -> None:
        """Late OCR arrived after the event was already stored — patch disk."""
        self._storage.update_metadata(event)
        self._viz_state.update(event.event_id, ocr_status=event.identity_state)

    def _on_identity_resolved(self, event: EventWindow) -> None:
        """Coach number just resolved (early or late) — rename this event's
        folder to it, and if this is the first coach resolved in the whole
        session, rename the session folder too (train-number proxy — see
        docstring in storage_manager.py for why there's no separate signal)."""
        self._storage.rename_event_for_coach_number(event.event_id, event.coach_number)
        self._storage.rename_session_for_train_number(event.coach_number)
        self._viz_state.update(event.event_id, ocr_status=event.identity_state)

    def _on_event_bounded(self, event: EventWindow) -> None:
        self._viz_state.update(
            event.event_id,
            gap_detected="closed_by_stream_end_not_gap_boundary" not in event.warnings,
            pipeline_stage="BOUNDED",
        )
        self._finalize_pool.submit(self._finalize_event, event)

    def _finalize_event(self, event: EventWindow) -> None:
        start = time.time()
        try:
            self._viz_state.update(event.event_id, pipeline_stage="ASSIGNING_FRAMES")
            event_frames = self._frame_assignment.assign(event)
            self._viz_state.update(
                event.event_id, frames_assigned=event_frames.total(),
                pipeline_stage="STITCHING",
            )

            stitched_images = self._stitcher.stitch_event(event_frames)
            self._viz_state.update(
                event.event_id, frames_stitched=len(stitched_images),
                pipeline_stage="DETECTING",
            )

            detection_results = self._detector.detect(event.event_id, stitched_images)
            components_detected = sum(len(r.detections) for r in detection_results)
            self._viz_state.update(
                event.event_id, components_detected=components_detected,
                pipeline_stage="SAVING",
            )

            self._event_manager.mark_unresolved_if_no_identity(event)
            self._viz_state.update(event.event_id, ocr_status=event.identity_state)

            elapsed = time.time() - start
            saved = self._storage.save_event(event, event_frames, stitched_images, detection_results, elapsed)

            if saved and not event.warnings:
                event.status = EventStatus.STORED
                self._events_completed += 1
                storage_status = "saved"
            else:
                event.status = EventStatus.FAILED_PARTIAL
                self._events_failed_partial += 1
                storage_status = "saved_partial" if saved else "failed"

            self._viz_state.update(
                event.event_id, storage_status=storage_status, pipeline_stage="COMPLETE",
            )
            log.info(
                "event finalized",
                extra={"extra_fields": {
                    "event_id": event.event_id, "status": event.status.value,
                    "warnings": event.warnings, "elapsed_s": round(elapsed, 3),
                }},
            )
        except Exception:
            log.exception(
                "event finalization crashed — event marked failed, pipeline continues",
                extra={"extra_fields": {"event_id": event.event_id}},
            )
            self._events_failed_partial += 1
            self._viz_state.update(event.event_id, storage_status="failed", pipeline_stage="FAILED")
