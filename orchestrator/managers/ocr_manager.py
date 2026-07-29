"""OCR camera consumer.

Mirrors the existing production pattern (GPU/ocr/server.py): a single YOLO
train_num_detector call per frame returns BOTH the bogie ROI (used for OCR
text recognition, via services/ocr_engine_adapter — the real GPU/ocr engine)
and any 'gap' class boxes (used for coach-boundary detection). Reusing one
call for both, exactly like the existing service does, avoids running the
same model twice per frame.

Runs on its own thread pool, polling the OCR camera's ProcessedFrameStore —
entirely decoupled from PreprocessingManager's thread pool, so OCR latency
can never block preprocessing (ADS / brief rule 1).

Gap boxes and OCR results are only *emitted* via callbacks; this manager does
not decide event boundaries (rule 2 — that is GapDetectionManager's job) and
never treats OCR as a primary identifier (rule 3 — it only ever produces
metadata for EventManager to attach).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from core.logging_config import get_stage_logger
from core.models import GapCandidate, OcrObservation, ProcessedFrame
from managers.processed_frame_store import ProcessedFrameStore
from services.ocr_engine_adapter import recognize_coach_number, OCR_ENGINE_AVAILABLE
from services.yolo_client import YoloClient

log = get_stage_logger("ocr_manager")


class OCRManager:
    def __init__(
        self,
        camera_name: str,
        store: ProcessedFrameStore,
        yolo_client: YoloClient,
        on_gap_candidate: Callable[[GapCandidate], None],
        on_ocr_result: Callable[[OcrObservation], None],
        sample_every_n: int = 1,
        thread_count: int = 2,
        poll_interval_s: float = 0.2,
    ):
        self._camera_name = camera_name
        self._store = store
        self._yolo = yolo_client
        self._on_gap_candidate = on_gap_candidate
        self._on_ocr_result = on_ocr_result
        self._sample_every_n = max(1, sample_every_n)
        self._pool = ThreadPoolExecutor(max_workers=thread_count)
        self._poll_interval_s = poll_interval_s

        self._last_ts = float("-inf")
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processed_count = 0
        self._valid_ocr_count = 0
        self._gap_candidates_emitted = 0

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._processed_count,
            "valid_ocr": self._valid_ocr_count,
            "gap_candidates": self._gap_candidates_emitted,
            "ocr_engine_available": OCR_ENGINE_AVAILABLE,
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="ocr-manager")
        self._thread.start()

    def stop_and_wait(self, timeout: Optional[float] = None) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._pool.shutdown(wait=True)

    def _poll_loop(self) -> None:
        while not self._stop_flag.is_set():
            latest = self._store.latest_timestamp(self._camera_name)
            if latest > self._last_ts:
                new_frames = self._store.query_range(self._camera_name, self._last_ts + 1e-3, latest)
                for frame in new_frames:
                    if frame.source.sequence_number % self._sample_every_n == 0:
                        self._pool.submit(self._process_one, frame)
                self._last_ts = latest
            time.sleep(self._poll_interval_s)

    def _process_one(self, frame: ProcessedFrame) -> None:
        try:
            boxes = self._yolo.predict_train_number(frame.image)
        except Exception:
            log.exception(
                "gap/OCR-ROI detection call failed — treating frame as no-detection",
                extra={"extra_fields": {"frame_id": frame.frame_id}},
            )
            boxes = []

        gap_boxes = [b for b in boxes if str(b.get("label", "")).lower() == "gap"]
        for box in gap_boxes:
            candidate = GapCandidate(
                camera_name=self._camera_name,
                timestamp_ms=frame.timestamp_ms,
                confidence=float(box.get("confidence", 0.0)),
                bbox_xyxy=box.get("bbox_xyxy", []),
            )
            self._gap_candidates_emitted += 1
            self._safe_callback(self._on_gap_candidate, candidate)

        boogie_boxes = [b for b in boxes if str(b.get("label", "")).lower() == "boogie"]
        roi_bbox = boogie_boxes[0]["bbox_xyxy"] if boogie_boxes else None

        try:
            coach_number, confidence, pass_used, roi_used = recognize_coach_number(frame.image, roi_bbox)
        except Exception:
            log.exception(
                "OCR recognition failed — event will remain identity_unresolved for this frame",
                extra={"extra_fields": {"frame_id": frame.frame_id}},
            )
            coach_number, confidence, pass_used, roi_used = None, 0.0, 0, False

        if coach_number:
            self._valid_ocr_count += 1

        observation = OcrObservation(
            timestamp_ms=frame.timestamp_ms,
            coach_number=coach_number,
            confidence=confidence,
            pass_used=pass_used,
            roi_used=roi_used,
        )
        self._processed_count += 1
        self._safe_callback(self._on_ocr_result, observation)

    @staticmethod
    def _safe_callback(fn, arg) -> None:
        try:
            fn(arg)
        except Exception:
            log.exception("downstream callback raised — continuing (OCR must never block capture)")
