"""Parallel preprocessing: lens correction (placeholder) → brightness norm →
resize → noise reduction → blur/quality score. Runs on a thread pool per
camera so it never waits on OCR (ADS rule: OCR must never block preprocessing —
these are separate consumers of the same ring buffer's downstream frames).
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

from core.logging_config import get_stage_logger
from core.models import FrameMessage, ProcessedFrame
from managers.processed_frame_store import ProcessedFrameStore
from managers.ring_buffer_manager import RingBufferManager

log = get_stage_logger("preprocessing")


def _lens_correction(image: np.ndarray) -> np.ndarray:
    """Placeholder — real intrinsics/distortion coefficients come from camera
    calibration (ADS Appendix B, integration-defined). No-op until supplied."""
    return image


def _blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess_image(image: np.ndarray, resize_w: int, resize_h: int):
    corrected = _lens_correction(image)
    resized = cv2.resize(corrected, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    brightness_mean = float(gray.mean())

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    normalized = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    denoised = cv2.fastNlMeansDenoisingColored(normalized, None, 5, 5, 7, 15)

    blur = _blur_score(gray)
    quality_score = round(min(1.0, blur / 200.0) * 0.6 + min(1.0, brightness_mean / 128.0) * 0.4, 4)

    return denoised, blur, quality_score, brightness_mean


class PreprocessingManager:
    def __init__(
        self,
        ring_buffer: RingBufferManager,
        store: ProcessedFrameStore,
        resize_w: int,
        resize_h: int,
        blur_threshold: float,
        thread_count: int = 2,
    ):
        self._ring_buffer = ring_buffer
        self._store = store
        self._resize_w = resize_w
        self._resize_h = resize_h
        self._blur_threshold = blur_threshold
        self._pool = ThreadPoolExecutor(max_workers=thread_count)
        self._puller: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._processed_count = 0
        self._low_quality_count = 0

    @property
    def processed_count(self) -> int:
        return self._processed_count

    def start(self) -> None:
        self._puller = threading.Thread(target=self._pull_loop, daemon=True)
        self._puller.start()

    def stop_and_wait(self, timeout: Optional[float] = None) -> None:
        self._stop_flag.set()
        if self._puller:
            self._puller.join(timeout=timeout)
        self._pool.shutdown(wait=True)

    def _pull_loop(self) -> None:
        while not self._stop_flag.is_set():
            frame = self._ring_buffer.get(timeout=0.5)
            if frame is None:
                if self._ring_buffer.closed and self._ring_buffer.depth == 0:
                    break
                continue
            self._pool.submit(self._process_one, frame)

    def _process_one(self, frame: FrameMessage) -> None:
        try:
            image, blur, quality, brightness = preprocess_image(
                frame.image, self._resize_w, self._resize_h
            )
            if blur < self._blur_threshold:
                self._low_quality_count += 1
                log.warning(
                    "low blur score — frame kept but flagged",
                    extra={"extra_fields": {
                        "camera": frame.camera_name, "frame_id": frame.frame_id, "blur": blur,
                    }},
                )
            processed = ProcessedFrame(
                source=frame, image=image, blur_score=blur,
                quality_score=quality, brightness_mean=brightness,
            )
            self._store.add(processed)
            self._processed_count += 1
        except Exception:
            log.exception(
                "preprocessing failed for frame — dropped",
                extra={"extra_fields": {"camera": frame.camera_name, "frame_id": frame.frame_id}},
            )
