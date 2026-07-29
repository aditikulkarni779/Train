"""Adapter over the existing GPU/ocr engine — imports the pure functions
directly (run_ocr, preprocess_frame, filter_train_numbers) instead of calling
the /ocr HTTP endpoint, because that endpoint hard-requires DATABASE_URL and
writes straight into the production Postgres schema (GPU/ocr/server.py:43,
222-269). This orchestrator is a standalone demo (decision: no backend/DB
coupling), so it reuses the same recognition algorithm without importing the
FastAPI/DB wrapper around it. No OCR logic is duplicated or rewritten here.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_GPU_OCR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "GPU", "ocr")
)
if _GPU_OCR_DIR not in sys.path:
    sys.path.insert(0, _GPU_OCR_DIR)

from core.logging_config import get_stage_logger

log = get_stage_logger("ocr_engine_adapter")

try:
    from ocr_engine import run_ocr              # existing GPU/ocr module
    from preprocess import preprocess_frame
    from train_number_filter import filter_train_numbers
    OCR_ENGINE_AVAILABLE = True
except Exception as exc:  # rapidocr/onnxruntime not installed in this environment
    log.warning(
        "GPU/ocr engine unavailable — OCR stage will run in degraded mode",
        extra={"extra_fields": {"error": str(exc)}},
    )
    OCR_ENGINE_AVAILABLE = False

    def run_ocr(image):
        return []

    def preprocess_frame(image):
        return image

    def filter_train_numbers(results):
        return []


def recognize_coach_number(
    frame: np.ndarray,
    roi_bbox_xyxy: Optional[list],
) -> Tuple[Optional[str], float, int, bool]:
    """Mirrors GPU/ocr/server.py's run_pipeline steps 3-6 (crop → 2-pass →
    full-frame fallback), given a bbox already found by the shared YOLO call.
    Returns (coach_number, confidence, pass_used, roi_used).
    """
    h, w = frame.shape[:2]

    if roi_bbox_xyxy:
        x1, y1, x2, y2 = roi_bbox_xyxy
        bw, bh = x2 - x1, y2 - y1
        pw, ph = int(bw * 0.15), int(bh * 0.15)
        x1p, y1p = max(0, x1 - pw), max(0, y1 - ph)
        x2p, y2p = min(w, x2 + pw), min(h, y2 + ph)
        crop = frame[y1p:y2p, x1p:x2p]

        if crop.size > 0:
            raw1 = run_ocr(crop)
            candidates = filter_train_numbers(raw1)
            if candidates:
                conf = max((r["confidence"] for r in raw1
                            if r["text"].replace(" ", "").strip() in candidates), default=0.0)
                return candidates[0], conf, 1, True

            raw2 = run_ocr(preprocess_frame(crop))
            candidates = filter_train_numbers(raw2)
            if candidates:
                conf = max((r["confidence"] for r in raw2
                            if r["text"].replace(" ", "").strip() in candidates), default=0.0)
                return candidates[0], conf, 2, True

    raw_ff = run_ocr(preprocess_frame(frame))
    candidates = filter_train_numbers(raw_ff)
    if candidates:
        conf = max((r["confidence"] for r in raw_ff
                    if r["text"].replace(" ", "").strip() in candidates), default=0.0)
        return candidates[0], conf, 1, False

    return None, 0.0, 0, False
