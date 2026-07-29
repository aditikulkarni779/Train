"""HTTP client for the existing GPU/yolo service (server.py, port 5002).

Reuse, not reimplementation: this only wraps the two endpoints that already
exist — /api/yolo/predict (component/defect detector, best.pt) and
/api/yolo/predict_train_number (bogie ROI + gap class, train_num_detector.pt).
No model code lives here. Requires the yolo service to be running.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import requests

from core.logging_config import get_stage_logger

log = get_stage_logger("yolo_client")


class YoloServiceUnavailable(RuntimeError):
    pass


class YoloClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    def _encode(self, image: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("cv2.imencode failed")
        return buf.tobytes()

    def predict_train_number(self, image: np.ndarray) -> List[dict]:
        """Returns raw boxes (label includes 'gap' and 'Boogie' among others)."""
        try:
            resp = requests.post(
                f"{self._base_url}/api/yolo/predict_train_number",
                files={"file": ("frame.jpg", self._encode(image), "image/jpeg")},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("boxes", [])
        except Exception as exc:
            log.warning(
                "yolo predict_train_number unreachable — returning no boxes",
                extra={"extra_fields": {"error": str(exc)}},
            )
            return []

    def predict_components(self, image: np.ndarray) -> List[dict]:
        """Returns component/defect detections from the defect model (best.pt)."""
        try:
            resp = requests.post(
                f"{self._base_url}/api/yolo/predict",
                files={"file": ("frame.jpg", self._encode(image), "image/jpeg")},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            return body.get("detections", body if isinstance(body, list) else [])
        except Exception as exc:
            log.warning(
                "yolo predict (component detection) unreachable — returning no detections",
                extra={"extra_fields": {"error": str(exc)}},
            )
            return []
