"""Runs the existing Component Detection model (GPU/yolo/server.py /predict,
best.pt) over each stitched image. No model logic here — this is a thin
call-and-collect wrapper, per rule: "the orchestrator should simply call it
after stitching."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.logging_config import get_stage_logger
from managers.stitch_manager import StitchedImage
from services.yolo_client import YoloClient

log = get_stage_logger("detection_manager")


@dataclass
class DetectionResult:
    timestamp_ms: float
    detections: List[dict]


class DetectionManager:
    def __init__(self, yolo_client: YoloClient, confidence_threshold: float):
        self._yolo = yolo_client
        self._confidence_threshold = confidence_threshold

    def detect(self, event_id: str, stitched_images: List[StitchedImage]) -> List[DetectionResult]:
        results = []
        for stitched in stitched_images:
            try:
                raw = self._yolo.predict_components(stitched.image)
            except Exception:
                log.exception(
                    "component detection call failed for stitched image — skipping",
                    extra={"extra_fields": {"event_id": event_id}},
                )
                raw = []

            filtered = [d for d in raw if float(d.get("confidence", 0.0)) >= self._confidence_threshold]
            results.append(DetectionResult(timestamp_ms=stitched.timestamp_ms, detections=filtered))

        total = sum(len(r.detections) for r in results)
        log.info(
            "component detection complete",
            extra={"extra_fields": {
                "event_id": event_id, "stitched_images": len(stitched_images), "detections_total": total,
            }},
        )
        return results
