"""YAML → typed config. Every tunable the managers use lives here — no
hardcoded thresholds/paths in manager code (mirrors NFR-M3 in the ADS)."""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import List

from core.models import CameraConfig, CameraType


@dataclass
class PipelineConfig:
    session_name: str
    output_folder: str
    cameras: List[CameraConfig]

    ring_buffer_size: int = 300
    ring_buffer_duration_s: float = 30.0

    preprocessing_thread_count: int = 3
    resize_width: int = 960
    resize_height: int = 540
    blur_score_threshold: float = 80.0

    yolo_service_url: str = "http://127.0.0.1:5002"
    yolo_request_timeout_s: float = 10.0

    gap_confidence_threshold: float = 0.4
    gap_cluster_window_ms: float = 800.0

    ocr_confidence_threshold: float = 0.4
    ocr_camera_sample_every_n: int = 1

    # Physical mount: cameras stacked vertically (stitch.py, repo root),
    # NOT side-by-side — "top"/"bottom" reflects the real rig, not a naming choice.
    stitch_camera_top: str = "cam1"
    stitch_camera_bottom: str = "cam2"
    stitch_pair_epsilon_ms: float = 150.0
    stitch_overlap_ratio: float = 0.15
    stitch_use_feature_refinement: bool = False

    detection_confidence_threshold: float = 0.35

    enable_visualization: bool = True
    visualization_refresh_ms: int = 500

    late_ocr_grace_period_ms: float = 5000.0


def _camera_from_dict(d: dict) -> CameraConfig:
    return CameraConfig(
        name=d["name"],
        source=d["source"],
        camera_type=CameraType(d["camera_type"]),
        target_fps=float(d.get("target_fps", 5.0)),
        resize_width=int(d["resize_width"]) if "resize_width" in d else None,
        resize_height=int(d["resize_height"]) if "resize_height" in d else None,
    )


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cameras = [_camera_from_dict(c) for c in raw["cameras"]]
    kwargs = {k: v for k, v in raw.items() if k != "cameras"}
    return PipelineConfig(cameras=cameras, **kwargs)
