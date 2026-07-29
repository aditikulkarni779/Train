"""
Inference backends (B4) — pluggable behind a stable interface so /api/yolo/predict
is identical whether PyTorch or TensorRT ran.

    INFERENCE_BACKEND=torch   -> TorchBackend  (ultralytics, existing path, fallback)
    INFERENCE_BACKEND=trt     -> TrtBackend    (TensorRT FP16, batched — GPU host)

Each backend returns *raw* detections [{label, confidence, bbox_xyxy}]; the server
runs them through inference_common.format_detections for the public schema.

select_backend_name() is pure and unit-tested. TorchBackend/TrtBackend lazily import
their heavy deps so importing this module is cheap.
"""
import os
import logging
from abc import ABC, abstractmethod

from inference_common import batch_iter

logger = logging.getLogger("yolo.backends")

VALID_BACKENDS = ("torch", "trt")


def normalize_ultralytics_result(result, names) -> list[dict]:
    """ultralytics Result -> [{label, confidence, bbox_xyxy}] (shared by torch + engine)."""
    out = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append({
            "label": names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox_xyxy": [x1, y1, x2, y2],
        })
    return out


def select_backend_name(cfg: dict) -> str:
    """Resolve the backend name from config + engine availability.

    cfg: {backend: 'torch'|'trt', engine_path: str}
    Rules: explicit invalid name -> ValueError. 'trt' requested but engine file
    missing -> fall back to 'torch' (don't crash a deploy over a missing engine).
    """
    name = (cfg.get("backend") or "torch").lower()
    if name not in VALID_BACKENDS:
        raise ValueError(f"unknown INFERENCE_BACKEND={name!r}; valid: {VALID_BACKENDS}")
    if name == "trt":
        engine = cfg.get("engine_path", "")
        if not engine or not os.path.exists(engine):
            logger.warning('{"event":"trt_engine_missing","engine":"%s","fallback":"torch"}', engine)
            return "torch"
    return name


class DetectionBackend(ABC):
    name = "base"

    @abstractmethod
    def infer(self, frame) -> list[dict]:
        """Single frame -> [{label, confidence, bbox_xyxy}]."""

    def infer_batch(self, frames: list) -> list[list[dict]]:
        """Default: loop infer(). TRT overrides with a true batched pass."""
        return [self.infer(f) for f in frames]


class TorchBackend(DetectionBackend):
    name = "torch"

    def __init__(self, model, conf_threshold: float = 0.35):
        self.model = model            # ultralytics YOLO, already loaded
        self.conf = conf_threshold

    def infer(self, frame) -> list[dict]:
        result = self.model(frame, conf=self.conf, verbose=False)[0]
        return normalize_ultralytics_result(result, self.model.names)

    def infer_batch(self, frames: list) -> list[list[dict]]:
        results = self.model(frames, conf=self.conf, verbose=False)  # ultralytics batches natively
        return [normalize_ultralytics_result(r, self.model.names) for r in results]


class UltralyticsEngineBackend(DetectionBackend):
    """TensorRT via ultralytics (loads .engine like a .pt) — NO pycuda. Pads each
    chunk up to the engine's fixed batch size and drops the padding results, so the
    service's single-frame /predict works against a batch-4 engine."""
    name = "trt"

    def __init__(self, engine_path: str = None, conf_threshold: float = 0.35,
                 batch: int = 4, model=None):
        if model is None:
            from ultralytics import YOLO   # lazy
            model = YOLO(engine_path)
        self.model = model
        self.conf = conf_threshold
        self.batch = batch

    def infer(self, frame) -> list[dict]:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames: list) -> list[list[dict]]:
        out: list[list[dict]] = []
        for chunk in batch_iter(frames, self.batch):
            n = len(chunk)
            padded = chunk + [chunk[-1]] * (self.batch - n) if n < self.batch else chunk
            results = self.model(padded, conf=self.conf, verbose=False)
            for r in results[:n]:            # drop padding results
                out.append(normalize_ultralytics_result(r, self.model.names))
        return out


def build_backend(cfg: dict, torch_model=None) -> DetectionBackend:
    """Factory. torch_model is the already-loaded ultralytics model (for the torch
    path / fallback). TrtBackend is imported lazily so non-GPU hosts don't need it."""
    name = select_backend_name(cfg)
    if name == "trt":
        loader = cfg.get("trt_loader", "ultralytics")
        if loader == "pycuda":
            from trt_engine import TrtBackend   # lazy: imports tensorrt/pycuda
            return TrtBackend(
                engine_path=cfg["engine_path"],
                class_names=cfg.get("class_names", {}),
                batch_size=int(cfg.get("batch_size", 4)),
                conf_threshold=float(cfg.get("conf", 0.35)),
                nms_iou=float(cfg.get("nms_iou", 0.5)),
            )
        # default: ultralytics loads the .engine like a .pt (no pycuda)
        return UltralyticsEngineBackend(
            engine_path=cfg["engine_path"],
            conf_threshold=float(cfg.get("conf", 0.35)),
            batch=int(cfg.get("batch_size", 4)),
        )
    if torch_model is None:
        raise RuntimeError("torch backend selected but no model provided")
    return TorchBackend(torch_model, conf_threshold=float(cfg.get("conf", 0.35)))
