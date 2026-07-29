"""
YOLO Service — port 5002
Two models loaded at startup on GPU:
  best.pt               → POST /api/yolo/predict               (defect detection, Phase 3)
  train_num_detector.pt → POST /api/yolo/predict_train_number  (bogie ROI, Phase 2 OCR)

Ported from POC/backend/YOLO/server.py (Flask → FastAPI).
"""
import os
import sys
import logging
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv
from model_manager import resolve_weights_path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from logging_utils import configure_logging  # noqa: E402
from health import health_payload  # noqa: E402
from inference_common import format_detections, SEVERITY_MAP  # noqa: E402,F401
from backends import build_backend  # noqa: E402

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

configure_logging("yolo")
logger = logging.getLogger(__name__)

DEFECT_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "best.pt"),
)
OCR_MODEL_PATH = os.environ.get(
    "YOLO_OCR_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "train_num_detector.pt"),
)
CONF_THRESHOLD = float(os.environ.get("YOLO_CONF", "0.35"))
# B4: torch (default/fallback) or trt (TensorRT FP16 batched). SEVERITY_MAP now lives
# in inference_common (imported above) so both backends format detections identically.

defect_model = None
ocr_detector_model = None
backend = None          # B4 DetectionBackend (torch|trt), built at startup

# Rolling metrics — updated on every predict call
import time  # noqa: E402
import collections  # noqa: E402
_latency_window = collections.deque(maxlen=100)  # last 100 inference times (ms)
_requests_total = 0
_service_start = time.time()

app = FastAPI(title="VandeInspect YOLO Service", version="1.0.0")


@app.on_event("startup")
def load_models():
    global defect_model, ocr_detector_model, DEFECT_MODEL_PATH, OCR_MODEL_PATH, backend
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return

    # Model registry takes priority over the static env-configured path —
    # a promoted/rolled-back model_versions row wins if one exists.
    DEFECT_MODEL_PATH = resolve_weights_path("defect_detector", DEFECT_MODEL_PATH)
    OCR_MODEL_PATH = resolve_weights_path("train_num_detector", OCR_MODEL_PATH)

    requested_device = os.environ.get("YOLO_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "YOLO_DEVICE=%s requested but CUDA is not available (torch build: %s) — falling back to CPU",
            requested_device, torch.__version__,
        )
        device = "cpu"
    else:
        device = requested_device
    blank = np.zeros((640, 640, 3), dtype="uint8")

    if os.path.exists(DEFECT_MODEL_PATH):
        logger.info("Loading defect model: %s → %s", DEFECT_MODEL_PATH, device)
        defect_model = YOLO(DEFECT_MODEL_PATH)
        defect_model.to(device)
        defect_model(blank, verbose=False)
        logger.info("Defect model ready. Classes: %s", list(defect_model.names.values())[:6])
        # B4: pick inference backend (torch default; trt if INFERENCE_BACKEND=trt + engine exists)
        inference_cfg = {
            "backend": os.environ.get("INFERENCE_BACKEND", "torch"),
            "trt_loader": os.environ.get("TRT_LOADER", "ultralytics"),
            "engine_path": os.environ.get(
                "YOLO_ENGINE_PATH", os.path.join(os.path.dirname(__file__), "models", "best.engine")),
            "batch_size": int(os.environ.get("YOLO_BATCH", "4")),
            "conf": CONF_THRESHOLD,
            "nms_iou": float(os.environ.get("YOLO_NMS_IOU", "0.5")),
            "class_names": dict(defect_model.names),
        }
        backend = build_backend(inference_cfg, torch_model=defect_model)
        logger.info("Inference backend: %s", backend.name)
    else:
        logger.warning("Defect model NOT found: %s (Phase 3 will fail)", DEFECT_MODEL_PATH)

    if os.path.exists(OCR_MODEL_PATH):
        logger.info("Loading OCR detector model: %s → %s", OCR_MODEL_PATH, device)
        ocr_detector_model = YOLO(OCR_MODEL_PATH)
        ocr_detector_model.to(device)
        ocr_detector_model(blank, verbose=False)
        logger.info("OCR detector model ready. Classes: %s", list(ocr_detector_model.names.values()))
    else:
        logger.warning("OCR detector model NOT found: %s (OCR will use full-frame fallback)", OCR_MODEL_PATH)


def decode_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Cannot decode image bytes")
    return frame


@app.get("/health")
def health():
    return health_payload(
        "yolo",
        port=5002,
        defect_model_loaded=defect_model is not None,
        ocr_model_loaded=ocr_detector_model is not None,
        defect_model_path=DEFECT_MODEL_PATH,
        ocr_model_path=OCR_MODEL_PATH,
        inference_backend=(backend.name if backend else None),
    )


@app.get("/metrics")
def metrics():
    global _requests_total, _latency_window, _service_start

    avg_latency = (sum(_latency_window) / len(_latency_window)) if _latency_window else 0.0
    uptime_s = time.time() - _service_start
    fps = _requests_total / uptime_s if uptime_s > 0 else 0.0

    gpu_memory_mb = None
    gpu_utilization_pct = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            gpu_memory_total_mb = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1)
            gpu_utilization_pct = round(gpu_memory_mb / gpu_memory_total_mb * 100, 1) if gpu_memory_total_mb else 0
    except Exception:
        pass

    return {
        "service": "yolo",
        "port": 5002,
        "requests_total": _requests_total,
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "gpu_memory_mb": gpu_memory_mb,
        "gpu_utilization_pct": gpu_utilization_pct,
        "defect_model_loaded": defect_model is not None,
        "ocr_model_loaded": ocr_detector_model is not None,
        "uptime_seconds": round(uptime_s, 1),
    }


@app.post("/api/yolo/predict")
async def predict(file: UploadFile = File(...)):
    """Defect detection — uses best.pt. Called by Phase 3 correlation pipeline."""
    global _requests_total
    if defect_model is None:
        raise HTTPException(status_code=503, detail="Defect model not loaded. Check models/best.pt.")

    frame = decode_bytes(await file.read())
    t0 = time.time()
    raw_dets = backend.infer(frame)        # B4: torch or trt, same raw shape
    _latency_window.append((time.time() - t0) * 1000)
    _requests_total += 1
    h, w = frame.shape[:2]
    return format_detections(raw_dets, w, h)   # identical public schema for both backends


@app.post("/api/yolo/predict_train_number")
async def predict_train_number(file: UploadFile = File(...)):
    """Bogie ROI detection — uses train_num_detector.pt. Called by OCR service (Phase 2)."""
    global _requests_total
    if ocr_detector_model is None:
        # Graceful: return empty so OCR service falls back to full-frame PaddleOCR
        logger.debug("OCR detector model not loaded — returning empty boxes")
        return {"boxes": [], "frame_size": [0, 0], "model_loaded": False}

    frame = decode_bytes(await file.read())
    t0 = time.time()
    results = ocr_detector_model(frame, conf=0.25, verbose=False)[0]
    _latency_window.append((time.time() - t0) * 1000)
    _requests_total += 1
    h, w = frame.shape[:2]

    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = ocr_detector_model.names[int(box.cls)]
        boxes.append({
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(float(box.conf), 3),
            "class_id": int(box.cls),
            "label": label,
        })

    if boxes:
        logger.info("predict_train_number: %d box(es) — labels: %s", len(boxes), [b["label"] for b in boxes])
    return {"boxes": boxes, "frame_size": [w, h], "model_loaded": True}
