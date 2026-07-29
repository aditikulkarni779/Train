"""
Backend-agnostic inference helpers (B4).

These are shared by BOTH the PyTorch and TensorRT backends so the /api/yolo/predict
response schema is byte-identical regardless of which backend ran. Pure logic
(numpy only, no torch / tensorrt / cv2) — unit-tested in-sandbox.
"""
import numpy as np

# Severity table — authoritative; kept here (not in server.py) so format_detections
# and its tests don't need to import the heavy FastAPI server.
SEVERITY_MAP = {
    "crack":          "CRITICAL",
    "leakage":        "CRITICAL",
    "smoke_emission": "CRITICAL",
    "broken":         "HIGH",
    "rust":           "HIGH",
    "deformation":    "HIGH",
    "hole":           "HIGH",
    "missing_part":   "MEDIUM",
    "puncture":       "MEDIUM",
    "hanging":        "MEDIUM",
    "loose":          "LOW",
}


def format_detections(raw_dets: list[dict], w: int, h: int) -> dict:
    """Normalize backend-raw detections into the public response schema.

    raw_dets: list of {label, confidence, bbox_xyxy:[x1,y1,x2,y2]}
    returns:  {detections:[...], frame_size:[w,h]} matching the legacy contract.
    """
    detections = []
    for i, d in enumerate(raw_dets):
        x1, y1, x2, y2 = d["bbox_xyxy"]
        label = d["label"]
        detections.append({
            "id": i,
            "label": label,
            "confidence": round(float(d["confidence"]), 3),
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_xywh": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "defect": label in SEVERITY_MAP,
            "severity": SEVERITY_MAP.get(label),
        })
    return {"detections": detections, "frame_size": [w, h]}


def batch_iter(seq: list, n: int):
    """Yield successive lists of up to n items (last batch may be smaller)."""
    if n < 1:
        raise ValueError("batch size must be >= 1")
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def decode_yolov8(preds: np.ndarray, conf_threshold: float):
    """Decode raw YOLOv8 head output into (boxes_xyxy, scores, class_ids), pre-NMS.

    preds: (num_boxes, 4+num_classes) — first 4 cols are cx,cy,w,h (input-space);
           remaining cols are per-class scores. Returns confidence-filtered arrays.
    Pure numpy — unit-tested without TensorRT.
    """
    if preds.ndim != 2 or preds.shape[1] < 5:
        raise ValueError("preds must be (N, 4+num_classes)")
    cls_scores = preds[:, 4:]
    class_ids = cls_scores.argmax(axis=1)
    scores = cls_scores.max(axis=1)
    keep = scores >= conf_threshold
    preds, scores, class_ids = preds[keep], scores[keep], class_ids[keep]
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, scores, class_ids


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """Non-Maximum Suppression. boxes: Nx4 (x1,y1,x2,y2). Returns kept indices,
    highest score first. Needed for TensorRT raw output (ultralytics does its own)."""
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou < iou_threshold]
    return keep
