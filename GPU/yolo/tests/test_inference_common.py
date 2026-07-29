"""B4 — backend-agnostic inference helpers (schema, batching, NMS, decode)."""
import numpy as np
import pytest

from inference_common import format_detections, batch_iter, nms, decode_yolov8, SEVERITY_MAP


def test_format_detections_schema_contract():
    raw = [{"label": "crack", "confidence": 0.9123, "bbox_xyxy": [10.4, 20.6, 110.2, 220.9]}]
    out = format_detections(raw, w=1280, h=720)
    assert out["frame_size"] == [1280, 720]
    d = out["detections"][0]
    # exact legacy contract
    assert set(d.keys()) == {"id", "label", "confidence", "bbox_xyxy", "bbox_xywh", "defect", "severity"}
    assert d["id"] == 0
    assert d["confidence"] == 0.912            # rounded to 3
    assert d["bbox_xyxy"] == [10, 20, 110, 220]
    # legacy contract: width/height are int(x2-x1) on the raw floats (10.4->110.2 => 99)
    assert d["bbox_xywh"] == [10, 20, 99, 200]
    assert d["defect"] is True
    assert d["severity"] == "CRITICAL"


def test_format_detections_non_defect_label():
    out = format_detections([{"label": "spring", "confidence": 0.5, "bbox_xyxy": [0, 0, 5, 5]}], 100, 100)
    d = out["detections"][0]
    assert d["defect"] is False
    assert d["severity"] is None
    assert "spring" not in SEVERITY_MAP


def test_batch_iter_groups_with_partial_last():
    assert list(batch_iter(list(range(10)), 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_batch_iter_rejects_zero():
    with pytest.raises(ValueError):
        list(batch_iter([1, 2], 0))


def test_nms_suppresses_overlap_keeps_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.9, 0.8, 0.95])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 2 in keep            # isolated high-score box kept
    assert 0 in keep            # better of the overlapping pair
    assert 1 not in keep        # suppressed by box 0


def test_nms_empty():
    assert nms(np.empty((0, 4)), np.empty((0,))) == []


def test_decode_yolov8_filters_and_converts():
    # 2 boxes, 3 classes. box0 strong (class1), box1 weak -> filtered
    preds = np.array([
        [50, 60, 20, 40, 0.1, 0.92, 0.0],    # cx,cy,w,h + 3 class scores
        [10, 10, 5, 5, 0.2, 0.1, 0.05],
    ], dtype=np.float32)
    boxes, scores, ids = decode_yolov8(preds, conf_threshold=0.5)
    assert len(boxes) == 1
    assert ids[0] == 1
    assert abs(scores[0] - 0.92) < 1e-6
    # cx50,cy60,w20,h40 -> xyxy 40,40,60,80
    assert list(boxes[0]) == [40, 40, 60, 80]
