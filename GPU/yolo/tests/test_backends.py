"""B4 — backend selection + TorchBackend normalization (no torch/trt needed)."""
import pytest
from backends import select_backend_name, build_backend, TorchBackend, UltralyticsEngineBackend


def test_default_is_torch():
    assert select_backend_name({}) == "torch"
    assert select_backend_name({"backend": "torch"}) == "torch"


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        select_backend_name({"backend": "openvino"})


def test_trt_without_engine_falls_back_to_torch():
    assert select_backend_name({"backend": "trt", "engine_path": "/nope/missing.engine"}) == "torch"


def test_trt_with_engine_selects_trt(tmp_path):
    eng = tmp_path / "mvis.engine"
    eng.write_bytes(b"fake")
    assert select_backend_name({"backend": "trt", "engine_path": str(eng)}) == "trt"


# ---- TorchBackend with a stub ultralytics model ----
class _T:
    def __init__(self, v): self.v = v
    def tolist(self): return self.v


class _Box:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [_T(xyxy)]
        self.cls = cls
        self.conf = conf


class _Result:
    def __init__(self, boxes): self.boxes = boxes


class _Model:
    names = {0: "crack", 1: "rust"}

    def __init__(self, boxes): self._boxes = boxes

    def __call__(self, frame, conf=0.35, verbose=False):
        if isinstance(frame, list):
            return [_Result(self._boxes) for _ in frame]
        return [_Result(self._boxes)]


def test_torch_backend_normalizes_detections():
    model = _Model([_Box([1, 2, 3, 4], 0, 0.91), _Box([5, 6, 7, 8], 1, 0.77)])
    be = build_backend({"backend": "torch", "conf": 0.35}, torch_model=model)
    assert isinstance(be, TorchBackend) and be.name == "torch"
    dets = be.infer("frame")
    assert dets == [
        {"label": "crack", "confidence": 0.91, "bbox_xyxy": [1, 2, 3, 4]},
        {"label": "rust", "confidence": 0.77, "bbox_xyxy": [5, 6, 7, 8]},
    ]


def test_torch_backend_batch():
    model = _Model([_Box([1, 2, 3, 4], 0, 0.9)])
    be = build_backend({"backend": "torch"}, torch_model=model)
    out = be.infer_batch(["f1", "f2", "f3"])
    assert len(out) == 3
    assert all(o[0]["label"] == "crack" for o in out)


def test_torch_selected_but_no_model_raises():
    with pytest.raises(RuntimeError):
        build_backend({"backend": "torch"}, torch_model=None)


# ---- UltralyticsEngineBackend: fixed-batch-4 padding ----
class _RecModel:
    names = {0: "crack"}

    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []                  # records batch size of each __call__

    def __call__(self, frames, conf=0.35, verbose=False):
        self.calls.append(len(frames))
        return [_Result(self._boxes) for _ in frames]


def test_engine_single_frame_pads_to_batch_and_returns_one():
    m = _RecModel([_Box([1, 2, 3, 4], 0, 0.9)])
    be = UltralyticsEngineBackend(model=m, batch=4)
    out = be.infer("frame")
    assert m.calls == [4]                       # 1 frame padded up to the engine's batch 4
    assert out == [{"label": "crack", "confidence": 0.9, "bbox_xyxy": [1, 2, 3, 4]}]


def test_engine_batch_chunks_and_drops_padding():
    m = _RecModel([_Box([1, 2, 3, 4], 0, 0.9)])
    be = UltralyticsEngineBackend(model=m, batch=4)
    out = be.infer_batch(["a", "b", "c", "d", "e", "f"])   # 6 frames -> [4],[2->pad4]
    assert m.calls == [4, 4]
    assert len(out) == 6                         # padding results dropped
    assert be.name == "trt"


def test_engine_exact_batch_no_padding():
    m = _RecModel([_Box([0, 0, 1, 1], 0, 0.5)])
    be = UltralyticsEngineBackend(model=m, batch=4)
    be.infer_batch(["a", "b", "c", "d"])
    assert m.calls == [4]
