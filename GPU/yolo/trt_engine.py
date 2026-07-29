"""
TensorRT FP16 batched backend (B4) — HOST ONLY (needs NVIDIA GPU + TensorRT + pycuda).

Loads a prebuilt FP16 engine (see scripts/export_trt.sh), runs batched inference
with a CUDA stream, then decodes + NMS via inference_common. Produces the SAME raw
detection shape as TorchBackend so the response schema is unchanged.

Not unit-tested in the sandbox (no CUDA). The pure pieces it relies on — preprocess
math, decode_yolov8, nms, batch_iter — ARE unit-tested in inference_common.
"""
import logging

import numpy as np

from backends import DetectionBackend
from inference_common import decode_yolov8, nms, batch_iter

logger = logging.getLogger("yolo.trt")

INPUT_SIZE = 640


def preprocess(frame: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, float]:
    """BGR HxWx3 uint8 -> (1x3xSxS float32 [0,1] RGB, scale_x, scale_y).
    Pure-ish (cv2 only for resize); scale factors map boxes back to original."""
    import cv2
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (size, size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw, w / size, h / size


class TrtBackend(DetectionBackend):
    name = "trt"

    def __init__(self, engine_path: str, class_names: dict, *,
                 batch_size: int = 4, conf_threshold: float = 0.35, nms_iou: float = 0.5):
        self.engine_path = engine_path
        self.class_names = class_names
        self.batch_size = batch_size
        self.conf = conf_threshold
        self.nms_iou = nms_iou
        self._load_engine()

    def _load_engine(self):
        import tensorrt as trt        # lazy
        import pycuda.autoinit        # noqa: F401  (initializes CUDA context)
        import pycuda.driver as cuda

        self._trt, self._cuda = trt, cuda
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        logger.info('{"event":"trt_engine_loaded","path":"%s","batch":%d}',
                    self.engine_path, self.batch_size)

    def infer(self, frame) -> list[dict]:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames: list) -> list[list[dict]]:
        results: list[list[dict]] = []
        for chunk in batch_iter(frames, self.batch_size):
            results.extend(self._run_chunk(chunk))
        return results

    def _run_chunk(self, frames: list) -> list[list[dict]]:
        pre = [preprocess(f) for f in frames]
        batch = np.stack([p[0] for p in pre], axis=0)
        raw = self._execute(batch)          # (B, num_boxes, 4+nc)
        out = []
        for i in range(len(frames)):
            scale_x, scale_y = pre[i][1], pre[i][2]
            out.append(self._postprocess(raw[i], scale_x, scale_y))
        return out

    def _execute(self, batch: np.ndarray) -> np.ndarray:
        cuda = self._cuda
        b = batch.shape[0]
        inp = np.ascontiguousarray(batch, dtype=np.float32)
        d_in = cuda.mem_alloc(inp.nbytes)
        out_shape = self.context.get_binding_shape(1)
        out_shape = (b,) + tuple(out_shape[1:])
        host_out = np.empty(out_shape, dtype=np.float32)
        d_out = cuda.mem_alloc(host_out.nbytes)
        cuda.memcpy_htod_async(d_in, inp, self.stream)
        self.context.execute_async_v2([int(d_in), int(d_out)], self.stream.handle)
        cuda.memcpy_dtoh_async(host_out, d_out, self.stream)
        self.stream.synchronize()
        # YOLOv8 head is (B, 4+nc, N); transpose to (B, N, 4+nc)
        if host_out.ndim == 3 and host_out.shape[1] < host_out.shape[2]:
            host_out = host_out.transpose(0, 2, 1)
        return host_out

    def _postprocess(self, preds: np.ndarray, scale_x: float, scale_y: float) -> list[dict]:
        boxes, scores, class_ids = decode_yolov8(preds, self.conf)
        if len(boxes) == 0:
            return []
        keep = nms(boxes, scores, self.nms_iou)
        out = []
        for k in keep:
            x1, y1, x2, y2 = boxes[k]
            out.append({
                "label": self.class_names.get(int(class_ids[k]), str(int(class_ids[k]))),
                "confidence": float(scores[k]),
                "bbox_xyxy": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
            })
        return out
