"""
B4 accuracy + latency harness (HOST — needs GPU + a built .engine).

Compares the SAME frames through the PyTorch (.pt) and TensorRT (.engine) backends,
both loaded via ultralytics (so an ultralytics-exported engine just works), and reports:
  * detection agreement (proxy for mAP parity — gate: >= 99% same labels)
  * per-backend latency P50/P95/P99 (gate: trt P99 < 20 ms)

    # build the engine first (needs: pip install tensorrt):
    python GPU/yolo/scripts/export_engine.py GPU/yolo/models/best.pt --batch 4
    # then benchmark (uses an images dir, or synthesizes frames if omitted):
    python GPU/yolo/scripts/compare_backends.py GPU/yolo/models/best.pt GPU/yolo/models/best.engine
    python GPU/yolo/scripts/compare_backends.py GPU/yolo/models/best.pt GPU/yolo/models/best.engine path/to/images

Exit 0 if agreement and trt P99 meet the gate, else 1.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared")))

import numpy as np  # noqa: E402
from backends import TorchBackend  # noqa: E402


def _p(vals, q):
    return float(np.percentile(vals, q)) if vals else 0.0


def _labels(dets):
    return sorted(d["label"] for d in dets)


def _load_frames(images_dir, n_synth):
    import cv2
    if images_dir and os.path.isdir(images_dir):
        paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        frames = [cv2.imread(p) for p in paths]
        frames = [f for f in frames if f is not None]
        if frames:
            print(f"Loaded {len(frames)} image(s) from {images_dir}")
            return frames
    print(f"No images dir given/usable — synthesizing {n_synth} random 640x640 frames "
          "(latency valid; agreement still compares the two backends on identical frames)")
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8) for _ in range(n_synth)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_model")
    ap.add_argument("engine")
    ap.add_argument("images_dir", nargs="?", default=None)
    ap.add_argument("--synth", type=int, default=48, help="frames to synthesize if no images dir")
    ap.add_argument("--batch", type=int, default=4, help="must match the exported engine batch")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--p99-budget-ms", type=float, default=20.0)
    args = ap.parse_args()

    from ultralytics import YOLO
    torch_be = TorchBackend(YOLO(args.pt_model))
    trt_be = TorchBackend(YOLO(args.engine))          # ultralytics loads .engine transparently

    frames = _load_frames(args.images_dir, args.synth)
    b = args.batch
    # pad to a whole number of batches (fixed-batch engine needs exactly b frames)
    if len(frames) % b:
        frames = frames + [frames[-1]] * (b - len(frames) % b)

    # warm up both with a full batch (TRT allocates buffers / builds context)
    warm = frames[:b]
    for _ in range(args.warmup):
        torch_be.infer_batch(warm)
        trt_be.infer_batch(warm)

    agree = total = 0
    trt_lat, torch_lat = [], []   # per-batch latency (the gate is batch-of-4)
    for i in range(0, len(frames), b):
        chunk = frames[i:i + b]
        t0 = time.time()
        td = torch_be.infer_batch(chunk)
        torch_lat.append((time.time() - t0) * 1000)
        t0 = time.time()
        rd = trt_be.infer_batch(chunk)
        trt_lat.append((time.time() - t0) * 1000)
        for a, c in zip(td, rd):
            total += 1
            if _labels(a) == _labels(c):
                agree += 1

    agreement = agree / total if total else 1.0
    trt_p99 = _p(trt_lat, 99)
    print(f"frames={total} batch={b} batches={len(trt_lat)} agreement={agreement:.3f}")
    print(f"torch  per-batch P50/P95/P99 = {_p(torch_lat,50):.1f}/{_p(torch_lat,95):.1f}/{_p(torch_lat,99):.1f} ms")
    print(f"trt    per-batch P50/P95/P99 = {_p(trt_lat,50):.1f}/{_p(trt_lat,95):.1f}/{trt_p99:.1f} ms")
    speedup = (_p(torch_lat, 50) / _p(trt_lat, 50)) if _p(trt_lat, 50) else 0.0
    print(f"speedup (P50) = {speedup:.2f}x")

    ok = agreement >= 0.99 and trt_p99 < args.p99_budget_ms
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (agreement>=0.99, trt P99<{args.p99_budget_ms}ms)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
