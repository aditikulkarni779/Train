"""
B4 — cross-platform TensorRT engine export (Windows-friendly, no bash/trtexec).

Uses ultralytics' built-in TensorRT exporter (Python API) to build an FP16 engine
straight from the .pt. The resulting .engine loads via YOLO("x.engine") with the
same API as the .pt, so it runs through the existing TorchBackend.

    python GPU/yolo/scripts/export_engine.py GPU/yolo/models/best.pt --batch 4

Requires: pip install tensorrt   (torch+CUDA already present).
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_model")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("ERROR: tensorrt not installed. Run:  "
              "GPU\\yolo\\venv\\Scripts\\pip install tensorrt", file=sys.stderr)
        sys.exit(2)

    from ultralytics import YOLO
    print(f"Exporting {args.pt_model} -> TensorRT FP16 engine (batch {args.batch}, imgsz {args.imgsz})")
    out = YOLO(args.pt_model).export(
        format="engine", half=True, batch=args.batch, imgsz=args.imgsz,
        device=args.device, dynamic=False, simplify=True, verbose=True,
    )
    # ultralytics writes alongside the .pt (best.engine). Report the path.
    engine_path = out if isinstance(out, str) else os.path.splitext(args.pt_model)[0] + ".engine"
    print(f"DONE. Engine: {engine_path}")
    print("Set INFERENCE_BACKEND=trt and inference.engine_path to this path "
          "(or load directly via YOLO(engine) in the bench).")


if __name__ == "__main__":
    main()
