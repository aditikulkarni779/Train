"""
NVIDIA-GPU-tuned trainer for the component detector (28-class union).

Faster than train_backbone.py: RAM image cache, high dataloader workers,
auto-batch to fill VRAM, AMP mixed precision, cos-LR, multi-GPU ready.
Registers the result to weights/VERSIONS.txt (no MLflow). Same class space
and output as the backbone — later swap into the pipeline unchanged.

Quick start (single GPU, fast):
  python GPU/yolo/scripts/train.py --model yolo11m.pt --epochs 150

Max speed / big GPU:
  python GPU/yolo/scripts/train.py --model yolo11m.pt --epochs 150 \
      --imgsz 960 --batch -1 --cache ram --workers 16 --device 0

Two GPUs:
  python GPU/yolo/scripts/train.py --model yolo11m.pt --device 0,1 --batch 64

Resume a crashed run:
  python GPU/yolo/scripts/train.py --resume
"""
from __future__ import annotations
import argparse, datetime, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]                 # D:/VandeBharat
WEIGHTS_DIR = REPO / "GPU/yolo/weights"
VERSIONS = WEIGHTS_DIR / "VERSIONS.txt"
DEFAULT_DATA = REPO / "GPU/yolo/datasets/union/data.yaml"


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def print_gpu():
    import torch
    if not torch.cuda.is_available():
        print("[warn] CUDA NOT available — will run on CPU (slow). Check driver / torch build.")
        return "cpu"
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"[gpu {i}] {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
    print(f"[gpu] torch {torch.__version__}  cuda {torch.version.cuda}  devices={n}")
    return "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt", help="base weights (yolo11n/s/m/l/x)")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto-fill ~60%% VRAM; set explicit for multi-GPU")
    ap.add_argument("--device", default="0", help="'0' | '0,1' | 'cpu'")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers (raise on Linux; keep ~8 on Windows)")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "false"],
                    help="ram = fastest if dataset fits RAM; disk = 2nd; false = off")
    ap.add_argument("--name", default="component_v1")
    ap.add_argument("--patience", type=int, default=50, help="early-stop patience")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--freeze", type=int, default=0, help=">0 to freeze first N layers (head-only fine-tune)")
    ap.add_argument("--smoke", action="store_true", help="tiny sanity run, not registered")
    args = ap.parse_args()

    from ultralytics import YOLO

    dev_kind = print_gpu()
    device = "cpu" if dev_kind == "cpu" else args.device
    cache = False if args.cache == "false" else args.cache

    data = Path(args.data)
    if not data.exists():
        sys.exit(f"[err] data.yaml missing: {data} — run rebuild_splits.py first")

    if args.resume:
        # resume finds last.pt in the run dir
        model = YOLO(str(REPO / "GPU/yolo/runs" / args.name / "weights" / "last.pt"))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            cache=cache,                 # RAM cache -> big speedup after epoch 1
            amp=True,                    # mixed precision -> faster + less VRAM
            cos_lr=True,                 # smoother convergence for fine-tunes
            freeze=args.freeze or None,
            patience=args.patience,
            name=args.name,
            project=str(REPO / "GPU/yolo/runs"),
            # detection aug recipe (matches backbone/head trainers)
            mosaic=1.0, close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5,
            exist_ok=True, verbose=True, plots=True,
        )

    best = Path(results.save_dir) / "weights" / "best.pt"
    metric = "smoke"
    try:
        metric = f"mAP50={results.results_dict.get('metrics/mAP50(B)', 'na'):.4f}"
    except Exception:
        pass

    if best.exists() and not args.smoke:
        dest = WEIGHTS_DIR / f"{args.name}.pt"
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        line = f"component | {dest.name} | {git_hash()} | {metric} | {datetime.date.today()} | imgsz={args.imgsz} ep={args.epochs} model={args.model}\n"
        with open(VERSIONS, "a", encoding="utf-8") as f:
            f.write(line)
        print("[versions] " + line.strip())
        print(f"[done] weights -> {dest}  ({metric})")
        print(f"[run]  infer:  python GPU/yolo/scripts/infer.py --weights {dest} --source <img|dir|video> --out GPU/yolo/runs/demo")
    else:
        print(f"[smoke] trained ok, best={best} metric={metric} — NOT registered")


if __name__ == "__main__":
    main()
