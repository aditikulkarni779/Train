"""
Stage 2 — Shared backbone training (YOLO11m on the 32-class union).

Build FIRST: every detect head (Stage 3) attaches to this frozen backbone.
No MLflow: on success, append one line to weights/VERSIONS.txt
(family | filename | git_hash | metric | date | notes).

Full run (on DGX Spark):
  python GPU/yolo/scripts/train_backbone.py --model yolo11m.pt --epochs 300 \
      --imgsz 1280 --batch -1 --name shared_backbone_v1

Smoke (prove the pipeline anywhere):
  python GPU/yolo/scripts/train_backbone.py --model yolo11n.pt --epochs 2 \
      --imgsz 320 --batch 8 --name smoke --smoke
"""
from __future__ import annotations
import argparse, datetime, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]          # D:/VandeBharat
WEIGHTS_DIR = REPO / "GPU/yolo/weights"
VERSIONS = WEIGHTS_DIR / "VERSIONS.txt"
DEFAULT_DATA = REPO / "GPU/yolo/datasets/union/data.yaml"


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "nogit"


def log_version(family: str, filename: str, metric: str, notes: str):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{family} | {filename} | {git_hash()} | {metric} | {datetime.date.today()} | {notes}\n"
    with open(VERSIONS, "a", encoding="utf-8") as f:
        f.write(line)
    print("[versions] " + line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt", help="base weights (yolo11m for real, yolo11n for smoke)")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto (uses ~60%% VRAM)")
    ap.add_argument("--name", default="shared_backbone_v1")
    ap.add_argument("--device", default=None, help="cuda idx or 'cpu'; None = auto")
    ap.add_argument("--smoke", action="store_true", help="tiny sanity run, not a release")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = Path(args.data)
    if not data.exists():
        sys.exit(f"[err] data.yaml missing: {data} — run rebuild_splits.py first")

    model = YOLO(args.model)
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        project=str(REPO / "GPU/yolo/runs"),
        # backbone recipe (plan §Stage 2): mosaic + HSV + scale aug
        mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5,
        exist_ok=True,
        verbose=True,
    )

    # locate best.pt, copy into weights/ under the release name
    best = Path(results.save_dir) / "weights" / "best.pt"
    metric = "smoke" if args.smoke else "see-results"
    try:
        metric = f"mAP50={results.results_dict.get('metrics/mAP50(B)', 'na'):.4f}"
    except Exception:
        pass

    if best.exists() and not args.smoke:
        dest = WEIGHTS_DIR / f"{args.name}.pt"
        shutil.copy2(best, dest)
        log_version("shared_backbone", dest.name, metric, f"imgsz={args.imgsz} ep={args.epochs}")
        print(f"[done] backbone -> {dest}  ({metric})")
        print("[next] build engine ON the Spark: GPU/yolo/scripts/export_engine.py")
    else:
        print(f"[smoke] trained ok, best={best} metric={metric} — NOT registered (smoke run)")


if __name__ == "__main__":
    main()
