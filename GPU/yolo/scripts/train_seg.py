"""
Specialist trainer — segmentation models (YOLO11-seg). One script, two tasks:

  --task crack : P2 crack/corrosion masks. Trains on GATED SAHI tiles (320px)
                 cut from LOSSLESS belly crops. Metric that gates release =
                 Dice/IoU + length-recall (NOT box mAP). FP16 serve.
  --task wheel : P3 wheel shelling/tread-crack masks on LOG-POLAR UNWRAPPED
                 wheel crops (seed circle from known Ø, not blind Hough).
                 Reports shelling LENGTH (>40mm) only; DEPTH out of AI scope.

One arch (YOLO11-seg) for both -> uniform tooling, one TensorRT path, FP16.
Both are GATED at serve (run only inside the anomaly ROI / on wheel crops) so
tile counts stay small -> latency bounded (EDGE §6). Do not remove the gate.

Data layout (datasets/SPECIALIST_DATA_LAYOUT.md):
  datasets/seg_crack/{images,labels}/{train,val,test}   + data.yaml  (polygons)
  datasets/seg_wheel/{images,labels}/{train,val,test}   + data.yaml  (unwrapped)

Release (Spark):
  python GPU/yolo/scripts/train_seg.py --task crack --model yolo11m-seg.pt --epochs 200 --imgsz 320
  python GPU/yolo/scripts/train_seg.py --task wheel --model yolo11m-seg.pt --epochs 200 --imgsz 640
Smoke:
  python GPU/yolo/scripts/train_seg.py --task crack --model yolo11n-seg.pt --epochs 2 --imgsz 320 --smoke
"""
from __future__ import annotations
import argparse, datetime, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = REPO / "GPU/yolo/weights"
VERSIONS = WEIGHTS_DIR / "VERSIONS.txt"

TASKS = {
    "crack": {"family": "p2_crackseg", "name": "p2_under_crackseg_v1", "imgsz": 320},
    "wheel": {"family": "p3_wheel",    "name": "p3_wheel_shelling_v1", "imgsz": 640},
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
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
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--data", default=None, help="default datasets/seg_<task>/data.yaml")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    spec = TASKS[args.task]
    data = Path(args.data) if args.data else REPO / f"GPU/yolo/datasets/seg_{args.task}/data.yaml"
    imgsz = args.imgsz or spec["imgsz"]
    if not data.exists():
        sys.exit(f"[err] seg data.yaml missing: {data} (see SPECIALIST_DATA_LAYOUT.md)")

    from ultralytics import YOLO
    model = YOLO(args.model)
    results = model.train(
        data=str(data), epochs=args.epochs, imgsz=imgsz, batch=args.batch,
        name=spec["name"] + ("_smoke" if args.smoke else ""),
        device=args.device, project=str(REPO / "GPU/yolo/runs"),
        # fine-detail seg: mild aug, copy-paste helps thin-defect recall
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, scale=0.3, copy_paste=0.3,
        exist_ok=True, verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    metric = "smoke"
    try:
        metric = f"mAP50-seg={results.results_dict.get('metrics/mAP50(M)', 'na'):.4f}"
    except Exception:
        pass
    if best.exists() and not args.smoke:
        dest = WEIGHTS_DIR / f"{spec['name']}.pt"
        shutil.copy2(best, dest)
        log_version(spec["family"], dest.name, metric, f"task={args.task} imgsz={imgsz} ep={args.epochs}")
        print(f"[done] {args.task}-seg -> {dest} ({metric}); gate at serve, FP16 engine on Spark")
    else:
        print(f"[smoke] trained ok metric={metric} — NOT registered")


if __name__ == "__main__":
    main()
