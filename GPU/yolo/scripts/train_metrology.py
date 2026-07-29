"""
Specialist trainer — geometry/metrology (YOLO11-pose keypoints). One script,
two tasks. Produces MEASUREMENTS (mm), not classes — the RDSO defects here are
metric thresholds, so a classifier can't answer them (decision Q3, 2026-07-22).

  --task buffer  : ICF buffer height. Keypoints = buffer-face-center + rail
                   reference -> pixels converted to mm via per-site scale
                   calibration -> {ok if 1030-1105mm, too_low, mismatch}.
  --task coupler : IV coupler sag. Keypoints = coupler + reference line ->
                   droop magnitude -> {ok, sagging}.

The MODEL learns keypoints only; pixels->mm is a CALIBRATION step (rail scale
from fixed camera geometry), NOT trained — lives in the serve wrapper. Tiny,
few crops/train -> negligible latency.

Data layout (datasets/SPECIALIST_DATA_LAYOUT.md), YOLO-pose:
  datasets/pose_<task>/{images,labels}/{train,val,test} + data.yaml (kpt_shape)

Release (Spark):
  python GPU/yolo/scripts/train_metrology.py --task buffer --model yolo11m-pose.pt --epochs 150
Smoke:
  python GPU/yolo/scripts/train_metrology.py --task coupler --model yolo11n-pose.pt --epochs 2 --smoke
"""
from __future__ import annotations
import argparse, datetime, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = REPO / "GPU/yolo/weights"
VERSIONS = WEIGHTS_DIR / "VERSIONS.txt"

TASKS = {
    "buffer":  {"family": "p1_buffer_metrology", "name": "p1_buffer_metrology_v1", "imgsz": 640},
    "coupler": {"family": "p1_sag_pose",          "name": "p1_sag_pose_v1",         "imgsz": 640},
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
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--data", default=None, help="default datasets/pose_<task>/data.yaml")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    spec = TASKS[args.task]
    data = Path(args.data) if args.data else REPO / f"GPU/yolo/datasets/pose_{args.task}/data.yaml"
    imgsz = args.imgsz or spec["imgsz"]
    if not data.exists():
        sys.exit(f"[err] pose data.yaml missing: {data} (see SPECIALIST_DATA_LAYOUT.md)")

    from ultralytics import YOLO
    model = YOLO(args.model)
    results = model.train(
        data=str(data), epochs=args.epochs, imgsz=imgsz, batch=args.batch,
        name=spec["name"] + ("_smoke" if args.smoke else ""),
        device=args.device, project=str(REPO / "GPU/yolo/runs"),
        # geometry must not be distorted: NO scale/shear/perspective aug
        degrees=0.0, shear=0.0, perspective=0.0, scale=0.0, mosaic=0.0,
        exist_ok=True, verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    metric = "smoke"
    try:
        metric = f"pose_mAP50={results.results_dict.get('metrics/mAP50(P)', 'na'):.4f}"
    except Exception:
        pass
    if best.exists() and not args.smoke:
        dest = WEIGHTS_DIR / f"{spec['name']}.pt"
        shutil.copy2(best, dest)
        log_version(spec["family"], dest.name, metric, f"task={args.task} imgsz={imgsz} ep={args.epochs}")
        print(f"[done] {args.task} metrology -> {dest} ({metric}); pixels->mm calibration in serve wrapper")
    else:
        print(f"[smoke] trained ok metric={metric} — NOT registered")


if __name__ == "__main__":
    main()
