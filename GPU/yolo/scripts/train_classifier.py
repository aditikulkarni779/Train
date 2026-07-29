"""
Specialist trainer — classifiers (YOLO11-cls). One script, three tasks:

  --task defect_state : UNIFIED defect-state classifier. Given a localized
                        component crop -> {ok, broken, missing, hanging,
                        displaced, leaking, damaged, ...}. FOLDS IN (decision
                        2026-07-22) FIBA red_condition + ICF brake sparking as
                        extra classes on RGB crops -> one model, one engine,
                        one training job (lowest latency + retrain).
  --task fastener     : slot-occupancy {fastener_present, fastener_missing}
                        on per-known-slot crops. Low-threshold + this verifier
                        replaces full-frame SAHI (the P3 budget killer).
  --task coach_type   : {LHB, ICF} on the coach panorama -> picks the manifest.

All are YOLO11-cls (ImageFolder layout) -> tiny, FP8, batched -> milliseconds.

Data layout (datasets/SPECIALIST_DATA_LAYOUT.md), ImageFolder:
  datasets/cls_<task>/{train,val,test}/<class_name>/*.png

Release (Spark):
  python GPU/yolo/scripts/train_classifier.py --task defect_state --model yolo11m-cls.pt --epochs 50
Smoke:
  python GPU/yolo/scripts/train_classifier.py --task fastener --model yolo11n-cls.pt --epochs 2 --smoke
"""
from __future__ import annotations
import argparse, datetime, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = REPO / "GPU/yolo/weights"
VERSIONS = WEIGHTS_DIR / "VERSIONS.txt"

TASKS = {
    "defect_state": {"family": "defect_classifier", "name": "defect_state_v1", "imgsz": 224},
    "fastener":     {"family": "p3_fastener",        "name": "p3_fastener_v1",  "imgsz": 128},
    "coach_type":   {"family": "coach_type",         "name": "coach_type_v1",   "imgsz": 320},
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
    ap.add_argument("--model", default="yolo11m-cls.pt")
    ap.add_argument("--data", default=None, help="default datasets/cls_<task> (ImageFolder root)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    spec = TASKS[args.task]
    data = Path(args.data) if args.data else REPO / f"GPU/yolo/datasets/cls_{args.task}"
    imgsz = args.imgsz or spec["imgsz"]
    if not (data / "train").exists():
        sys.exit(f"[err] classifier data missing: {data}/train (ImageFolder; see SPECIALIST_DATA_LAYOUT.md)")

    from ultralytics import YOLO
    model = YOLO(args.model)
    results = model.train(
        data=str(data), epochs=args.epochs, imgsz=imgsz, batch=args.batch,
        name=spec["name"] + ("_smoke" if args.smoke else ""),
        device=args.device, project=str(REPO / "GPU/yolo/runs"),
        # class-imbalanced defect crops: light aug, no mosaic for cls
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, fliplr=0.5,
        exist_ok=True, verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    metric = "smoke"
    try:
        metric = f"top1={results.results_dict.get('metrics/accuracy_top1', 'na'):.4f}"
    except Exception:
        pass
    if best.exists() and not args.smoke:
        dest = WEIGHTS_DIR / f"{spec['name']}.pt"
        shutil.copy2(best, dest)
        log_version(spec["family"], dest.name, metric, f"task={args.task} imgsz={imgsz} ep={args.epochs}")
        print(f"[done] {args.task} classifier -> {dest} ({metric}); FP8 engine on Spark")
    else:
        print(f"[smoke] trained ok metric={metric} — NOT registered")


if __name__ == "__main__":
    main()
