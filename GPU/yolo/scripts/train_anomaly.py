"""
Specialist trainer — P2 belly anomaly GATE (PatchCore, Anomalib).

DECISION (2026-07-22): gate model = PatchCore (accuracy). PatchCore inference
is a kNN over a memory bank, so latency + memory grow with bank size. Because
this gate runs full-field on EVERY train and gates all downstream seg, two
latency guards are MANDATORY and baked in here:
  1. --coreset-ratio (default 0.01) subsamples the memory bank to ~1% -> bounds
     the kNN cost. Do NOT raise without re-measuring gate latency.
  2. input = the DOWNSAMPLED belly strip (edge bulk_ref), never full-res.
PatchCore is fit on CONFIRMED-NORMAL frames only; one defect poisons it -> the
data layout keeps a human-verified normal_dir (purity gate, risk F1).

No backprop -> "training" = memory-bank fit (~minutes), the cheapest retrain.

Data layout (see datasets/SPECIALIST_DATA_LAYOUT.md):
  datasets/anomaly_p2_belly/
    normal/            # confirmed-normal downsampled belly strips (fit set)
    test/good/         # held-out normal (AUROC)
    test/defect/       # known-anomaly strips (AUROC)

Release (Spark):
  python GPU/yolo/scripts/train_anomaly.py --zone p2_belly --coreset-ratio 0.01
Smoke (few imgs, proves plumbing):
  python GPU/yolo/scripts/train_anomaly.py --zone p2_belly --smoke
"""
from __future__ import annotations
import argparse, datetime, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = REPO / "GPU/yolo/weights"
VERSIONS = WEIGHTS_DIR / "VERSIONS.txt"


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
    ap.add_argument("--zone", default="p2_belly")
    ap.add_argument("--data-root", default=None, help="override; default datasets/anomaly_<zone>")
    ap.add_argument("--coreset-ratio", type=float, default=0.01,
                    help="memory-bank subsample; LATENCY knob — keep small")
    ap.add_argument("--image-size", type=int, default=256, help="downsampled strip input")
    ap.add_argument("--name", default="p2_under_anomaly_v1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else REPO / f"GPU/yolo/datasets/anomaly_{args.zone}"
    normal_dir = root / "normal"
    if not normal_dir.exists():
        sys.exit(f"[err] confirmed-normal set missing: {normal_dir} (see SPECIALIST_DATA_LAYOUT.md)")

    # Lazy import — anomalib is heavy and only needed at train time.
    try:
        from anomalib.data import Folder
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
    except Exception as exc:
        sys.exit(f"[err] anomalib not installed ({exc}). pip install anomalib (train box only).")

    datamodule = Folder(
        name=args.zone,
        root=str(root),
        normal_dir="normal",
        normal_test_dir="test/good",
        abnormal_dir="test/defect",
        image_size=(args.image_size, args.image_size),
    )
    model = Patchcore(coreset_sampling_ratio=args.coreset_ratio)   # LATENCY guard #1
    engine = Engine()
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)

    metric = "smoke" if args.smoke else f"AUROC={_auroc(results)}"
    if not args.smoke:
        dest = WEIGHTS_DIR / f"{args.name}.ckpt"
        engine.export(model=model, export_root=str(WEIGHTS_DIR), export_type="torch")
        log_version("p2_anomaly", args.name, metric, f"coreset={args.coreset_ratio} img={args.image_size}")
        print(f"[done] anomaly gate -> {dest} ({metric})")
        print("[next] export to ONNX/engine ON the Spark; keep coreset small for gate latency")
    else:
        print(f"[smoke] fit ok, {metric} — NOT registered")


def _auroc(results) -> str:
    try:
        r = results[0] if isinstance(results, list) else results
        for k, v in r.items():
            if "AUROC" in k or "auroc" in k:
                return f"{float(v):.4f}"
    except Exception:
        pass
    return "na"


if __name__ == "__main__":
    main()
