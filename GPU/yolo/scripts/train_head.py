"""
Stage 3 — Detect-head training on the frozen shared backbone.

One head per zone (p1_side / p2_under / p3_bogie). Starts from the Stage-2
backbone weights, FREEZES the backbone layers, fine-tunes only the head on a
zone-filtered view of the union (classes whose zone == target, reindexed).

Zone -> class assignment comes from datasets/class_map.yaml (PROPOSED there;
confirm before a release run). Zone-filtered labels are generated here from
the union labels — no new labeling, pure reindex.

Full run (on Spark, after backbone exists):
  python GPU/yolo/scripts/train_head.py --zone p3 \
      --backbone GPU/yolo/weights/shared_backbone_v1.pt \
      --epochs 300 --imgsz 1280 --batch -1

Smoke:
  python GPU/yolo/scripts/train_head.py --zone p3 --backbone yolo11n.pt \
      --epochs 2 --imgsz 320 --batch 8 --smoke
"""
from __future__ import annotations
import argparse, datetime, shutil, subprocess, sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
DS = REPO / "GPU/yolo/datasets"
UNION = DS / "union"
WEIGHTS_DIR = REPO / "GPU/yolo/weights"
VERSIONS = WEIGHTS_DIR / "VERSIONS.txt"
ZONE_HEAD_NAME = {"p1": "p1_side", "p2": "p2_under", "p3": "p3_bogie"}


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def zone_classes(zone: str):
    """Return {union_id: (name, new_zone_id)} for the target zone."""
    cm = yaml.safe_load((DS / "class_map.yaml").read_text(encoding="utf-8"))
    z = zone.upper()
    sel = [(uid, c["name"]) for uid, c in cm["components"].items() if c["zone"] == z]
    sel.sort()
    return {uid: (name, i) for i, (uid, name) in enumerate(sel)}


def build_zone_view(zone: str):
    """Filter union labels to the zone's classes, reindex, write zone data.yaml.
    Returns the zone data.yaml path."""
    mapping = zone_classes(zone)             # union_id -> (name, zone_id)
    zdir = DS / zone
    names = [None] * len(mapping)
    for uid, (name, zid) in mapping.items():
        names[zid] = name

    for split in ("train", "val", "test"):
        src_list = UNION / f"{split}.txt"
        if not src_list.exists():
            continue
        out_lbl = zdir / split / "labels"
        out_lbl.mkdir(parents=True, exist_ok=True)
        kept_imgs = []
        for img_line in src_list.read_text(encoding="utf-8").splitlines():
            img = Path(img_line.strip())
            if not img_line.strip():
                continue
            ulbl = UNION / split / "labels" / f"{img.stem}.txt"
            if not ulbl.exists():
                continue
            lines = []
            for ln in ulbl.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                uid = int(p[0])
                if uid in mapping:
                    lines.append(f"{mapping[uid][1]} {' '.join(p[1:5])}")
            if lines:                          # keep only images with zone objects
                (out_lbl / f"{img.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                kept_imgs.append(str(img.resolve()))
        (zdir / f"{split}.txt").write_text("\n".join(kept_imgs) + "\n", encoding="utf-8")
        print(f"[{zone}] {split}: {len(kept_imgs)} images with zone objects")

    data_yaml = zdir / "data.yaml"
    lines = [
        f"train: {(zdir / 'train.txt').resolve()}",
        f"val: {(zdir / 'val.txt').resolve()}",
        f"test: {(zdir / 'test.txt').resolve()}",
        f"nc: {len(names)}",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(names)]
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{zone}] classes ({len(names)}): {names}")
    return data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=["p1", "p2", "p3"])
    ap.add_argument("--backbone", default=str(WEIGHTS_DIR / "shared_backbone_v1.pt"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--freeze", type=int, default=10, help="freeze first N layers (backbone)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    if not Path(args.backbone).exists() and not args.smoke:
        sys.exit(f"[err] backbone weights missing: {args.backbone} — run Stage 2 first")

    data_yaml = build_zone_view(args.zone)
    name = ZONE_HEAD_NAME[args.zone] + ("_smoke" if args.smoke else "_v1")

    model = YOLO(args.backbone)
    results = model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        freeze=args.freeze,                       # freeze backbone, train head only
        name=name, device=args.device,
        project=str(REPO / "GPU/yolo/runs"),
        mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5,
        exist_ok=True, verbose=True,
    )

    metric = "smoke"
    try:
        metric = f"mAP50={results.results_dict.get('metrics/mAP50(B)', 'na'):.4f}"
    except Exception:
        pass
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists() and not args.smoke:
        dest = WEIGHTS_DIR / f"{ZONE_HEAD_NAME[args.zone]}_v1.pt"
        shutil.copy2(best, dest)
        line = f"{ZONE_HEAD_NAME[args.zone]} | {dest.name} | {git_hash()} | {metric} | {datetime.date.today()} | freeze={args.freeze} zone={args.zone}\n"
        with open(VERSIONS, "a", encoding="utf-8") as f:
            f.write(line)
        print("[versions] " + line.strip())
        print(f"[done] {args.zone} head -> {dest} ({metric})")
    else:
        print(f"[smoke] {args.zone} head trained ok, best={best} metric={metric} — NOT registered")


if __name__ == "__main__":
    main()
