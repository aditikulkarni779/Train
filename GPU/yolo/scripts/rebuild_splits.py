"""
Stage 1 — Dataset split builder for component detection.

Consumes multiple YOLO source datasets, remaps every label into the union
class space (GPU/yolo/datasets/class_map.yaml), and emits train/val/test for
the shared backbone. No images copied — writes remapped label files + image
path-list txts (Ultralytics reads txt file lists). Windows-safe, no symlinks.

HARD RULES enforced here (each maps to a risk in model_update.md):
  * defect/excluded classes DROPPED from detection labels (leakage never ships) [F1]
  * synthetic/augmented sources -> TRAIN split ONLY, never val/test           [F1]
  * image-hash dedup: any image present in val/test is REMOVED from train
    (eval never contaminated by a training image)                             [F1]
  * emits split_report.json: per-class counts/split, box-size histogram,
    train<->eval overlap count, dropped-label counts                          [C2]

Usage:
  python GPU/yolo/scripts/rebuild_splits.py \
      --sources "D:/VANDE_BHARAT/Datasets_ALL" \
      --out GPU/yolo/datasets/union \
      --class-map GPU/yolo/datasets/class_map.yaml
  add --dry-run to analyze without writing labels.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
from collections import defaultdict
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def norm(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def load_class_map(path: Path):
    m = yaml.safe_load(path.read_text(encoding="utf-8"))
    union = {norm(v["name"]): int(k) for k, v in m["components"].items()}
    aliases = {norm(k): norm(v) for k, v in (m.get("aliases") or {}).items()}
    exclude = {norm(x) for x in (m.get("exclude_classes") or [])}
    markers = [x.lower() for x in (m.get("synthetic_markers") or [])]
    return union, aliases, exclude, markers


def source_names(data_yaml: Path):
    """Return {source_idx: normalized_name} from a dataset data.yaml."""
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = d.get("names")
    out = {}
    if isinstance(names, dict):
        for k, v in names.items():
            out[int(k)] = norm(v)
    elif isinstance(names, list):
        for i, v in enumerate(names):
            out[i] = norm(v)
    return out


def img_for_label(lbl: Path):
    # roboflow names contain dots (frame_jpg.rf.HASH.txt) -> never use with_suffix.
    img_dir = lbl.parent.parent / "images"
    stem = lbl.name[:-4] if lbl.name.lower().endswith(".txt") else lbl.stem
    for e in IMG_EXT:
        for ext in (e, e.upper()):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def is_synthetic(path: Path, markers) -> bool:
    s = str(path).lower()
    return any(m in s for m in markers)


def split_of(label_path: Path) -> str:
    parts = [p.lower() for p in label_path.parts]
    if any(p in ("valid", "val") for p in parts):
        return "val"
    if "test" in parts:
        return "test"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="root holding YOLO datasets")
    ap.add_argument("--out", default="GPU/yolo/datasets/union")
    ap.add_argument("--class-map", default="GPU/yolo/datasets/class_map.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    union, aliases, exclude, markers = load_class_map(Path(args.class_map))
    out = Path(args.out)
    src_root = Path(args.sources)

    # discover datasets = any dir containing a data.yaml with a labels tree
    data_yamls = [p for p in src_root.rglob("data.yaml")]
    print(f"[scan] {len(data_yamls)} data.yaml found under {src_root}")

    # split -> list of (img_path, remapped_label_lines)
    staged = {"train": [], "val": [], "test": []}
    stats = {
        "per_class": defaultdict(lambda: defaultdict(int)),   # cls -> split -> count
        "box_area_bins": defaultdict(int),                     # coarse size histogram
        "dropped_labels": defaultdict(int),                    # reason -> count
        "sources": [],
    }

    for dy in data_yamls:
        names = source_names(dy)
        ds_dir = dy.parent
        labels = list(ds_dir.rglob("labels/*.txt"))
        if not labels:
            continue
        synth = is_synthetic(ds_dir, markers)
        stats["sources"].append({"path": str(ds_dir), "labels": len(labels), "synthetic": synth})
        print(f"[src] {ds_dir}  labels={len(labels)}  synthetic={synth}")

        for lbl in labels:
            img = img_for_label(lbl)
            if img is None:
                stats["dropped_labels"]["no_image"] += 1
                continue
            split = "train" if synth else split_of(lbl)   # synthetic -> train only
            kept = []
            for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    sid = int(float(parts[0]))
                except ValueError:
                    continue
                sname = names.get(sid)
                if sname is None:
                    stats["dropped_labels"]["unknown_source_id"] += 1
                    continue
                sname = aliases.get(sname, sname)
                if sname in exclude:
                    stats["dropped_labels"][f"excluded:{sname}"] += 1
                    continue
                uid = union.get(sname)
                if uid is None:
                    stats["dropped_labels"][f"not_in_union:{sname}"] += 1
                    continue
                kept.append(f"{uid} {' '.join(parts[1:5])}")
                stats["per_class"][sname][split] += 1
                try:
                    area = float(parts[3]) * float(parts[4])
                    b = "tiny(<0.01)" if area < 0.01 else "small(<0.05)" if area < 0.05 else \
                        "med(<0.2)" if area < 0.2 else "large"
                    stats["box_area_bins"][b] += 1
                except ValueError:
                    pass
            if kept:
                staged[split].append((img, kept))

    # image-hash dedup: remove any TRAIN image whose hash appears in val/test
    eval_hashes = set()
    for sp in ("val", "test"):
        for img, _ in staged[sp]:
            eval_hashes.add(sha1(img))
    before = len(staged["train"])
    staged["train"] = [(img, k) for img, k in staged["train"] if sha1(img) not in eval_hashes]
    removed = before - len(staged["train"])
    stats["dropped_labels"]["train_leaked_into_eval_removed"] = removed
    print(f"[dedup] removed {removed} train images that also appear in val/test")

    report = {
        "out": str(out),
        "counts": {sp: len(v) for sp, v in staged.items()},
        "per_class": {c: dict(d) for c, d in sorted(stats["per_class"].items())},
        "box_area_bins": dict(stats["box_area_bins"]),
        "dropped_labels": dict(stats["dropped_labels"]),
        "sources": stats["sources"],
        "union_classes": len(union),
    }

    if not args.dry_run:
        for sp, items in staged.items():
            lbl_dir = out / sp / "labels"
            lbl_dir.mkdir(parents=True, exist_ok=True)
            list_path = out / f"{sp}.txt"
            with open(list_path, "w", encoding="utf-8") as lst:
                for img, kept in items:
                    (lbl_dir / f"{img.stem}.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
                    lst.write(str(img.resolve()) + "\n")
        (out / "data.yaml").write_text(_data_yaml(out, union), encoding="utf-8")

    rep_path = out / "split_report.json"
    out.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[report] {rep_path}")
    print(json.dumps(report["counts"], indent=2))
    print("dropped:", json.dumps(report["dropped_labels"], indent=2))


def _data_yaml(out: Path, union: dict) -> str:
    names = [None] * len(union)
    for name, uid in union.items():
        names[uid] = name
    lines = [
        f"train: {(out / 'train.txt').resolve()}",
        f"val: {(out / 'val.txt').resolve()}",
        f"test: {(out / 'test.txt').resolve()}",
        f"nc: {len(union)}",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
