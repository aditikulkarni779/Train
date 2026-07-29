"""
Stage 5 — Inference worker / demo runner (single process, NO Triton).

Runs the component detector on an image, a folder, or a video, draws boxes,
and writes detection records (the same shape voting_manager will consume).
For the demo this is monolithic: one detector = the shared backbone + a full
component head. Later stages swap `--weights` to shared_backbone_v1.pt and add
the gated specialists — this file does NOT change shape, so no flow disruption.

Examples
--------
# instant demo on side-view images with the existing detector:
python GPU/yolo/scripts/infer.py \
    --weights GPU/yolo/models/component_detector.pt \
    --source "D:/VANDE_BHARAT/Datasets_ALL/6-6-26-Dataset-Clean/valid/images" \
    --out GPU/yolo/runs/demo --conf 0.35

# single image:
python GPU/yolo/scripts/infer.py --weights <w.pt> --source frame.jpg --out out/

# video:
python GPU/yolo/scripts/infer.py --weights <w.pt> --source side.mp4 --out out/
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VID_EXT


def gather_images(src: Path):
    if src.is_dir():
        return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    return [src]


def to_records(result, source_name: str, zone: str | None, model_name: str):
    """Ultralytics Result -> list of detection_record-shaped dicts.
    coach_index/side/view = null in demo (spine wires these at integration)."""
    recs = []
    names = result.names
    b = result.boxes
    if b is None:
        return recs
    for i in range(len(b)):
        cls = int(b.cls[i])
        recs.append({
            "coach_index": None,           # spine stamps this at integration (Stage 4)
            "coach_type": None,
            "zone": zone,                  # e.g. "P1" for side-view demo
            "class": names[cls],
            "conf": round(float(b.conf[i]), 4),
            "bbox_xywhn": [round(float(x), 5) for x in b.xywhn[i].tolist()],
            "source": source_name,
            "source_model": model_name,    # weights that emitted this record
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image | folder | video")
    ap.add_argument("--out", default="GPU/yolo/runs/demo")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--zone", default=None, help="tag records with a zone, e.g. P1")
    ap.add_argument("--no-save-img", action="store_true", help="records only, skip annotated images")
    args = ap.parse_args()

    from ultralytics import YOLO

    wpath = Path(args.weights)
    if not wpath.exists():
        sys.exit(f"[err] weights not found: {wpath}")
    src = Path(args.source)
    if not src.exists():
        sys.exit(f"[err] source not found: {src}")

    out = Path(args.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(wpath))
    print(f"[infer] weights={wpath.name}  classes={len(model.names)}  conf={args.conf}")

    all_records = []
    save_img = not args.no_save_img

    if is_video(src):
        # stream=True keeps memory flat over long videos (single resident process)
        gen = model.predict(source=str(src), conf=args.conf, iou=args.iou,
                            imgsz=args.imgsz, device=args.device, stream=True, verbose=False)
        for fi, r in enumerate(gen):
            recs = to_records(r, f"{src.name}#frame{fi}", args.zone, wpath.name)
            all_records.extend(recs)
            if save_img and recs:
                r.save(filename=str(out / "annotated" / f"{src.stem}_f{fi:06d}.jpg"))
        print(f"[infer] video frames processed, detections={len(all_records)}")
    else:
        images = gather_images(src)
        print(f"[infer] {len(images)} image(s)")
        for img in images:
            r = model.predict(source=str(img), conf=args.conf, iou=args.iou,
                              imgsz=args.imgsz, device=args.device, verbose=False)[0]
            recs = to_records(r, img.name, args.zone, wpath.name)
            all_records.extend(recs)
            if save_img:
                r.save(filename=str(out / "annotated" / f"{img.stem}.jpg"))

    # write detection records (voting_manager-shaped) + a per-class summary
    rec_path = out / "detections.json"
    rec_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": wpath.name,
        "conf": args.conf,
        "count": len(all_records),
        "records": all_records,
    }, indent=2), encoding="utf-8")

    summary = {}
    for rc in all_records:
        summary[rc["class"]] = summary.get(rc["class"], 0) + 1
    print(f"[done] {len(all_records)} detections -> {rec_path}")
    print("[done] annotated -> " + str(out / "annotated"))
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"    {v:4d}  {k}")


if __name__ == "__main__":
    main()
