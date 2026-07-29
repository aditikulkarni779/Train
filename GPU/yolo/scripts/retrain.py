"""
Manual-trigger retrain script (Phase 9). NOT scheduled/automatic — run by hand
when enough reviewer feedback (false_positive / false_negative / confirmed
entries in defect_review_logs) has accumulated to justify a retrain. Per the
architecture review's own recommendation: automate only once volume justifies it.

Flow:
  1. Pull a training-export manifest (produced by `POST /api/training/export`)
  2. Download each referenced frame from Cloudinary
  3. Build a YOLO-format dataset (binary defect/normal, per the existing
     project decision — see progress.md "Binary YOLO model" row) from the
     false_negative / confirmed samples; false_positive samples are excluded
     from positive labels (they tell you what NOT to reinforce)
  4. Fine-tune from a base .pt checkpoint with ultralytics
  5. Upload the resulting best.pt to Cloudinary
  6. Register the new weights as a 'staging' ModelVersion via the backend API
     (does NOT activate it — promotion is a separate, deliberate
     `PATCH /api/models/:id/activate` call, ideally after a manual eval pass)

Usage:
  python retrain.py \
    --manifest-url https://res.cloudinary.com/.../export-169.../manifest.json \
    --base-model models/best.pt \
    --epochs 50 \
    --api-base http://localhost:8001 \
    --api-token <admin JWT> \
    --model-name defect_detector
"""
import argparse
import json
import os
import tempfile
import time

import requests


def download(url: str, dest: str):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)


def build_dataset(manifest: dict, workdir: str) -> str:
    """
    Builds a minimal YOLO dataset.yaml + images/labels split from manifest
    samples. Binary single-class: 0 = defect. false_negative/confirmed samples
    with a bbox become positive labels; false_positive samples are downloaded
    too but get NO label file (background-only image), which teaches the
    model the AI's own past mistake was not a defect.
    """
    from PIL import Image

    images_dir = os.path.join(workdir, "images", "train")
    labels_dir = os.path.join(workdir, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    kept = 0
    for i, sample in enumerate(manifest["samples"]):
        if not sample.get("frame_url"):
            continue
        img_path = os.path.join(images_dir, f"sample_{i}.jpg")
        try:
            download(sample["frame_url"], img_path)
        except Exception as exc:
            print(f"  skip sample {i}: download failed ({exc})")
            continue

        label_path = os.path.join(labels_dir, f"sample_{i}.txt")
        if sample["label_type"] in ("false_negative", "confirmed") and sample.get("bbox"):
            bbox = sample["bbox"]
            # bbox stored as absolute pixel coords (bbox_x/y/w/h on the Defect
            # row). Normalize against the actual downloaded image's dimensions
            # (frame resolution varies by camera) into YOLO's cx,cy,w,h in 0-1.
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                with open(label_path, "w") as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            except Exception as exc:
                print(f"  skip sample {i}: bbox normalization failed ({exc})")
                os.remove(img_path)
                continue
        # false_positive samples: image saved, no label file -> background example
        kept += 1

    dataset_yaml = os.path.join(workdir, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {workdir}\ntrain: images/train\nval: images/train\nnc: 1\nnames: ['defect']\n")

    print(f"Dataset built: {kept} samples in {workdir}")
    return dataset_yaml


def train(base_model: str, dataset_yaml: str, epochs: int, workdir: str) -> str:
    from ultralytics import YOLO

    model = YOLO(base_model)
    results = model.train(data=dataset_yaml, epochs=epochs, project=workdir, name="retrain", exist_ok=True)
    best_path = os.path.join(workdir, "retrain", "weights", "best.pt")
    if not os.path.exists(best_path):
        raise RuntimeError(f"Training finished but {best_path} not found")
    return best_path, results


def upload_weights(local_path: str, version: str) -> str:
    import cloudinary
    import cloudinary.uploader

    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
    )
    result = cloudinary.uploader.upload(
        local_path, resource_type="raw", folder="vande/model-weights", public_id=version,
    )
    return result["secure_url"]


def register_version(api_base: str, api_token: str, model_name: str, version: str, weights_url: str, metrics: dict, manifest_url: str):
    resp = requests.post(
        f"{api_base}/api/models",
        headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
        json={"model_name": model_name, "version": version, "weights_url": weights_url, "metrics": metrics, "trained_from": manifest_url},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-url", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--api-base", default="http://localhost:8001")
    ap.add_argument("--api-token", required=True, help="Admin JWT — obtain via POST /api/auth/login")
    ap.add_argument("--model-name", default="defect_detector")
    args = ap.parse_args()

    manifest = requests.get(args.manifest_url, timeout=30).json()
    print(f"Manifest: {manifest['sample_count']} samples, {manifest['counts_by_label']}")

    with tempfile.TemporaryDirectory() as workdir:
        dataset_yaml = build_dataset(manifest, workdir)
        version = f"{args.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"

        best_path, results = train(args.base_model, dataset_yaml, args.epochs, workdir)
        metrics = {"precision": float(results.box.p.mean()) if hasattr(results, "box") else None}

        weights_url = upload_weights(best_path, version)
        registered = register_version(args.api_base, args.api_token, args.model_name, version, weights_url, metrics, args.manifest_url)

        print(json.dumps(registered, indent=2))
        print(f"\nRegistered as STAGING. Promote with:\n  PATCH {args.api_base}/api/models/{registered['version']['id']}/activate")


if __name__ == "__main__":
    main()
