#!/usr/bin/env bash
# B4 — export PyTorch YOLO -> ONNX -> TensorRT FP16 engine (run on the GPU host).
#
#   bash scripts/export_trt.sh models/best.pt models/mvis.engine 4
#
# Validates on host; produces an engine the YOLO service loads when
# INFERENCE_BACKEND=trt. Keep batch arg == runtime batch_size in config.yaml.
set -euo pipefail

PT_MODEL="${1:-models/best.pt}"
ENGINE_OUT="${2:-models/mvis.engine}"
BATCH="${3:-4}"
ONNX_OUT="${PT_MODEL%.pt}.onnx"

echo "[1/2] Exporting $PT_MODEL -> $ONNX_OUT (opset 17, batch $BATCH)"
python - "$PT_MODEL" "$BATCH" <<'PY'
import sys
from ultralytics import YOLO
pt, batch = sys.argv[1], int(sys.argv[2])
YOLO(pt).export(format="onnx", opset=17, batch=batch, dynamic=False, simplify=True)
print("ONNX export done")
PY

echo "[2/2] Building TensorRT FP16 engine -> $ENGINE_OUT"
trtexec \
  --onnx="$ONNX_OUT" \
  --saveEngine="$ENGINE_OUT" \
  --fp16 \
  --workspace=4096

echo "Done. Set INFERENCE_BACKEND=trt and inference.engine_path=$ENGINE_OUT"
