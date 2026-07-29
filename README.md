# Train — AI Detection Pipeline

AI detection pipeline for automated rolling-stock (Vande Bharat) inspection.
Consumes the 10 area-camera lossless-PNG feed and produces a per-coach
inspection report. On-site inference on an NVIDIA DGX Spark.

Full developer specification: **[pipeline.md](pipeline.md)**.

**Scope:** detection pipeline only. The underbelly line-scan feed, the OCR
(coach-number) model, and the gap-detection model are handled by separate,
already-deployed systems and are not part of this repository.

## Structure

```
orchestrator/          # AI pipeline (the detection unit)
├── core/              # shared data models, config, logging
├── managers/          # spine · voting · completeness · report ·
│                      #   gate_cascade · degrade (+ reused capture/stitch)
├── inspection_pipeline.py   # end-to-end runner + latency harness
├── config/detection_pipeline_config.yaml   # single source of tunables
└── tests/             # unit tests
GPU/
├── yolo/              # model trainers (backbone, heads, seg, classifier,
│   └── scripts/       #   metrology) + serving + registry
└── shared/            # shared GPU utilities
docs/
├── contracts/         # detection_record, inspection_report, edge handoff,
│                      #   component_defect_taxonomy (label schema)
└── MODEL_INVENTORY.md # the model set (8, metrology folded into the detector)
pipeline.md            # developer specification
```

## Models (detection assembly)

Shared detector (+ metrology keypoints) · defect-state classifier ·
anomaly gate (PatchCore) · crack/corrosion segmentation · wheel segmentation ·
fastener slot-occupancy · coach-type classifier. See `pipeline.md` §5.

## Test

```bash
cd orchestrator && python -m pytest tests/ -q
```

## Requirements

`numpy`, `opencv-python`, `pyyaml`, `pytest`. Model training additionally needs
`ultralytics` / `anomalib` / `torch` (imported lazily by the trainer scripts).
