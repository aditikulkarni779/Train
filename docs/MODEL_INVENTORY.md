# FINAL MODEL INVENTORY

Device: DGX Spark (GB10), all models resident. Rule: same architecture reused across zones = one model.

## Count: **8 trainable models + 2 non-model components**

| # | Model | Type | Zone |
|---|---|---|---|
| 1 | Shared detector (backbone + zone heads) + **metrology keypoints** | detector + pose | P1/P2/P3 |
| 2 | Defect-state classifier (incl. FIBA-red, sparking) | classifier | P1/P2/P3 |
| 3 | Anomaly gate (PatchCore) | anomaly, no-backprop | P2 |
| 4 | Crack / corrosion segmentation | segmentation | P2 |
| 5 | Wheel-shelling segmentation (log-polar) | segmentation | P3 |
| 6 | Fastener slot-occupancy | classifier | P3 |
| 7 | OCR (coach number) + Gap Detection | text recognition | spine |
| 8 | Coach-type classifier (LHB / ICF) | classifier | spine |
| — | Completeness engine | rules (no model) | P1/P2 |
| — | Coordinate spine | deterministic (no model) | all |


## What was NOT merged and cost if merged
- **Crack segmentation** stays dedicated — thin/hairline cracks and length measurement are lost if merged into boxes/classifier.
- **Wheel segmentation** stays dedicated — merging with crack-seg costs Dice/recall on two safety tasks.
- **Corrosion** may later fold into the defect-state classifier (loses area precision, keeps detection) — optional, not done.

Non-model items (completeness, spine) never train and never use the GPU.
