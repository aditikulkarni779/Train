# YOLO Model Files

Place your trained model weights here:

| File | Used by | Endpoint |
|---|---|---|
| `best.pt` | Defect detection (Phase 3) | `POST /api/yolo/predict` |
| `train_num_detector.pt` | Bogie ROI for OCR (Phase 2) | `POST /api/yolo/predict_train_number` |

Both models are loaded at server startup (`GPU/yolo/server.py`).
The server starts cleanly even if files are missing — it returns `503` on those endpoints until the files are present.

These files are git-ignored. Never commit trained weights to the repo.
