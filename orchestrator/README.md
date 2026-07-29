# Railway AI Inspection System — Phase-1 Orchestration Layer

Standalone demo of the **Event-Driven Framing and Processing Pipeline**
described in `Railway_AI_Inspection_System_ADS.md`, built for a 3-camera
subset (OCR camera + Cam1 + Cam2) ahead of the full 12-camera rollout.

This package is **new orchestration code only**. It reuses, without
modifying, four things that already existed in the repository:

| Reused component | Where it actually lives | How this package calls it |
|---|---|---|
| Gap Detection model | `GPU/yolo` (`train_num_detector.pt`, "gap" class) | HTTP, `services/yolo_client.py` |
| OCR Detection model | `GPU/ocr` (`ocr_engine.py`, `preprocess.py`, `train_number_filter.py`) | direct Python import, `services/ocr_engine_adapter.py` |
| Component Detection model | `GPU/yolo` (`best.pt`) | HTTP, `services/yolo_client.py` |
| Stitching logic | `stitch.py` (repo root) | direct Python import, `managers/stitch_manager.py` |

`stitch.py` implements the real rig geometry: Cam1 and Cam2 are mounted
**vertically** (top/bottom), not side-by-side, with a known, fixed 15%
overlap band (IDS GV-5040, GSD 0.15 mm/px). It places both frames at a
calibrated offset and alpha-blends the shared band — robust on bare,
low-texture metal where ORB feature matching has nothing to grab onto — with
an optional ORB-based drift-correction pass (`stitch_use_feature_refinement`
in config) layered on top for small mechanical misalignment. An earlier
version of this file used raw ORB+homography stitching as a placeholder
before `stitch.py` existed; that code is gone, replaced entirely by the real
one.

This orchestrator does **not** touch `backend/`, Prisma, Postgres, Redis, or
Cloudinary — it is a self-contained process with its own file-based output,
by design (see the conversation's architecture-conflict discussion: the
existing `backend/` pipeline is batch/DB-driven; this phase demonstrates the
real-time framing logic standalone before any integration decision is made).

## Why the pipeline is structured this way

- **OCR never blocks preprocessing.** `OCRManager` runs on its own thread
  pool, polling `ProcessedFrameStore` independently of
  `PreprocessingManager`'s pool. A slow or hung OCR call cannot stall frame
  capture or preprocessing.
- **Gap Detection creates the Event; OCR only enriches it.** `EventManager`
  only ever transitions state in response to `GapDetectionManager` boundaries
  (`on_gap_boundary`) or stream end (`close_final_window`). OCR results only
  ever call `enrich_with_ocr`, which updates metadata on an event that
  already exists — it can never create one.
- **Event IDs, not coach numbers, are the primary key.** `EVT_000001`,
  `EVT_000002`, ... — assigned the instant a window opens, before OCR has
  even run. `coach_number` is nullable metadata that can arrive late or never
  (`identity_state: unresolved`).
- **Frame assignment uses closed intervals, never equality.**
  `EventWindow.contains()` is `start_ts_ms <= ts <= end_ts_ms`; see
  `FrameAssignmentManager.assign()`.

## Folder naming: train number / coach number

Output is **not** `Session_001/EVT_000001/...` once OCR resolves — folders
are renamed live:

```
<train_number>/                 e.g. 22439/  (see caveat below)
    <coach_number>/              e.g. 12345/
        event.json                event_id preserved here even after rename
        metadata.json
        OCR.json
        Cam1/  Cam2/  Stitched/  Detection/
```

Until OCR resolves, folders sit under their stable fallback names
(`Session_001/EVT_000002/...`) — `event_id` never changes, only the folder
label. `StorageManager.rename_event_for_coach_number()` and
`rename_session_for_train_number()` do the rename on disk, both under one
lock so a rename can never land mid-write (see the module docstring in
`managers/storage_manager.py`).

**Caveat — there is no separate train-number signal.** The OCR model reads a
per-coach number, not a whole-train number; nothing in this pipeline (or the
existing `services/sync_engine`) extracts one. As a pragmatic stand-in, the
**first coach number resolved in the session** also renames the session
folder — meaning the top-level directory ends up named after whichever
coach happened to be OCR'd first, not a true train/rake number. If a real
train-number source becomes available later (operator input, a dedicated
locomotive view, a scheduling system), swap it in at
`PipelineCoordinator._on_identity_resolved` — that's the only place this
proxy logic lives.

**Collisions:** if two coaches resolve to the same number (misread, or a
genuine duplicate), the second one gets `_dup2`, `_dup3`, ... appended —
never silently overwritten.

## Directory layout

```
orchestrator/
  core/            data models, YAML config loader, structured logging
  managers/        the 11 managers named in the brief, one file each
  services/        thin reuse-wrappers around existing GPU/yolo + GPU/ocr code
  config/          pipeline_config.yaml (all tunables)
  tests/           pytest unit tests (pure logic, no live camera/model needed)
  docs/            architecture.md — UML / sequence / thread / queue diagrams
  sample_videos/   place your 3 demo video files here
  main.py          entry point
```

## Prerequisites

1. Python 3.10+ (developed/tested on 3.14).
2. `pip install -r requirements.txt`
3. The existing `GPU/yolo` service running and reachable (default
   `http://127.0.0.1:5002`) — start it exactly as it already runs today:
   ```
   cd GPU/yolo && uvicorn server:app --port 5002
   ```
   If it is unreachable, `YoloClient` logs a warning and returns empty
   detections per call — the pipeline degrades (events get
   `identity_unresolved` / empty detections) instead of crashing.
4. `rapidocr` + `onnxruntime` installed for real OCR text recognition (same
   dependency `GPU/ocr` already needs). If missing, `OCRManager` still runs —
   gap detection and component detection are unaffected, only coach-number
   recognition degrades to `identity_unresolved`.

## Running the demo

1. Put 3 video files in `sample_videos/`: `ocr_camera.mp4`, `cam1.mp4`, `cam2.mp4`
   (or point `config/pipeline_config.yaml` at wherever they actually are).
2. Start the YOLO service (see above).
3. From `orchestrator/`:
   ```
   python main.py --config config/pipeline_config.yaml
   ```
4. A live Tkinter window opens showing per-event progress (falls back to a
   console table if Tkinter/display isn't available). Structured JSON logs
   stream to stdout.
5. Output lands in `./output/Session_001/EVT_000001/`, `EVT_000002/`, ... —
   each with `event.json`, `metadata.json`, `OCR.json`, `Cam1/`, `Cam2/`,
   `Stitched/`, `Detection/detection.json`.

## Expanding to 12 cameras

Add entries to the `cameras:` list in `pipeline_config.yaml` — no manager
code changes required. `FrameAssignmentManager`, `StitchManager`
(currently hardcoded to `cam1`/`cam2` via config, not code), and
`DetectionManager` all read camera names from config. Cameras beyond
Cam1/Cam2 that aren't part of the stitched pair are captured and assigned
identically; a real deployment would extend `StitchManager` (or add
per-bank managers) to handle the undercarriage/wheel banks per ADS Ch. 3 —
that is out of scope for this phase.

## Logging example

Every manager logs structured JSON lines, one stage per logger name:

```json
{"ts": 1731600000.123, "level": "INFO", "stage": "gap_detection", "msg": "gap boundary detected", "boundary_ts_ms": 1731600012345.0, "confidence": 0.87, "cluster_size": 3}
{"ts": 1731600000.456, "level": "INFO", "stage": "event_manager", "msg": "event window bounded", "event_id": "EVT_000001", "start_ts_ms": 1731599999000.0, "end_ts_ms": 1731600012345.0, "boundary_confidence": 0.87}
{"ts": 1731600001.001, "level": "WARNING", "stage": "frame_assignment", "msg": "no frames found for camera in event window — marking data_unavailable", "event_id": "EVT_000002", "camera": "cam2"}
{"ts": 1731600001.900, "level": "INFO", "stage": "storage_manager", "msg": "event finalized", "event_id": "EVT_000001", "status": "STORED"}
```

## Error handling reference

| Failure | Behaviour |
|---|---|
| Camera disconnected / file won't open | `FrameCaptureManager` logs error, closes its ring buffer, signals finished — other cameras unaffected |
| Missing/failed OCR | `identity_state` stays `unresolved`; event still stores; flagged in `warnings` |
| Gap detection produces zero boundaries | Whole video becomes one Event, closed by `close_final_window` at stream end |
| Missing frames for a camera in a window | Logged, `no_frames_assigned:<camera>` warning, other camera(s) still processed |
| Late OCR (arrives after storage) | `EventManager.enrich_with_ocr` → `on_event_updated` → `StorageManager.update_metadata` patches `metadata.json`/`OCR.json` in place |
| Duplicate/out-of-order frames | `ProcessedFrameStore` is timestamp-sorted via bisect, not append-order dependent |
| Ring buffer overflow | Oldest frame dropped, counter incremented, warning logged — never silent, never a crash |
| Storage failure (disk full, permissions) | `StorageManager.save_event` catches, logs, flags `FAILED_PARTIAL` on the event, pipeline continues |

## Running tests

```
python -m pytest tests/ -v
```

19 tests cover gap clustering, event window state transitions, frame
interval assignment, stitching pairing/fallback logic, and config loading —
all pure-logic, no live camera or model service required.

## What is explicitly out of scope for this phase

- Real camera hardware, PTP, or encoder triggering (ADS Ch. 3) — timestamps
  are a synthetic monotonic clock derived from each video's target fps.
- The other 6 cameras / remaining detection models from the full ADS scope.
- Any change to `backend/`, Prisma schema, Redis queues, or Cloudinary — this
  is intentionally a parallel, non-invasive demo track.
