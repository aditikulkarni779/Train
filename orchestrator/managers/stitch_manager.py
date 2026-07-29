"""Cam1 + Cam2 stitching.

Delegates to the project's real stitcher, `stitch.py` (repo root) — a
calibrated vertical 1x2 stitch for the two IDS GV-5040 cameras, mounted one
above the other with a known, fixed 15% overlap band. That module is reused
directly (imported, not copied): the calibrated-offset + alpha-blend path is
robust on bare metal where feature matching has nothing to grab onto, with
an optional ORB-based drift correction (`use_features`) layered on top of it.

This file no longer does its own feature-matching/homography — that was a
placeholder written before `stitch.py` existed. Everything here is now just
the orchestrator-side concern: pairing Cam1/Cam2 processed frames by
timestamp (unrelated to the stitch algorithm itself) and calling
`stitch_vertical()` once per pair.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from core.logging_config import get_stage_logger
from core.models import ProcessedFrame
from managers.frame_assignment_manager import EventFrames

log = get_stage_logger("stitch_manager")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stitch import stitch_vertical  # noqa: E402  (repo-root module, reused as-is)


@dataclass
class StitchedImage:
    timestamp_ms: float
    image: np.ndarray
    method: str  # "calibrated" | "calibrated_features" | "skipped_size_mismatch"


def _pair_frames(
    top: List[ProcessedFrame], bottom: List[ProcessedFrame], epsilon_ms: float
) -> List[Tuple[ProcessedFrame, ProcessedFrame]]:
    pairs = []
    used_bottom = set()
    for tf in top:
        best_idx, best_diff = None, None
        for i, bf in enumerate(bottom):
            if i in used_bottom:
                continue
            diff = abs(tf.timestamp_ms - bf.timestamp_ms)
            if diff <= epsilon_ms and (best_diff is None or diff < best_diff):
                best_idx, best_diff = i, diff
        if best_idx is not None:
            used_bottom.add(best_idx)
            pairs.append((tf, bottom[best_idx]))
    return pairs


class StitchManager:
    def __init__(
        self,
        top_camera: str,
        bottom_camera: str,
        pair_epsilon_ms: float,
        overlap_ratio: float = 0.15,
        use_feature_refinement: bool = False,
    ):
        self._top_camera = top_camera
        self._bottom_camera = bottom_camera
        self._pair_epsilon_ms = pair_epsilon_ms
        self._overlap_ratio = overlap_ratio
        self._use_feature_refinement = use_feature_refinement

    def stitch_event(self, event_frames: EventFrames) -> List[StitchedImage]:
        top_frames = event_frames.frames_by_camera.get(self._top_camera, [])
        bottom_frames = event_frames.frames_by_camera.get(self._bottom_camera, [])

        if not top_frames or not bottom_frames:
            log.warning(
                "cannot stitch — one or both cameras have zero assigned frames",
                extra={"extra_fields": {
                    "event_id": event_frames.event_id,
                    "top_count": len(top_frames), "bottom_count": len(bottom_frames),
                }},
            )
            return []

        pairs = _pair_frames(top_frames, bottom_frames, self._pair_epsilon_ms)
        if not pairs:
            log.warning(
                "no timestamp-aligned Cam1/Cam2 pairs within epsilon — stitching skipped",
                extra={"extra_fields": {"event_id": event_frames.event_id}},
            )
            return []

        results = [self._stitch_pair(event_frames.event_id, top, bottom) for top, bottom in pairs]
        results = [r for r in results if r is not None]

        log.info(
            "stitching complete",
            extra={"extra_fields": {
                "event_id": event_frames.event_id, "pairs": len(pairs), "stitched": len(results),
                "unpaired_top": len(top_frames) - len(pairs),
                "unpaired_bottom": len(bottom_frames) - len(pairs),
            }},
        )
        return results

    def _stitch_pair(
        self, event_id: str, top_frame: ProcessedFrame, bottom_frame: ProcessedFrame
    ):
        avg_ts = (top_frame.timestamp_ms + bottom_frame.timestamp_ms) / 2.0
        try:
            image = stitch_vertical(
                top_frame.image, bottom_frame.image,
                overlap_ratio=self._overlap_ratio,
                use_features=self._use_feature_refinement,
            )
        except ValueError:
            log.warning(
                "stitch skipped — Cam1/Cam2 frame sizes did not match",
                extra={"extra_fields": {
                    "event_id": event_id,
                    "top_shape": top_frame.image.shape, "bottom_shape": bottom_frame.image.shape,
                }},
            )
            return None
        except Exception:
            log.exception(
                "stitch of pair failed unexpectedly — skipping this pair",
                extra={"extra_fields": {"event_id": event_id}},
            )
            return None

        method = "calibrated_features" if self._use_feature_refinement else "calibrated"
        return StitchedImage(timestamp_ms=avg_ts, image=image, method=method)
