"""
Stage 6.1 — Temporal voting + entry/exit fusion.

Raw per-frame detections are noisy: a real defect appears across several
frames, a false positive usually does not. So we do NOT flag on a single
detection. We quantize every detection to a coach-coordinate CELL
(coach_index, longitudinal band, side) and require k-of-n confirmations
before a class in that cell is flagged.

Two guards from the risk register (model_update.md):
  C4  entry/exit wheel fusion keyed on AXLE SEQUENCE, never appearance.
  E2  idempotency key (coach_index, cell, class) so re-ingesting the same
      train does not create duplicate flags in the console.

Pure logic, dependency-light — unit-testable without torch/orchestrator core
(see tests/test_voting_manager.py).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, Optional


# Wheel classes get entry+exit two-view fusion; keyed on axle, not pixels.
WHEEL_CLASSES = {"wheel_shelling", "wheel_shelling_len_gt_40mm", "tread_crack", "wheel_flat"}


@dataclass(frozen=True)
class VoteCell:
    """A coach-coordinate bucket. Detections that fall in the same cell and
    carry the same class are treated as the same physical observation."""
    coach_index: int
    position_band: int          # longitudinal_position_mm // cell_size_mm
    side: Optional[str]         # 'L' | 'R' | None

    def key(self, cls: str) -> tuple:
        return (self.coach_index, self.position_band, self.side, cls)


@dataclass
class VotedFlag:
    coach_index: int
    zone: str
    cls: str
    side: Optional[str]
    position_band: int
    longitudinal_position_mm: float     # representative (mean) position
    conf: float                         # mean confidence of confirming votes
    votes: int                          # number of confirmations
    tier: str
    source_model: Optional[str] = None
    fused_views: tuple = field(default_factory=tuple)   # e.g. ('entry','exit')

    def idempotency_key(self) -> tuple:
        """E2: stable across re-ingest so the console never double-lists."""
        return (self.coach_index, self.position_band, self.side, self.cls)

    def to_record(self) -> dict:
        return {
            "coach_index": self.coach_index,
            "zone": self.zone,
            "class": self.cls,
            "side": self.side,
            "longitudinal_position_mm": self.longitudinal_position_mm,
            "conf": round(self.conf, 4),
            "votes": self.votes,
            "tier": self.tier,
            "source_model": self.source_model,
            "fused_views": list(self.fused_views),
        }


class VotingManager:
    def __init__(self, config: dict):
        v = config.get("voting", {})
        self.k_of_n: int = v.get("k_of_n", 3)
        self.cell_size_mm: float = v.get("cell_size_mm", 100.0)
        self.fuse_entry_exit: bool = v.get("fuse_entry_exit", True)
        # per-class thresholds fall back to the global component threshold
        self.default_tau: float = config.get("confidence", {}).get("component", 0.35)
        self.tau: dict = v.get("tau", {})

    def _tau_for(self, cls: str) -> float:
        return self.tau.get(cls, self.default_tau)

    def _cell(self, det: dict) -> VoteCell:
        pos = float(det.get("longitudinal_position_mm") or 0.0)
        return VoteCell(
            coach_index=det.get("coach_index"),
            position_band=int(pos // self.cell_size_mm),
            side=det.get("side"),
        )

    def vote(self, detections: Iterable[dict]) -> list[VotedFlag]:
        """Accumulate detections into cells, emit only k-of-n confirmed flags.

        Each detection dict: coach_index, zone, class, conf,
        longitudinal_position_mm, side, view(entry|exit|None), tier, source_model.
        """
        # bucket confirming detections per (cell, class, VIEW). View is part of
        # the key so entry and exit vote independently (each needs k-of-n) and
        # are only combined later in _fuse_entry_exit — otherwise same-band
        # entry/exit detections would average together before fusion sees them.
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for det in detections:
            cls = det.get("class")
            if det.get("conf", 0.0) < self._tau_for(cls):
                continue                                   # below threshold: not a vote
            cell = self._cell(det)
            buckets[cell.key(cls) + (det.get("view"),)].append({**det, "_cell": cell})

        flags: list[VotedFlag] = []
        for (coach_index, band, side, cls, view), dets in buckets.items():
            if len(dets) < self.k_of_n:
                continue                                   # not enough confirmations
            confs = [d["conf"] for d in dets]
            positions = [float(d.get("longitudinal_position_mm") or 0.0) for d in dets]
            views = (view,) if view else ()
            flags.append(VotedFlag(
                coach_index=coach_index,
                zone=dets[0].get("zone"),
                cls=cls,
                side=side,
                position_band=band,
                longitudinal_position_mm=mean(positions),
                conf=mean(confs),
                votes=len(dets),
                tier=dets[0].get("tier", "structural"),
                source_model=dets[0].get("source_model"),
                fused_views=views,
            ))

        if self.fuse_entry_exit:
            flags = self._fuse_entry_exit(flags)
        return flags

    def _fuse_entry_exit(self, flags: list[VotedFlag]) -> list[VotedFlag]:
        """C4: for wheel classes, fuse the entry and exit view of the SAME
        physical wheel — keyed on (coach_index, side, class), never appearance.
        Either view flagging keeps recall; both agreeing raises confidence."""
        wheel: dict[tuple, list[VotedFlag]] = defaultdict(list)
        passthrough: list[VotedFlag] = []
        for f in flags:
            if f.cls in WHEEL_CLASSES:
                wheel[(f.coach_index, f.side, f.cls)].append(f)
            else:
                passthrough.append(f)

        fused: list[VotedFlag] = []
        for (coach_index, side, cls), group in wheel.items():
            views = tuple(sorted({v for g in group for v in (g.fused_views or ())}))
            both = len(views) >= 2
            fused.append(VotedFlag(
                coach_index=coach_index,
                zone=group[0].zone,
                cls=cls,
                side=side,
                position_band=group[0].position_band,
                longitudinal_position_mm=mean([g.longitudinal_position_mm for g in group]),
                # both views agree -> take the max (precision); one view -> keep it (recall)
                conf=max(g.conf for g in group) if both else group[0].conf,
                votes=sum(g.votes for g in group),
                tier=group[0].tier,
                source_model=group[0].source_model,
                fused_views=views,
            ))
        return passthrough + fused

    @staticmethod
    def dedupe(flags: Iterable[VotedFlag]) -> list[VotedFlag]:
        """E2: collapse identical idempotency keys (safe re-ingest)."""
        seen: dict[tuple, VotedFlag] = {}
        for f in flags:
            k = f.idempotency_key()
            # keep the higher-confidence instance if a key repeats
            if k not in seen or f.conf > seen[k].conf:
                seen[k] = f
        return list(seen.values())
