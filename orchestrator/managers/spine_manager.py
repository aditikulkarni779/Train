"""
Stage 4 — Coordinate spine.

The single source of coach identity + position for the whole AI unit.
Ground truth = axle-pass count (VB fixed formation -> deterministic
coach_index). OCR = LABEL only, never the key. Positions from encoder,
not wall-clock. Every detection inherits the stamp produced here.

Hard invariants (each guards a risk in model_update.md group A):
  A1  axle_count must equal expected(formation); mismatch -> HARD FAIL,
      never silently renumber a train.
  A3  coach_index derives ONLY from axle order; set_ocr_label() can never
      change it — OCR only fills coach_number.
  A4  longitudinal_position_mm comes from encoder positions; a monotonicity
      check rejects wall-clock / jitter.
  A2  ptp_ok=False -> caller must treat zones as per-zone only (flagged).

Dependency-light on purpose: pure logic, unit-testable without the
orchestrator core (see tests/test_spine_manager.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class SpineError(Exception):
    """Raised on an invariant violation. Caller HARD-FAILS the train."""


@dataclass(frozen=True)
class CoachSpec:
    coach_type: str          # LHB | ICF | VB-motor | VB-trailer
    axles: int = 4           # axles per coach (bogies x 2)


@dataclass(frozen=True)
class FormationSpec:
    """Known, fixed rake formation for a Vande Bharat train."""
    coaches: tuple[CoachSpec, ...]

    @property
    def expected_axles(self) -> int:
        return sum(c.axles for c in self.coaches)


@dataclass
class AxleEvent:
    """One axle pass detected by the bogie cams / encoder."""
    seq: int                 # 0-based order of the pass (truth)
    encoder_position_mm: float
    side: Optional[str] = None   # 'L' | 'R' (axle has both; view pairs entry/exit)


@dataclass
class CoachStamp:
    coach_index: int
    coach_type: str
    axle_ids: list[int]
    encoder_positions_mm: list[float]
    coach_number: Optional[str] = None       # OCR label, may be None/late
    ocr_confidence: float = 0.0
    spine_source: str = "axle_count"          # or 'formation_fallback'
    ptp_ok: bool = True

    def stamp_for_axle(self, axle_id: int) -> dict:
        """The per-detection coordinate stamp (goes into detection_record)."""
        i = self.axle_ids.index(axle_id)
        return {
            "coach_index": self.coach_index,
            "coach_type": self.coach_type,
            "axle_id": axle_id,
            "longitudinal_position_mm": self.encoder_positions_mm[i],
        }


class SpineManager:
    def __init__(self, formation: FormationSpec, tau_ocr: float = 0.60, require_ptp: bool = True):
        self.formation = formation
        self.tau_ocr = tau_ocr
        self.require_ptp = require_ptp
        self._coaches: list[CoachStamp] = []

    # ---- build the spine from axle events (truth) -----------------------

    def build(self, axle_events: list[AxleEvent], ptp_ok: bool = True) -> list[CoachStamp]:
        events = sorted(axle_events, key=lambda e: e.seq)

        # A1: axle count must match the known formation.
        if len(events) != self.formation.expected_axles:
            raise SpineError(
                f"axle_count={len(events)} != expected={self.formation.expected_axles}; "
                "HARD FAIL — not renumbering (risk A1)"
            )

        # A4: encoder positions must be monotonic non-decreasing.
        pos = [e.encoder_position_mm for e in events]
        if any(b < a for a, b in zip(pos, pos[1:])):
            raise SpineError("encoder positions non-monotonic — bad/ wall-clock source (risk A4)")

        # slice axle sequence into coaches by the fixed formation.
        stamps: list[CoachStamp] = []
        k = 0
        for ci, coach in enumerate(self.formation.coaches):
            evs = events[k:k + coach.axles]
            stamps.append(CoachStamp(
                coach_index=ci,
                coach_type=coach.coach_type,
                axle_ids=[e.seq for e in evs],
                encoder_positions_mm=[e.encoder_position_mm for e in evs],
                ptp_ok=ptp_ok,
            ))
            k += coach.axles

        if self.require_ptp and not ptp_ok:
            # A2: don't crash — mark so caller degrades to per-zone only.
            for s in stamps:
                s.ptp_ok = False

        self._coaches = stamps
        return stamps

    # ---- OCR is a LABEL only (A3) ---------------------------------------

    def set_ocr_label(self, coach_index: int, coach_number: str, confidence: float) -> None:
        """Attach an OCR reading. Can NEVER change coach_index (risk A3)."""
        s = self._coaches[coach_index]
        if confidence >= self.tau_ocr:
            s.coach_number = coach_number
            s.ocr_confidence = confidence
            s.spine_source = "axle_count"
        else:
            # low-confidence OCR -> keep label but fall back on formation identity
            s.coach_number = None
            s.ocr_confidence = confidence
            s.spine_source = "formation_fallback"

    def coaches(self) -> list[CoachStamp]:
        return self._coaches
