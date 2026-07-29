"""
Stage 6.3 — InspectionReport assembly -> Report generator.

Turns the voted flags + completeness table into one per-coach
InspectionReport (contracts/inspection_report.schema.json) and hands it off.

Guardrails from the risk register (model_update.md):
  E6  a zone whose model did not run is coverage=data_unavailable/degraded,
      NEVER emitted as clean. In the detection-only slice the specialist zones
      (crack-seg, wheel-seg, fastener, metrology) have not shipped yet, so they
      are declared un-inspected here — the report must not imply they passed.
  C3  sub-mm defects are out of scope at 55 MB downsampling -> their zones stay
      data_unavailable until specialists land.
  E1  validate the report shape before handoff so the Report generator never
      rejects a malformed payload mid-integration.

Pure assembly, unit-testable (see tests/test_report_manager.py).
"""
from __future__ import annotations

import time
from typing import Optional


# Zones the detection-only slice does NOT yet inspect -> always flagged.
SPECIALIST_ZONES_NOT_SHIPPED = ["p2_crack_corrosion", "p3_wheel_seg", "p3_fastener", "p1_metrology"]

_TIER_RANK = {"cosmetic": 1, "structural": 2, "safety": 3}
_REQUIRED_KEYS = {"coach_index", "coach_type", "manifest_status", "defects", "coverage", "generated_at"}


class ReportManager:
    def __init__(self, config: dict):
        self.config = config

    def build(
        self,
        stamp,                          # CoachStamp (spine_manager) or dict-like
        voted_flags,                    # list[VotedFlag]
        manifest_status,                # list[ComponentStatus]
        latency_s: float,
        coverage: Optional[list] = None,
        dropped_regions: Optional[list] = None,
    ) -> dict:
        coach_index = getattr(stamp, "coach_index", None) if not isinstance(stamp, dict) else stamp.get("coach_index")
        coach_type = getattr(stamp, "coach_type", None) if not isinstance(stamp, dict) else stamp.get("coach_type")
        coach_number = getattr(stamp, "coach_number", None) if not isinstance(stamp, dict) else stamp.get("coach_number")
        ptp_ok = getattr(stamp, "ptp_ok", True) if not isinstance(stamp, dict) else stamp.get("ptp_ok", True)

        defects = [f.to_record() if hasattr(f, "to_record") else f for f in voted_flags]
        manifest_rows = [s.to_record() if hasattr(s, "to_record") else s for s in manifest_status]

        report = {
            "coach_index": coach_index,
            "coach_number": coach_number,
            "coach_type": coach_type,
            "generated_at": _utc_now(),
            "latency_s": round(latency_s, 3),
            "manifest_status": manifest_rows,
            "defects": defects,
            "defect_map_ref": None,
            "coverage": coverage if coverage is not None else self._default_coverage(ptp_ok, dropped_regions),
            "review_status": self._review_status(defects, manifest_rows),
            "worst_tier": self._worst_tier(defects),
        }
        return report

    def _default_coverage(self, ptp_ok: bool, dropped_regions: Optional[list]) -> list:
        cov = [
            {"zone": "P1", "region": "side", "state": "inspected"},
            {"zone": "P2", "region": "underbelly_components", "state": "inspected"},
            {"zone": "P3", "region": "bogie_components", "state": "inspected"},
        ]
        # E6/C3: specialist zones have not shipped in this slice -> not clean.
        for z in SPECIALIST_ZONES_NOT_SHIPPED:
            cov.append({"zone": z, "region": "specialist", "state": "data_unavailable"})
        # A2: no PTP -> cross-zone fusion invalid, degrade coverage honestly.
        if not ptp_ok:
            for c in cov:
                if c["state"] == "inspected":
                    c["state"] = "degraded"
        # B3: dropped regions are data_unavailable, never clean.
        for dr in (dropped_regions or []):
            cov.append({"zone": "P?", "region": f"dropped:{dr}", "state": "data_unavailable"})
        return cov

    @staticmethod
    def _worst_tier(defects: list) -> str:
        rank = 0
        for d in defects:
            rank = max(rank, _TIER_RANK.get(d.get("tier"), 0))
        return {0: "none", 1: "cosmetic", 2: "structural", 3: "safety"}[rank]

    @staticmethod
    def _review_status(defects: list, manifest_rows: list) -> str:
        has_safety = any(_TIER_RANK.get(d.get("tier"), 0) == 3 for d in defects)
        has_missing = any(r.get("status") == "missing" for r in manifest_rows)
        return "needs_review" if (has_safety or has_missing) else "auto_pass"

    @staticmethod
    def validate(report: dict) -> None:
        """E1: cheap shape check before handoff (no jsonschema dependency)."""
        missing = _REQUIRED_KEYS - set(report)
        if missing:
            raise ValueError(f"InspectionReport missing required keys: {sorted(missing)} (risk E1)")
        if not isinstance(report["defects"], list) or not isinstance(report["coverage"], list):
            raise ValueError("InspectionReport defects/coverage must be lists (risk E1)")
        # never emit an empty coverage list — that would imply nothing was checked
        if not report["coverage"]:
            raise ValueError("InspectionReport coverage is empty — refusing to imply full clean (risk E6)")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
