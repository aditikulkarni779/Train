"""
Stage 6.2 — Completeness engine (pure rules, no NN).

Answers "is every expected component present?" per coach. The expected set is
coach-type-conditional (LHB vs ICF expect different parts — taxonomy in
contracts/component_defect_taxonomy.yaml), so a wrong coach_type would flag a
storm of false-missing (risk R14 / C2). We guard that two ways:

  C2  a component is declared MISSING only when it had 0 hits across at least
      `missing_min_views` opportunities. Too few views -> data_unavailable,
      never "missing" and never "present".
  B3  a slot with no coverage (dropped region, or a component the source
      assigned NO camera) is data_unavailable, NEVER reported clean.

Runs on the VOTED present-set (not raw detections) so transient false
positives don't mask a real missing part, and transient misses don't fake one.

Pure logic, unit-testable (see tests/test_completeness_manager.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


PRESENT = "present"
MISSING = "missing"
DISPLACED = "displaced"
DATA_UNAVAILABLE = "data_unavailable"


@dataclass
class ComponentStatus:
    component: str
    status: str
    views_seen: int

    def to_record(self) -> dict:
        return {"component": self.component, "status": self.status, "views_seen": self.views_seen}


class CompletenessManager:
    def __init__(self, config: dict):
        self.missing_min_views: int = config.get("completeness", {}).get("missing_min_views", 2)

    def diff(
        self,
        coach_type: str,
        manifest: dict,
        present_view_counts: dict,
        total_views: int,
        displaced: Optional[set] = None,
        coverage_gap: Optional[set] = None,
    ) -> list[ComponentStatus]:
        """Compare the voted present-set against the coach-type manifest.

        manifest:            {coach_type: [expected component class, ...]}
        present_view_counts: {component: how many views saw it}
        total_views:         how many views/frames covered this coach
        displaced:           components flagged dislocated/displaced by a defect model
        coverage_gap:        components with no camera / dropped coverage (B3)
        """
        displaced = displaced or set()
        coverage_gap = coverage_gap or set()
        expected = manifest.get(coach_type)
        if expected is None:
            raise KeyError(f"no manifest for coach_type={coach_type!r} (risk R14 — refuse to guess)")

        out: list[ComponentStatus] = []
        for component in expected:
            hits = int(present_view_counts.get(component, 0))
            if component in coverage_gap:
                status = DATA_UNAVAILABLE                     # B3: never clean on no coverage
            elif total_views < self.missing_min_views:
                status = DATA_UNAVAILABLE                     # C2: not enough views to judge
            elif hits == 0:
                status = MISSING
            elif component in displaced:
                status = DISPLACED
            else:
                status = PRESENT
            out.append(ComponentStatus(component=component, status=status, views_seen=hits))
        return out

    @staticmethod
    def present_counts_from_flags(flags) -> dict:
        """Build present_view_counts from voted presence flags (votes = views)."""
        counts: dict = {}
        for f in flags:
            cls = getattr(f, "cls", None) or (f.get("class") if isinstance(f, dict) else None)
            votes = getattr(f, "votes", None) or (f.get("votes", 1) if isinstance(f, dict) else 1)
            if cls is not None:
                counts[cls] = counts.get(cls, 0) + int(votes)
        return counts
