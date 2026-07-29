"""
Stage 8 (start) — end-to-end AI-unit runner + latency harness.

Wires the whole detection slice in ONE process and times every stage:

    axle events ─► SpineManager ─► (attach stamps to detections)
        ─► VotingManager ─► CompletenessManager ─► ReportManager ─► InspectionReport(s)

Detections are INJECTED (a list, or anything iterable), not produced here, so
the path is testable on any box without torch/GPU: tests feed fixtures, the
live worker feeds `infer.py` output (same record shape). This is the
measurement foundation — you cannot refine performance you have not measured,
and this is the first time the full chain runs together.

Per-stage wall-clock is captured against the 8-min budget (risk D1). Nothing
here is GPU-bound; the detector's own latency is measured separately and added
as the `detect` stage when a real inference source is wired.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from core.logging_config import get_stage_logger
from managers.voting_manager import VotingManager
from managers.completeness_manager import CompletenessManager, MISSING, DISPLACED
from managers.report_manager import ReportManager

log = get_stage_logger("inspection_pipeline")

DEFAULT_BUDGET_S = 8 * 60      # 8-min input->detection budget


class StageTimer:
    """Records wall-clock per named stage. `with timer('vote'): ...`"""
    def __init__(self):
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def __call__(self, name: str):
        return _StageCtx(self, name)

    def total_s(self) -> float:
        return time.perf_counter() - self._t0


class _StageCtx:
    def __init__(self, timer: StageTimer, name: str):
        self._timer, self._name = timer, name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self._start
        self._timer.stages[self._name] = self._timer.stages.get(self._name, 0.0) + dt
        return False


@dataclass
class LatencyBreakdown:
    stages_s: dict = field(default_factory=dict)
    total_s: float = 0.0
    budget_s: float = DEFAULT_BUDGET_S

    @property
    def over_budget(self) -> bool:
        return self.total_s > self.budget_s

    @property
    def bottleneck(self) -> Optional[str]:
        return max(self.stages_s, key=self.stages_s.get) if self.stages_s else None

    def table(self) -> str:
        rows = [f"  {k:<14} {v*1000:8.1f} ms" for k, v in sorted(self.stages_s.items(), key=lambda x: -x[1])]
        head = f"total {self.total_s*1000:.1f} ms / budget {self.budget_s*1000:.0f} ms" \
               f" ({'OVER' if self.over_budget else 'ok'}); bottleneck={self.bottleneck}"
        return head + "\n" + "\n".join(rows)


@dataclass
class RunResult:
    reports: list          # list[InspectionReport dict], one per coach
    latency: LatencyBreakdown
    n_detections: int
    n_voted: int


class InspectionPipeline:
    def __init__(self, config: dict, formation, manifest: dict,
                 defect_classes: Optional[set] = None,
                 component_tier: Optional[dict] = None):
        # formation: spine_manager.FormationSpec ; manifest: {coach_type: [component,...]}
        from managers.spine_manager import SpineManager
        self.config = config
        self.formation = formation
        self.manifest = manifest
        self.defect_classes = defect_classes or set()       # empty => detector emits presence only
        self.component_tier = component_tier or {}
        self.budget_s = config.get("sla", {}).get("budget_s", DEFAULT_BUDGET_S)

        self.spine = SpineManager(
            formation,
            tau_ocr=config.get("spine", {}).get("tau_ocr", 0.60),
            require_ptp=config.get("spine", {}).get("require_ptp", True),
        )
        self.voting = VotingManager(config)
        self.completeness = CompletenessManager(config)
        self.report = ReportManager(config)

    def run(
        self,
        axle_events,
        detections: Iterable[dict],
        views_per_coach: Optional[dict] = None,
        ptp_ok: bool = True,
        dropped_regions: Optional[dict] = None,
    ) -> RunResult:
        views_per_coach = views_per_coach or {}
        dropped_regions = dropped_regions or {}
        timer = StageTimer()
        detections = list(detections)

        with timer("spine"):
            stamps = self.spine.build(axle_events, ptp_ok=ptp_ok)

        with timer("attach"):
            detections = self._attach_stamps(detections, stamps)

        with timer("vote"):
            voted = self.voting.vote(detections)

        presence = [f for f in voted if f.cls not in self.defect_classes]
        defect_flags = [f for f in voted if f.cls in self.defect_classes]

        reports = []
        for stamp in stamps:
            ci = stamp.coach_index
            coach_presence = [f for f in presence if f.coach_index == ci]
            coach_defects = [f for f in defect_flags if f.coach_index == ci]

            with timer("completeness"):
                status = self.completeness.diff(
                    coach_type=stamp.coach_type,
                    manifest=self.manifest,
                    present_view_counts=CompletenessManager.present_counts_from_flags(coach_presence),
                    total_views=int(views_per_coach.get(ci, 0)),
                )

            # missing/displaced components become defect records too (tier from map)
            derived = self._defects_from_status(status, ci)

            with timer("report"):
                rep = self.report.build(
                    stamp=stamp,
                    voted_flags=[f.to_record() for f in coach_defects] + derived,
                    manifest_status=status,
                    latency_s=timer.total_s(),
                    dropped_regions=dropped_regions.get(ci),
                )
                ReportManager.validate(rep)
            reports.append(rep)

        latency = LatencyBreakdown(stages_s=dict(timer.stages), total_s=timer.total_s(), budget_s=self.budget_s)
        log.info("inspection complete", extra={"extra_fields": {
            "coaches": len(reports), "detections": len(detections), "voted": len(voted),
            "total_ms": round(latency.total_s * 1000, 1), "over_budget": latency.over_budget,
            "bottleneck": latency.bottleneck,
        }})
        return RunResult(reports=reports, latency=latency, n_detections=len(detections), n_voted=len(voted))

    # ---- helpers --------------------------------------------------------

    def _attach_stamps(self, detections: list, stamps: list) -> list:
        """Fill coach_index + longitudinal_position_mm from the spine when a
        detection carries axle_id but not coach_index. Detections already
        coach-assigned (side components) pass through unchanged."""
        by_axle = {}
        for s in stamps:
            for a in s.axle_ids:
                by_axle[a] = s
        out = []
        for d in detections:
            d = dict(d)
            if d.get("coach_index") is None and d.get("axle_id") is not None:
                s = by_axle.get(d["axle_id"])
                if s is not None:
                    st = s.stamp_for_axle(d["axle_id"])
                    d["coach_index"] = st["coach_index"]
                    d.setdefault("longitudinal_position_mm", st["longitudinal_position_mm"])
                    d["coach_type"] = st["coach_type"]
            out.append(d)
        return out

    def _defects_from_status(self, status, coach_index: int) -> list:
        out = []
        for s in status:
            if s.status in (MISSING, DISPLACED):
                out.append({
                    "coach_index": coach_index,
                    "zone": None,
                    "class": f"{s.component}_{s.status}",
                    "conf": 1.0 if s.status == MISSING else 0.9,
                    "votes": s.views_seen,
                    "tier": self.component_tier.get(s.component, "structural"),
                    "source_model": "completeness_engine",
                })
        return out
