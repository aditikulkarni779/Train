"""
Stage 8 (perf) — GATE CASCADE orchestration.

The single biggest latency lever in the design. Raw full-frame SAHI over the
belly + bogie = ~480k tiles/train -> infeasible even on the Spark. So we GATE:
a cheap full-field pass yields a small ROI, and the expensive seg/tiling runs
ONLY inside it.

  P1 side   : detect direct, no SAHI (parts >=15 mm).
  P2 belly  : anomaly ROI mask -> tile (320px @0.20) ONLY inside mask -> crack-seg.
  P3 bogie  : component detect -> wheels -> log-polar unwrap; expected bracket
              SLOTS -> fastener check. NO full-frame SAHI anywhere in P3.

The anomaly/detect gate itself is INJECTED (an ROI list), so this manager is
pure orchestration — testable without the GPU models. It emits a WorkPlan
(the list of specialist jobs) and reports gated-vs-ungated tile counts so the
~20x reduction is measured, not assumed (risk D6/D7).

Pure logic, dependency-light (see tests/test_gate_cascade_manager.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkItem:
    specialist: str        # 'crack_seg' | 'wheel_seg' | 'fastener' | 'detect'
    zone: str              # 'P1' | 'P2' | 'P3'
    tier: str              # 'safety' | 'structural' | 'cosmetic'
    kind: str              # 'tile' | 'crop' | 'slot'
    ref: tuple             # coords / id
    est_cost: float = 1.0  # relative units (1 = one tile-inference)


@dataclass
class WorkPlan:
    items: list = field(default_factory=list)
    ungated_tiles: int = 0     # what a naive full-SAHI would have cost (for the reduction metric)

    def total_cost(self) -> float:
        return sum(i.est_cost for i in self.items)

    def count(self, *, zone: Optional[str] = None, specialist: Optional[str] = None,
              tier: Optional[str] = None) -> int:
        return sum(
            1 for i in self.items
            if (zone is None or i.zone == zone)
            and (specialist is None or i.specialist == specialist)
            and (tier is None or i.tier == tier)
        )

    @property
    def gated_tiles(self) -> int:
        return sum(1 for i in self.items if i.kind == "tile")

    @property
    def reduction_x(self) -> float:
        g = self.gated_tiles
        return (self.ungated_tiles / g) if g else float("inf")


def _sahi_grid(w: int, h: int, tile: int, overlap: float) -> list[tuple]:
    """Top-left coords of every SAHI tile over a w×h frame."""
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    # ensure the far edge is covered
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs]


def _intersects(tx: int, ty: int, tile: int, region: tuple) -> bool:
    rx, ry, rw, rh = region
    return not (tx + tile <= rx or tx >= rx + rw or ty + tile <= ry or ty >= ry + rh)


class GateCascadeManager:
    def __init__(self, config: dict):
        t = config.get("tiling", {})
        p2 = t.get("p2", {})
        self.tile = p2.get("tile", 320)
        self.overlap = p2.get("overlap", 0.20)
        # tiers per specialist (crack/corrosion structural by default; wheel+fastener safety)
        self.tiers = config.get("gate_tiers", {
            "crack_seg": "structural", "wheel_seg": "safety",
            "fastener": "safety", "detect": "structural",
        })

    # ---- P2: tile ONLY inside the anomaly ROI mask -----------------------

    def plan_p2(self, strip_wh: tuple, roi_mask: list) -> WorkPlan:
        """roi_mask = list of (x,y,w,h) hot regions from the anomaly gate."""
        w, h = strip_wh
        grid = _sahi_grid(w, h, self.tile, self.overlap)
        plan = WorkPlan(ungated_tiles=len(grid))
        for (tx, ty) in grid:
            if any(_intersects(tx, ty, self.tile, r) for r in roi_mask):
                plan.items.append(WorkItem("crack_seg", "P2", self.tiers["crack_seg"],
                                           "tile", (tx, ty), est_cost=1.0))
        return plan

    # ---- P3: wheels + expected slots, NEVER full-frame SAHI --------------

    def plan_p3(self, frame_wh: tuple, wheels: list, expected_slots: list) -> WorkPlan:
        """wheels = [(axle_id, side, view, crop_ref), ...]; slots = [slot_ref, ...]."""
        w, h = frame_wh
        # what a naive P3 full-SAHI WOULD have cost (for the reduction metric)
        ungated = len(_sahi_grid(w, h, self.tile, self.overlap))
        plan = WorkPlan(ungated_tiles=ungated)
        for (axle_id, side, view, ref) in wheels:
            plan.items.append(WorkItem("wheel_seg", "P3", self.tiers["wheel_seg"],
                                       "crop", (axle_id, side, view, ref), est_cost=1.0))
        for slot in expected_slots:
            plan.items.append(WorkItem("fastener", "P3", self.tiers["fastener"],
                                       "slot", (slot,), est_cost=0.2))   # slot check is cheap
        return plan

    # ---- P1: detect direct, no SAHI --------------------------------------

    def plan_p1(self, n_detect_tiles: int) -> WorkPlan:
        plan = WorkPlan(ungated_tiles=n_detect_tiles)     # P1 has no SAHI; ungated == gated
        for i in range(n_detect_tiles):
            plan.items.append(WorkItem("detect", "P1", self.tiers["detect"], "tile", (i,), est_cost=1.0))
        return plan

    def merge(self, *plans: WorkPlan) -> WorkPlan:
        out = WorkPlan()
        for p in plans:
            out.items.extend(p.items)
            out.ungated_tiles += p.ungated_tiles
        return out
