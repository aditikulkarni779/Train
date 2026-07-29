"""
Stage 8 (perf) — bounded FIFO + DEGRADE policy.

Single Spark, serial, no dispatch gate (C2/R1). Bursty back-to-back trains can
push the queue past the 8-min budget. Under pressure we shed load by TIER —
never by dropping capture, never safety:

    cosmetic  ->  structural  ->  [SAFETY NEVER DROPPED]

Two triggers:
  1. work cost for a train exceeds the per-train budget (tiles the Spark can
     clear in the remaining time), or
  2. the bounded queue crosses its high-watermark (too many trains waiting).

Dropped items are RECORDED, not silently lost — the report marks those zones
`coverage=degraded`, never clean (risk E6). Safety items that would blow the
budget are kept and the overrun is flagged (post-departure SLA, R1) rather than
skipping a safety check.

Pure logic, dependency-light (see tests/test_degrade_manager.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DEGRADE_ORDER = ["cosmetic", "structural"]     # safety intentionally absent


@dataclass
class DegradeResult:
    kept: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    over_budget_safety: bool = False           # safety kept but budget blown -> flag, don't drop
    reason: Optional[str] = None

    def kept_cost(self) -> float:
        return sum(getattr(i, "est_cost", 1.0) for i in self.kept)

    def dropped_tiers(self) -> set:
        return {getattr(i, "tier", None) for i in self.dropped}


class DegradeManager:
    def __init__(self, config: dict):
        d = config.get("degrade", {})
        self.high_watermark = d.get("queue_high_watermark", 0.8)
        self.order = d.get("order", DEGRADE_ORDER)

    def apply(self, items: list, budget_cost: float, queue_load: float = 0.0) -> DegradeResult:
        """items: WorkItem-like (have .tier and .est_cost).
        budget_cost: max cost the device can clear for this train in time.
        queue_load: current queue depth fraction [0..1]."""
        total = sum(getattr(i, "est_cost", 1.0) for i in items)
        pressured = total > budget_cost or queue_load >= self.high_watermark
        if not pressured:
            return DegradeResult(kept=list(items), dropped=[], reason=None)

        kept = list(items)
        dropped: list = []
        # shed cosmetic first, then structural, until under budget — safety never touched
        for tier in self.order:
            if sum(getattr(i, "est_cost", 1.0) for i in kept) <= budget_cost and queue_load < self.high_watermark:
                break
            shed = [i for i in kept if getattr(i, "tier", None) == tier]
            kept = [i for i in kept if getattr(i, "tier", None) != tier]
            dropped.extend(shed)

        kept_cost = sum(getattr(i, "est_cost", 1.0) for i in kept)
        over_safety = kept_cost > budget_cost      # only safety left and still over
        reason = "queue_watermark" if queue_load >= self.high_watermark else "tile_budget"
        return DegradeResult(kept=kept, dropped=dropped, over_budget_safety=over_safety, reason=reason)


@dataclass
class BoundedTrainFIFO:
    """Per-train FIFO with a hard cap. Never drops a train silently — a full
    queue rejects new enqueue so the edge keeps spilling (nothing lost)."""
    capacity: int = 4
    _q: list = field(default_factory=list)

    def load(self) -> float:
        return len(self._q) / self.capacity if self.capacity else 1.0

    def enqueue(self, train_id) -> bool:
        if len(self._q) >= self.capacity:
            return False                # rejected -> caller alerts, edge holds on spill
        self._q.append(train_id)
        return True

    def dequeue(self):
        return self._q.pop(0) if self._q else None
