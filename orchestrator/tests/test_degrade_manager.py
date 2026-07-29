from managers.degrade_manager import DegradeManager, BoundedTrainFIFO
from managers.gate_cascade_manager import WorkItem

CFG = {"degrade": {"queue_high_watermark": 0.8, "order": ["cosmetic", "structural"]}}


def _items():
    return [
        WorkItem("crack_seg", "P2", "cosmetic", "tile", (0,), 1.0),
        WorkItem("crack_seg", "P2", "cosmetic", "tile", (1,), 1.0),
        WorkItem("detect", "P1", "structural", "tile", (2,), 1.0),
        WorkItem("wheel_seg", "P3", "safety", "crop", (3,), 1.0),
        WorkItem("fastener", "P3", "safety", "slot", (4,), 1.0),
    ]


def test_under_budget_keeps_everything():
    res = DegradeManager(CFG).apply(_items(), budget_cost=100)
    assert len(res.kept) == 5 and res.dropped == []


def test_over_budget_drops_cosmetic_first():
    # budget 3 -> must shed 2 cost; cosmetic (2 items) go first
    res = DegradeManager(CFG).apply(_items(), budget_cost=3)
    assert res.dropped_tiers() == {"cosmetic"}
    assert all(i.tier != "cosmetic" for i in res.kept)
    assert res.reason == "tile_budget"


def test_safety_never_dropped_even_if_over_budget():
    # budget 1 -> shed cosmetic + structural, but 2 safety remain > budget
    res = DegradeManager(CFG).apply(_items(), budget_cost=1)
    assert {i.tier for i in res.kept} == {"safety"}
    assert res.over_budget_safety is True         # flagged, NOT dropped
    assert "cosmetic" in res.dropped_tiers() and "structural" in res.dropped_tiers()


def test_high_queue_load_triggers_shedding():
    res = DegradeManager(CFG).apply(_items(), budget_cost=100, queue_load=0.9)
    assert res.reason == "queue_watermark"
    assert res.dropped != []                      # pressured despite budget headroom


def test_bounded_fifo_rejects_when_full_never_silent():
    q = BoundedTrainFIFO(capacity=2)
    assert q.enqueue("t1") is True
    assert q.enqueue("t2") is True
    assert q.enqueue("t3") is False               # full -> reject, edge holds on spill
    assert q.load() == 1.0
    assert q.dequeue() == "t1"
    assert q.enqueue("t3") is True                # space freed
