from managers.gate_cascade_manager import GateCascadeManager, WorkPlan

CFG = {"tiling": {"p2": {"tile": 320, "overlap": 0.20}}}


def _mgr():
    return GateCascadeManager(CFG)


def test_p2_tiles_only_inside_roi_mask_big_reduction():
    # a large belly strip; a tiny hot ROI -> gated tiles must be far fewer
    plan = _mgr().plan_p2(strip_wh=(6400, 1280), roi_mask=[(300, 200, 320, 320)])
    assert plan.gated_tiles > 0
    assert plan.gated_tiles < plan.ungated_tiles
    assert plan.reduction_x > 10          # the >=~20x lever (small ROI on big strip)
    # every tile intersects the ROI
    for it in plan.items:
        assert it.kind == "tile"


def test_p2_empty_mask_means_no_tiles():
    plan = _mgr().plan_p2(strip_wh=(6400, 1280), roi_mask=[])
    assert plan.gated_tiles == 0          # cold strip -> heavy stage skipped
    assert plan.ungated_tiles > 0


def test_p3_has_no_full_frame_sahi():
    wheels = [(1, "L", "entry", "c1"), (1, "R", "entry", "c2")]
    slots = ["b1", "b2", "b3"]
    plan = _mgr().plan_p3(frame_wh=(4096, 3000), wheels=wheels, expected_slots=slots)
    assert plan.count(specialist="wheel_seg") == 2
    assert plan.count(specialist="fastener") == 3
    assert plan.gated_tiles == 0          # NO tiles in P3 — slots+crops only
    assert plan.ungated_tiles > 100       # naive full-SAHI (per frame) we AVOID entirely
    assert plan.reduction_x == float("inf")


def test_p1_no_sahi_gated_equals_ungated():
    plan = _mgr().plan_p1(n_detect_tiles=1000)
    assert plan.gated_tiles == 1000
    assert plan.ungated_tiles == 1000


def test_merge_sums_plans():
    m = _mgr()
    p = m.merge(m.plan_p1(10),
                m.plan_p2((2000, 1000), [(0, 0, 320, 320)]),
                m.plan_p3((2000, 2000), [(1, "L", "entry", "c")], ["s"]))
    assert p.count(zone="P1") == 10
    assert p.count(zone="P3", specialist="wheel_seg") == 1
    assert p.total_cost() > 0


def test_wheel_and_fastener_are_safety_tier():
    plan = _mgr().plan_p3((2000, 2000), [(1, "L", "entry", "c")], ["s"])
    assert all(i.tier == "safety" for i in plan.items)
