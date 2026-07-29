import pytest

from managers.completeness_manager import (
    CompletenessManager, PRESENT, MISSING, DISPLACED, DATA_UNAVAILABLE,
)

CFG = {"completeness": {"missing_min_views": 2}}
MANIFEST = {
    "LHB": ["primary_coil_spring", "control_arm", "footboard", "axle_box_cover"],
    "ICF": ["axle_box_cover", "brake_block"],
}


def _status_map(rows):
    return {r.component: r.status for r in rows}


def test_present_when_seen_enough_views():
    mgr = CompletenessManager(CFG)
    rows = mgr.diff("LHB", MANIFEST,
                    present_view_counts={"primary_coil_spring": 3, "control_arm": 2,
                                         "footboard": 4, "axle_box_cover": 2},
                    total_views=4)
    assert all(s == PRESENT for s in _status_map(rows).values())


def test_missing_only_when_zero_hits_and_enough_views():
    mgr = CompletenessManager(CFG)
    rows = mgr.diff("LHB", MANIFEST,
                    present_view_counts={"primary_coil_spring": 3, "control_arm": 3, "footboard": 3},
                    total_views=4)                       # axle_box_cover unseen
    assert _status_map(rows)["axle_box_cover"] == MISSING


def test_too_few_views_is_data_unavailable_not_missing():
    mgr = CompletenessManager(CFG)
    rows = mgr.diff("LHB", MANIFEST,
                    present_view_counts={},               # nothing seen
                    total_views=1)                        # < missing_min_views
    assert all(s == DATA_UNAVAILABLE for s in _status_map(rows).values())


def test_coverage_gap_component_never_reported_clean():
    mgr = CompletenessManager(CFG)
    rows = mgr.diff("LHB", MANIFEST,
                    present_view_counts={"primary_coil_spring": 5, "control_arm": 5,
                                         "footboard": 5, "axle_box_cover": 5},
                    total_views=5,
                    coverage_gap={"axle_box_cover"})      # dropped region on this slot
    assert _status_map(rows)["axle_box_cover"] == DATA_UNAVAILABLE


def test_displaced_when_defect_flag_present():
    mgr = CompletenessManager(CFG)
    rows = mgr.diff("LHB", MANIFEST,
                    present_view_counts={"primary_coil_spring": 3, "control_arm": 3,
                                         "footboard": 3, "axle_box_cover": 3},
                    total_views=3,
                    displaced={"control_arm"})
    assert _status_map(rows)["control_arm"] == DISPLACED


def test_unknown_coach_type_refuses_to_guess():
    mgr = CompletenessManager(CFG)
    with pytest.raises(KeyError):
        mgr.diff("VB-motor", MANIFEST, present_view_counts={}, total_views=5)


def test_icf_uses_its_own_manifest():
    mgr = CompletenessManager(CFG)
    rows = mgr.diff("ICF", MANIFEST,
                    present_view_counts={"axle_box_cover": 3},   # brake_block unseen
                    total_views=3)
    m = _status_map(rows)
    assert set(m) == {"axle_box_cover", "brake_block"}           # ICF set only
    assert m["brake_block"] == MISSING
