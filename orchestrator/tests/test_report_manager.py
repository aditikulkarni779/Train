import pytest

from managers.report_manager import ReportManager, SPECIALIST_ZONES_NOT_SHIPPED
from managers.voting_manager import VotedFlag
from managers.completeness_manager import ComponentStatus


class _Stamp:
    def __init__(self, ci=3, ct="LHB", num="B7-21456", ptp=True):
        self.coach_index, self.coach_type, self.coach_number, self.ptp_ok = ci, ct, num, ptp


def _mgr():
    return ReportManager({})


def test_report_has_required_shape_and_validates():
    r = _mgr().build(_Stamp(), voted_flags=[], manifest_status=[], latency_s=12.5)
    ReportManager.validate(r)                              # must not raise
    assert r["coach_index"] == 3 and r["coach_type"] == "LHB"
    assert r["latency_s"] == 12.5


def test_specialist_zones_flagged_not_shipped():
    r = _mgr().build(_Stamp(), [], [], latency_s=5)
    states = {c["zone"]: c["state"] for c in r["coverage"]}
    for z in SPECIALIST_ZONES_NOT_SHIPPED:
        assert states[z] == "data_unavailable"            # E6/C3: never clean


def test_no_ptp_degrades_inspected_zones():
    r = _mgr().build(_Stamp(ptp=False), [], [], latency_s=5)
    inspected = [c for c in r["coverage"] if c["zone"] in ("P1", "P2", "P3")]
    assert all(c["state"] == "degraded" for c in inspected)   # A2


def test_worst_tier_and_review_status_from_safety_defect():
    flag = VotedFlag(coach_index=3, zone="P3", cls="wheel_shelling", side="L",
                     position_band=0, longitudinal_position_mm=10, conf=0.8, votes=3, tier="safety")
    r = _mgr().build(_Stamp(), [flag], [], latency_s=5)
    assert r["worst_tier"] == "safety"
    assert r["review_status"] == "needs_review"


def test_missing_component_triggers_review():
    st = ComponentStatus("axle_box_cover", "missing", 0)
    r = _mgr().build(_Stamp(), [], [st], latency_s=5)
    assert r["review_status"] == "needs_review"


def test_clean_train_auto_passes():
    st = ComponentStatus("footboard", "present", 3)
    r = _mgr().build(_Stamp(), [], [st], latency_s=5)
    assert r["worst_tier"] == "none"
    assert r["review_status"] == "auto_pass"


def test_dropped_region_is_data_unavailable():
    r = _mgr().build(_Stamp(), [], [], latency_s=5, dropped_regions=["cam3_frame_88"])
    assert any(c["state"] == "data_unavailable" and "dropped" in c["region"] for c in r["coverage"])


def test_validate_rejects_empty_coverage():
    r = _mgr().build(_Stamp(), [], [], latency_s=5)
    r["coverage"] = []
    with pytest.raises(ValueError):
        ReportManager.validate(r)
