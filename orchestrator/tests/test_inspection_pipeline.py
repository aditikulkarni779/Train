"""End-to-end: axle events + injected detections -> InspectionReports + latency."""
from inspection_pipeline import InspectionPipeline
from managers.spine_manager import FormationSpec, CoachSpec, AxleEvent


CFG = {
    "voting": {"k_of_n": 3, "cell_size_mm": 100.0, "fuse_entry_exit": True},
    "confidence": {"component": 0.35},
    "completeness": {"missing_min_views": 2},
    "spine": {"tau_ocr": 0.60, "require_ptp": True},
}

# 2-coach LHB rake, 4 axles each = 8 axle passes.
FORMATION = FormationSpec(coaches=(CoachSpec("LHB", 4), CoachSpec("LHB", 4)))
MANIFEST = {"LHB": ["control_arm", "footboard", "axle_box_cover"]}


def _axles():
    return [AxleEvent(seq=i, encoder_position_mm=float(i * 1000)) for i in range(8)]


def _present(cls, coach, pos, n=3, conf=0.9):
    # n confirming detections in one cell -> voted present
    return [{"coach_index": coach, "zone": "P1", "class": cls, "conf": conf,
             "longitudinal_position_mm": pos, "side": None, "view": None,
             "tier": "structural", "source_model": "component_detector"} for _ in range(n)]


def _pipeline():
    return InspectionPipeline(CFG, FORMATION, MANIFEST)


def test_end_to_end_produces_one_report_per_coach():
    res = _pipeline().run(
        _axles(),
        detections=_present("control_arm", 0, 10) + _present("footboard", 0, 10)
                   + _present("axle_box_cover", 0, 10)
                   + _present("control_arm", 1, 10) + _present("footboard", 1, 10)
                   + _present("axle_box_cover", 1, 10),
        views_per_coach={0: 3, 1: 3},
    )
    assert len(res.reports) == 2
    for r in res.reports:
        assert r["coach_type"] == "LHB"
        statuses = {s["component"]: s["status"] for s in r["manifest_status"]}
        assert statuses == {"control_arm": "present", "footboard": "present", "axle_box_cover": "present"}
        assert r["worst_tier"] == "none"


def test_missing_component_becomes_defect_and_needs_review():
    # coach 0 missing axle_box_cover
    res = _pipeline().run(
        _axles(),
        detections=_present("control_arm", 0, 10) + _present("footboard", 0, 10)
                   + _present("control_arm", 1, 10) + _present("footboard", 1, 10)
                   + _present("axle_box_cover", 1, 10),
        views_per_coach={0: 3, 1: 3},
    )
    c0 = res.reports[0]
    statuses = {s["component"]: s["status"] for s in c0["manifest_status"]}
    assert statuses["axle_box_cover"] == "missing"
    assert c0["review_status"] == "needs_review"
    assert any(d["class"] == "axle_box_cover_missing" for d in c0["defects"])


def test_latency_recorded_and_under_budget():
    res = _pipeline().run(
        _axles(),
        detections=_present("control_arm", 0, 10),
        views_per_coach={0: 3, 1: 3},
    )
    lat = res.latency
    assert lat.total_s > 0
    assert set(lat.stages_s) >= {"spine", "attach", "vote", "completeness", "report"}
    assert not lat.over_budget                      # pure orchestration is milliseconds
    assert lat.bottleneck in lat.stages_s


def test_too_few_views_reports_data_unavailable_not_missing():
    res = _pipeline().run(
        _axles(),
        detections=_present("control_arm", 0, 10),
        views_per_coach={0: 1, 1: 1},               # < missing_min_views
    )
    statuses = {s["component"]: s["status"] for s in res.reports[0]["manifest_status"]}
    assert set(statuses.values()) == {"data_unavailable"}


def test_no_ptp_degrades_coverage_end_to_end():
    res = _pipeline().run(
        _axles(),
        detections=_present("control_arm", 0, 10),
        views_per_coach={0: 3, 1: 3},
        ptp_ok=False,
    )
    inspected = [c for c in res.reports[0]["coverage"] if c["zone"] in ("P1", "P2", "P3")]
    assert all(c["state"] == "degraded" for c in inspected)
