"""Stage 4 — spine invariant tests (risks A1/A3/A4)."""
import pytest
from managers.spine_manager import (
    SpineManager, FormationSpec, CoachSpec, AxleEvent, SpineError,
)


def formation(n=3):
    return FormationSpec(coaches=tuple(CoachSpec("VB-trailer", 4) for _ in range(n)))


def axle_events(n_axles, step=2000.0):
    return [AxleEvent(seq=i, encoder_position_mm=i * step) for i in range(n_axles)]


def test_build_assigns_coach_index_from_axle_order():
    sm = SpineManager(formation(3))
    coaches = sm.build(axle_events(12))
    assert [c.coach_index for c in coaches] == [0, 1, 2]
    # coach 1 owns axles 4..7
    assert coaches[1].axle_ids == [4, 5, 6, 7]
    # per-detection stamp carries encoder position, not clock
    stamp = coaches[1].stamp_for_axle(5)
    assert stamp["coach_index"] == 1
    assert stamp["longitudinal_position_mm"] == 5 * 2000.0


def test_A1_axle_count_mismatch_hard_fails():
    sm = SpineManager(formation(3))          # expects 12 axles
    with pytest.raises(SpineError):
        sm.build(axle_events(11))            # one missed axle -> never renumber


def test_A4_non_monotonic_positions_rejected():
    sm = SpineManager(formation(1))
    evs = axle_events(4)
    evs[2].encoder_position_mm = 100.0       # goes backwards (jitter/wall-clock)
    with pytest.raises(SpineError):
        sm.build(evs)


def test_A3_ocr_cannot_change_coach_index():
    sm = SpineManager(formation(2), tau_ocr=0.60)
    sm.build(axle_events(8))
    sm.set_ocr_label(0, "B7-21456", 0.94)    # high conf -> label attaches
    sm.set_ocr_label(1, "garbage", 0.20)     # low conf -> formation fallback
    c = sm.coaches()
    assert c[0].coach_index == 0 and c[0].coach_number == "B7-21456"
    assert c[0].spine_source == "axle_count"
    assert c[1].coach_index == 1                      # index UNCHANGED by OCR
    assert c[1].coach_number is None                  # low-conf label dropped
    assert c[1].spine_source == "formation_fallback"


def test_A2_ptp_absent_flags_per_zone_only():
    sm = SpineManager(formation(1), require_ptp=True)
    coaches = sm.build(axle_events(4), ptp_ok=False)
    assert all(not c.ptp_ok for c in coaches)         # caller degrades, no crash
