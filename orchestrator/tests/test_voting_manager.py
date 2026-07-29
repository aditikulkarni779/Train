from managers.voting_manager import VotingManager


CFG = {"voting": {"k_of_n": 3, "cell_size_mm": 100.0, "fuse_entry_exit": True},
       "confidence": {"component": 0.35}}


def _det(cls, conf, pos, coach=0, side=None, view=None, zone="P1", tier="structural"):
    return {"coach_index": coach, "zone": zone, "class": cls, "conf": conf,
            "longitudinal_position_mm": pos, "side": side, "view": view, "tier": tier,
            "source_model": "p1_side_v1"}


def test_below_threshold_detections_are_not_votes():
    mgr = VotingManager(CFG)
    dets = [_det("footboard", 0.2, 10) for _ in range(5)]   # all below 0.35
    assert mgr.vote(dets) == []


def test_needs_k_of_n_confirmations_in_same_cell():
    mgr = VotingManager(CFG)
    # 2 confirmations in one cell -> below k=3 -> no flag
    assert mgr.vote([_det("footboard", 0.9, 10), _det("footboard", 0.8, 20)]) == []
    # 3 confirmations in the same 100mm band -> flag
    flags = mgr.vote([_det("footboard", 0.9, 10), _det("footboard", 0.8, 20), _det("footboard", 0.7, 30)])
    assert len(flags) == 1
    assert flags[0].votes == 3
    assert flags[0].cls == "footboard"


def test_detections_in_different_bands_do_not_share_a_cell():
    mgr = VotingManager(CFG)
    # band 0 (0-99) gets 2, band 1 (100-199) gets 1 -> neither reaches k=3
    dets = [_det("footboard", 0.9, 10), _det("footboard", 0.9, 50), _det("footboard", 0.9, 150)]
    assert mgr.vote(dets) == []


def test_mean_confidence_and_position_reported():
    mgr = VotingManager(CFG)
    flags = mgr.vote([_det("cbc", 0.6, 10), _det("cbc", 0.8, 20), _det("cbc", 1.0, 30)])
    assert len(flags) == 1
    assert abs(flags[0].conf - 0.8) < 1e-6
    assert abs(flags[0].longitudinal_position_mm - 20.0) < 1e-6


def test_wheel_entry_exit_fuse_both_views_take_max_conf():
    mgr = VotingManager(CFG)
    entry = [_det("wheel_shelling", 0.6, 10, side="L", view="entry", zone="P3", tier="safety") for _ in range(3)]
    exit_ = [_det("wheel_shelling", 0.9, 12, side="L", view="exit", zone="P3", tier="safety") for _ in range(3)]
    flags = mgr.vote(entry + exit_)
    assert len(flags) == 1                       # fused into one physical wheel
    assert tuple(flags[0].fused_views) == ("entry", "exit")
    assert abs(flags[0].conf - 0.9) < 1e-6       # both agree -> max (precision)


def test_wheel_single_view_still_flags_for_recall():
    mgr = VotingManager(CFG)
    entry = [_det("wheel_shelling", 0.7, 10, side="R", view="entry", zone="P3", tier="safety") for _ in range(3)]
    flags = mgr.vote(entry)
    assert len(flags) == 1                       # either view alone -> recall
    assert tuple(flags[0].fused_views) == ("entry",)


def test_left_and_right_wheels_are_separate():
    mgr = VotingManager(CFG)
    left = [_det("wheel_shelling", 0.8, 10, side="L", view="entry", zone="P3", tier="safety") for _ in range(3)]
    right = [_det("wheel_shelling", 0.8, 10, side="R", view="entry", zone="P3", tier="safety") for _ in range(3)]
    flags = mgr.vote(left + right)
    assert len(flags) == 2
    assert {f.side for f in flags} == {"L", "R"}


def test_dedupe_is_idempotent_on_reingest():
    mgr = VotingManager(CFG)
    dets = [_det("cbc", 0.9, 10), _det("cbc", 0.9, 20), _det("cbc", 0.9, 30)]
    once = mgr.vote(dets)
    twice = mgr.vote(dets + dets)                 # re-ingest same train
    assert len(VotingManager.dedupe(once + twice)) == 1
