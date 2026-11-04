"""CPU checks of the tipper scorer (no simulator): python -m pytest tests"""
import math

import numpy as np

from duplo_tipper import design as d
from duplo_tipper.scoring import HOLD_S, Observed, Scorer

DT = 0.05
BUILT = d.assembled_poses()
JOINED = {"brick": True, "cab": True, "hinge": True, "bed": True}
FAR = [np.array([0.0, 0.5, 1.2])] * 4  # both TCPs and pad tips far from every part


def still():
    return {n: (0.0, 0.0) for n in ("chassis", "brick", "cab", "hinge", "bed", "payload")}


def tipped_bed(tip):
    """Bed pose opened by ``tip`` on the hinge (rotation about the plate's -x)."""
    return d.world_pose(BUILT["hinge"], d.BED_IN_HINGE, d.rot_x(-tip))


def obs(payload, joined=JOINED, speeds=None, hands=FAR, open_=True, bed_tip=0.0):
    poses = {k: v.copy() for k, v in BUILT.items()}
    if bed_tip:
        poses["bed"] = tipped_bed(bed_tip)
    poses["payload"] = np.asarray(payload, float)
    return Observed(poses=poses, speeds=speeds or still(), joined=dict(joined), hands=hands, grippers_open=open_, dt=DT)


def in_bed():
    bed = d.world_pose(BUILT["hinge"], d.BED_IN_HINGE)
    return d.world_pose(bed, (0.0, -0.030, 0.0076 + d.PAYLOAD_SIZE / 2))


def in_tray():
    return np.r_[d.TRAY_CENTER[0], d.TRAY_CENTER[1], d.TOP + d.TRAY_FLOOR + d.PAYLOAD_SIZE / 2, 0, 0, 0, 1.0]


def run(scorer, states):
    return [scorer.update(o) for o in states]


def test_loose_parts_score_zero():
    s = Scorer()
    score, info = s.update(obs(in_tray(), joined={k: False for k in JOINED}))
    assert score == 0.0 and not info["success"]


def test_connections_give_partial_credit():
    s = Scorer()
    assert s.update(obs(in_tray(), joined={"brick": True, "cab": True, "hinge": False, "bed": False}))[0] == 0.25
    assert s.update(obs(in_tray(), joined={"brick": True, "cab": True, "hinge": True, "bed": False}))[0] == 0.5
    assert s.update(obs(in_tray()))[0] == 0.75  # built, but never loaded and unloaded


def test_payload_in_tray_without_unloading_is_not_success():
    s = Scorer()
    out = run(s, [obs(in_tray())] * 60)
    assert all(sc == 0.75 for sc, _ in out) and not out[-1][1]["success"]


def _loaded_then_dumped(s):
    run(s, [obs(in_bed())] * 30)  # resting in the closed bed: loaded
    assert s.loaded
    exiting = d.world_pose(tipped_bed(math.radians(60)), (0.0, 0.012, 0.018))  # past the open rear
    run(s, [obs(exiting, bed_tip=math.radians(60))] * 2)
    assert s.unloaded


def test_full_sequence_succeeds_after_the_hold():
    s = Scorer()
    _loaded_then_dumped(s)
    out = run(s, [obs(in_tray())] * (round(HOLD_S / DT) + 1))
    assert out[-1][0] == 1.0 and out[-1][1]["success"]
    assert not out[round(HOLD_S / DT) - 2][1]["success"]


def test_moving_part_blocks_success():
    s = Scorer()
    _loaded_then_dumped(s)
    moving = still()
    moving["bed"] = (0.0, 0.5)  # still turning on its hinge
    out = run(s, [obs(in_tray(), speeds=moving)] * 60)
    assert not any(i["success"] for _, i in out)
    moving = still()
    moving["chassis"] = (0.02, 0.0)  # truck still rolling
    out = run(s, [obs(in_tray(), speeds=moving)] * 60)
    assert not any(i["success"] for _, i in out)


def test_hands_near_parts_block_success():
    s = Scorer()
    _loaded_then_dumped(s)
    near = [BUILT["cab"][:3] + np.array([0.0, 0.0, 0.03])] * 4
    out = run(s, [obs(in_tray(), hands=near)] * 60)
    assert not any(i["success"] for _, i in out)


def test_lost_bed_connection_invalidates_unloading():
    s = Scorer()
    run(s, [obs(in_bed())] * 30)
    lost = dict(JOINED, bed=False)
    exiting = d.world_pose(tipped_bed(math.radians(60)), (0.0, 0.012, 0.018))
    run(s, [obs(exiting, joined=lost, bed_tip=math.radians(60))] * 2)
    assert not s.unloaded
    out = run(s, [obs(in_tray(), joined=lost)] * 60)
    assert not any(i["success"] for _, i in out)


def test_reset_clears_progress():
    s = Scorer()
    _loaded_then_dumped(s)
    s.reset()
    assert not s.loaded and not s.unloaded and s.hold_s == 0.0
    out = run(s, [obs(in_tray())] * 60)
    assert not any(i["success"] for _, i in out)


def test_nonfinite_pose_does_not_succeed():
    s = Scorer()
    _loaded_then_dumped(s)
    bad = in_tray()
    bad[0] = math.nan
    out = run(s, [obs(bad)] * 60)
    assert not any(i["success"] for _, i in out)


def test_bed_task_gives_no_credit_for_the_preassembled_truck():
    s = Scorer(preassembled=("brick", "cab", "hinge"))
    assert s.update(obs(in_tray(), joined=dict(JOINED, bed=False)))[0] == 0.0  # untouched start
    assert s.update(obs(in_tray()))[0] == 0.5  # bed attached
    s = Scorer(preassembled=("brick", "cab", "hinge"))
    run(s, [obs(in_bed())] * 30)
    assert s.update(obs(in_bed()))[0] == 0.75  # loaded
    s.reset()
    assert s.preassembled == ("brick", "cab", "hinge") and not s.loaded
