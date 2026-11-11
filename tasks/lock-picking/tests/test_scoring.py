"""CPU tests of the lock-picking scorer and design invariants (no simulator)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lock_picking import design as d  # noqa: E402
from lock_picking.scoring import LockScore  # noqa: E402
from lock_picking.scoring import pin_set  # noqa: E402

HZ = 20
REST_D, REST_K = d.DRIVER_BOTTOM0, d.KEY_TOP0
SET_D = d.R - 0.0002
FAR = [0.2, 0.2]
OPEN = [1.0, 1.0]
OUT = (d.FACE_X - 0.05, 0.0, d.AXIS_Z)
IN = (d.FACE_X + 0.03, 0.0, d.AXIS_Z - 0.012)
TABLE = (0.27, 0.12, d.TOP + 0.002)


def run(s, sec, theta=0.0, flex=0.0, n_set=0, lifted=0, pick_x=IN, grip=OPEN, clear=FAR, moved=0.0):
    drv = [SET_D] * n_set + [REST_D] * (3 - n_set)
    key = [REST_K] * 3
    for i in range(n_set, n_set + lifted):  # held up by the pick: driver at the shear line, key pin under it
        drv[i], key[i] = SET_D, d.R - 0.0001
    v = 0.0
    for _ in range(round(sec * HZ)):
        v = s.update(theta, flex, drv, key, pick_x, grip, clear, moved)
    return v


def test_untouched_is_zero():
    assert run(LockScore(HZ), 5.0) == 0.0


def test_partial_per_pin_latched():
    s = LockScore(HZ)
    assert run(s, 1.0, theta=3, flex=6, n_set=1) == 0.25
    assert run(s, 1.0, theta=6, flex=6, n_set=2) == 0.5
    assert run(s, 1.0, theta=0, flex=0, n_set=0) == 0.5  # released: pins drop, credit stays
    assert run(s, 1.0, theta=12, flex=6, n_set=3) == 0.75


def test_pin_held_by_pick_is_not_set():
    s = LockScore(HZ)
    assert run(s, 2.0, theta=1.4, flex=6, n_set=0, lifted=3) == 0.0


def test_brief_set_needs_half_second():
    s = LockScore(HZ)
    assert run(s, 0.3, theta=3, flex=6, n_set=1) == 0.0


def test_full_success_needs_release_and_tools_out():
    s = LockScore(HZ)
    run(s, 1.0, theta=12, flex=6, n_set=3)
    run(s, 0.05, theta=16, flex=6, n_set=3)
    assert run(s, 3.0, theta=90, flex=6, pick_x=OUT, clear=FAR) == 0.75  # wrench still loaded
    assert run(s, 3.0, theta=90, flex=0.5, pick_x=IN) == 0.75  # pick still in the keyway
    assert run(s, 3.0, theta=90, flex=0.5, pick_x=OUT, clear=[0.2, 0.05]) == 0.75  # hand near the lock
    assert run(s, 3.0, theta=90, flex=0.5, pick_x=OUT, grip=[1.0, 0.5]) == 0.75  # gripper closed
    assert run(s, 1.0, theta=90, flex=0.5, pick_x=OUT) == 0.75  # not yet 2 s
    assert run(s, 1.05, theta=90, flex=0.5, pick_x=OUT) == 1.0
    assert s.success


def test_forcing_is_zero_latched():
    s = LockScore(HZ)
    run(s, 1.0, theta=3, flex=6, n_set=1)
    assert run(s, 0.05, theta=3, flex=15.5, n_set=1) == 0.0
    assert run(s, 5.0, theta=90, flex=0.0, pick_x=OUT) == 0.0


def test_moved_body_and_nonfinite_are_zero():
    s = LockScore(HZ)
    assert run(s, 1.0, theta=3, flex=6, n_set=1, moved=0.003) == 0.0
    s = LockScore(HZ)
    assert run(s, 1.0, theta=float("nan")) == 0.0


def test_design_binding_order_and_ledge():
    order = d.binding_order()
    assert order == [1, 2, 0]
    gaps = sorted(d.GAP)
    assert min(b - a for a, b in zip(gaps, gaps[1:])) >= 0.0015  # >= 1.5 mm ledge for each set driver
    assert d.bind_angle_deg(order[-1]) < 15  # all binding happens in the first ~10 degrees


def test_design_pick_clears_other_pins():
    shaft_top_rest = -0.014 + d.PICK_SHAFT[2] / 2
    hook_top_rest = shaft_top_rest + d.PICK_HOOK[2]
    assert hook_top_rest < d.KEY_BOTTOM0 - 0.0015  # the hook passes under the key pins
    lift_to_set = (d.SET_DRIVER_MIN + 0.0005 - d.DRIVER_BOTTOM0) + (d.KEY_BOTTOM0 - hook_top_rest)
    assert shaft_top_rest + lift_to_set < d.KEY_BOTTOM0  # the shaft never lifts a pin in front
    assert -0.014 - d.PICK_SHAFT[2] / 2 >= d.KEYWAY_FLOOR - 1e-9


def test_design_tool_clearance():
    # the wrench sleeve stays at least 12 cm (y) from the pick post over the whole turn
    post_y = d.PICK_LEG_Y
    for deg in range(0, 91, 5):
        assert post_y - d.sleeve_center(math.radians(deg))[1] >= 0.12


def test_pick_on_table_counts_as_out():
    s = LockScore(HZ)
    run(s, 1.0, theta=12, flex=6, n_set=3)
    run(s, 0.05, theta=16, flex=6, n_set=3)
    assert run(s, 2.05, theta=90, flex=0.5, pick_x=TABLE) == 1.0


def test_binding_play_without_setting_pins_is_not_a_failure():
    s = LockScore(HZ)
    assert run(s, 2.0, theta=10.6) == 0.0
    assert not s.failed


def test_crossing_with_unset_pin_is_zero_even_after_partial_credit():
    s = LockScore(HZ)
    assert run(s, 1.0, theta=10, n_set=2) == 0.5
    assert run(s, 0.05, theta=15.01, n_set=2) == 0.0
    assert s.failed
    assert run(s, 3.0, theta=90, n_set=3, pick_x=OUT) == 0.0
    assert not s.success


def test_old_set_count_does_not_authorize_later_bypass():
    s = LockScore(HZ)
    run(s, 1.0, theta=12, n_set=3)
    run(s, 1.0, theta=0)
    assert run(s, 3.0, theta=90, pick_x=OUT) == 0.0
    assert s.failed


def test_return_to_binding_requires_fresh_clearance():
    s = LockScore(HZ)
    run(s, 0.05, theta=16, n_set=3)
    run(s, 0.05, theta=10)
    assert run(s, 0.05, theta=16, n_set=2) == 0.0
    assert s.failed


def test_reset_clears_binding_permission():
    s = LockScore(HZ)
    run(s, 0.05, theta=16, n_set=3)
    s.reset()
    assert run(s, 3.0, theta=90, pick_x=OUT) == 0.0
    assert s.failed


def test_driver_far_above_shear_line_is_not_set():
    assert pin_set(SET_D, REST_K)
    assert not pin_set(d.R + 0.002, REST_K)


def test_incomplete_pin_state_is_invalid():
    s = LockScore(HZ)
    assert s.update(12, 0, [SET_D] * 2, [REST_K] * 2, OUT, OPEN, FAR, 0) == 0.0
    assert s.failed


def test_failure_clears_success_flag():
    s = LockScore(HZ)
    run(s, 0.05, theta=16, n_set=3)
    assert run(s, 2.05, theta=90, pick_x=OUT) == 1.0
    assert run(s, 0.05, theta=90, flex=16, pick_x=OUT) == 0.0
    assert not s.success


def test_return_spring_target_within_180():
    assert -180 < d.RETURN_TARGET_DEG < 0
    assert abs(d.RETURN_PRELOAD + d.RETURN_STIFF_DEG * 90 - d.TORQUE_CAP) > 0.05  # turning to open stays well under the cap
