"""CPU checks of the blade-change scorer: python -m pytest tests (no simulator needed)."""

import math

from blade_change import design as d
from blade_change.scoring import BladeScore

HZ = 20
OPEN, HELD = [1.0, 1.0], [0.2, 0.2]
AWAY, NEAR = [0.2, 0.2], [0.003, 0.002]
TUG_OPEN, TUG_CLEAR = [1.0, 0.2], [0.08, 0.002]
SEATED = {"rel_mm": [0.2, -0.9, -0.3], "roll_deg": 0.5, "yaw_deg": 0.3, "pitch_deg": -0.4, "world": [0.30, -0.05, 0.07]}
RACK = {"rel_mm": [100.0, -10.0, -30.0], "roll_deg": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0, "world": [0.40, -0.05, 0.04]}
TRAY = {"rel_mm": [-75.0, -100.0, -50.0], "roll_deg": 1.0, "yaw_deg": 2.0, "pitch_deg": 0.0,
        "world": [d.TRAY_XY[0], d.TRAY_XY[1], 0.012]}


def state(old=SEATED, new=RACK, collar=0.0, pin=0.0, ext=0.0, speed=0.0,
          new_grip_dist=(0.2, 0.002), mechanism_clear=(0.08, 0.15)):
    return {"carriage_ext_mm": ext, "collar_deg": collar, "pin_lift_mm": pin, "old_blade": old, "new_blade": new,
            "speeds": dict.fromkeys(("holder", "collar", "pin", "old_blade", "new_blade"), speed),
            "new_grip_dist_m": new_grip_dist, "mechanism_clear_m": mechanism_clear}


def run(seq):
    s = BladeScore(HZ)
    return s, [s.update(*x) for x in seq]


def done(new=SEATED, **kw):
    return state(old=TRAY, new=new, **kw)


def test_installed_start_scores_zero():
    s, out = run([(state(), OPEN, AWAY)] * 100)
    assert max(out) == 0.0


def test_old_blade_stowed_is_quarter():
    s, out = run([(state(old=TRAY), OPEN, AWAY)] * 20)
    assert out[-1] == 0.25


def test_seated_with_collar_still_turned_is_half():
    s, out = run([(done(collar=-33.0, pin=14.0), HELD, NEAR)] * 20)
    assert out[-1] == 0.5


def test_upside_down_replacement_is_not_seated():
    flipped = dict(SEATED, roll_deg=180.0)
    s, out = run([(state(old=TRAY, new=flipped), OPEN, AWAY)] * 20)
    assert out[-1] == 0.25


def test_locked_without_tug_is_three_quarters():
    s, out = run([(done(), OPEN, AWAY)] * 200)
    assert out[-1] == 0.75 and not s.success


def test_wrist_command_without_carriage_load_is_not_a_tug():
    s, out = run([(done(ext=1.0), TUG_OPEN, TUG_CLEAR)] * 40 + [(done(), OPEN, AWAY)] * 100)
    assert not s.tug_passed and not s.success


def test_full_cycle_succeeds_after_tug_and_release_hold():
    pulled = dict(SEATED, rel_mm=[0.2, -1.9, -0.6])
    seq = [(done(ext=8.5, new=pulled), TUG_OPEN, TUG_CLEAR)] * 12 + [(done(), OPEN, AWAY)] * (2 * HZ + 1)
    s, out = run(seq)
    assert s.tug_passed and s.success and out[-1] == 1.0 and max(out[:12 + 2 * HZ - 1]) < 1.0


def test_tug_that_pulls_the_blade_out_fails():
    out_blade = dict(SEATED, rel_mm=[0.2, -12.0, -0.6])
    s, out = run([(done(ext=8.5, new=out_blade), TUG_OPEN, TUG_CLEAR)] * 20 + [(done(), OPEN, AWAY)] * 60)
    assert not s.tug_passed and not s.success


def test_still_held_after_tug_is_not_full_credit():
    s, out = run([(done(ext=8.5), TUG_OPEN, TUG_CLEAR)] * 12 + [(done(), HELD, NEAR)] * 100)
    assert s.tug_passed and not s.success and out[-1] == 0.75


def test_nonfinite_state_is_invalid():
    bad = dict(SEATED, rel_mm=[math.nan, 0.0, 0.0])
    s, out = run([(done(), OPEN, AWAY)] * 5 + [(state(old=TRAY, new=bad), OPEN, AWAY)])
    assert out[-1] == 0.0 and s.window_s == 0.0


def test_unlocking_and_reinstalling_requires_a_new_tug():
    removed = dict(SEATED, rel_mm=[0.0, -60.0, 0.0])
    seq = ([(done(ext=8.5), TUG_OPEN, TUG_CLEAR)] * 12
           + [(done(new=removed, collar=-33.0, pin=14.0), HELD, NEAR)]
           + [(done(), OPEN, AWAY)] * 60)
    s, out = run(seq)
    assert not s.tug_passed and not s.success and out[-1] == 0.75
    for _ in range(12):
        s.update(done(ext=8.5), TUG_OPEN, TUG_CLEAR)
    for _ in range(40):
        score = s.update(done(), OPEN, AWAY)
    assert score == 1.0


def test_unlocking_without_removal_invalidates_tug():
    s, out = run([(done(ext=8.5), TUG_OPEN, TUG_CLEAR)] * 12
                 + [(done(collar=-33.0, pin=14.0), HELD, NEAR)]
                 + [(done(), OPEN, AWAY)] * 60)
    assert not s.tug_passed and out[-1] == 0.75


def test_direct_holder_load_without_blue_grip_is_not_a_tug():
    # The right hand grips the holder; the left is open and away.
    direct = done(ext=8.5, new_grip_dist=(0.2, 0.16), mechanism_clear=(0.08, 0.002))
    s, out = run([(direct, TUG_OPEN, TUG_CLEAR)] * 30 + [(done(), OPEN, AWAY)] * 60)
    assert not s.tug_passed and out[-1] == 0.75


def test_open_grippers_cannot_register_a_tug():
    s, out = run([(done(ext=8.5), OPEN, AWAY)] * 30 + [(done(), OPEN, AWAY)] * 60)
    assert not s.tug_passed and out[-1] == 0.75


def test_other_hand_loading_holder_while_blue_is_held_is_not_a_tug():
    assisted = done(ext=8.5, mechanism_clear=(0.002, 0.15))
    s, out = run([(assisted, TUG_OPEN, TUG_CLEAR)] * 30 + [(done(), OPEN, AWAY)] * 60)
    assert not s.tug_passed and out[-1] == 0.75


def test_either_arm_can_perform_the_retention_check():
    left = done(ext=8.5, new_grip_dist=(0.002, 0.2), mechanism_clear=(0.15, 0.08))
    s, out = run([(left, TUG_OPEN[::-1], TUG_CLEAR[::-1])] * 12 + [(done(), OPEN, AWAY)] * 40)
    assert s.success and out[-1] == 1.0


def test_invalid_measurement_invalidates_prior_tug():
    bad = done(new_grip_dist=(math.nan, 0.002))
    s, out = run([(done(ext=8.5), TUG_OPEN, TUG_CLEAR)] * 12
                 + [(bad, OPEN, AWAY)] + [(done(), OPEN, AWAY)] * 60)
    assert not s.tug_passed and out[-1] == 0.75
