"""CPU checks of the scorer: python -m pytest tests (no simulator needed)."""

import math

from mount_aim.scoring import MountScore

HZ = 20
OPEN, CLOSED = [1.0, 1.0], [0.3, 1.0]
CLEAR, NEAR = [0.20, 0.20], [0.01, 0.20]


def run(states):
    s = MountScore(HZ)
    out = [s.update(*st) for st in states]
    return s, out


def test_untouched_start_scores_zero():
    s, out = run([(-31.0, False, OPEN, CLEAR)] * 200)
    assert max(out) == 0.0 and not s.success


def test_wrong_angle_locked_scores_zero():
    s, out = run([(-30.0, True, OPEN, CLEAR)] * 200)
    assert out[-1] == 0.0 and not s.success


def test_right_angle_still_held_is_partial():
    s, out = run([(15.0, False, CLOSED, NEAR)] * 200)
    assert out[-1] == 0.5 and not s.success
    s, out = run([(15.0, True, CLOSED, NEAR)] * 200)
    assert out[-1] == 0.75 and not s.success


def test_unlocked_release_falls_and_keeps_only_reach_credit():
    held = [(14.6, False, CLOSED, NEAR)] * 40
    fall = [(14.6 - 2.3 * k, False, OPEN, CLEAR) for k in range(20)] + [(-31.0, False, OPEN, CLEAR)] * 60
    s, out = run(held + fall)
    assert out[-1] == 0.5 and not s.success


def test_full_stable_release_succeeds_after_two_seconds():
    held = [(14.6, True, CLOSED, NEAR)] * 40
    released = [(14.7, True, OPEN, CLEAR)] * (2 * HZ + 1)
    s, out = run(held + released)
    assert out[-1] == 1.0 and s.success
    assert max(out[: 40 + 2 * HZ - 1]) < 1.0  # not before the full window


def test_drift_beyond_two_degrees_restarts_the_window():
    released = [(14.0 + 0.1 * k, True, OPEN, CLEAR) for k in range(30)]  # creeps 2.9 deg in 1.5 s
    s, out = run(released)
    assert not s.success


def test_outside_tolerance_is_not_full_credit():
    s, out = run([(19.0, True, OPEN, CLEAR)] * 200)
    assert out[-1] == 0.25 and not s.success


def test_nonfinite_state_is_invalid():
    s, out = run([(15.0, True, OPEN, CLEAR)] * 10 + [(math.nan, True, OPEN, CLEAR)] + [(15.0, True, OPEN, [math.inf, 0.2])])
    assert out[-2] == 0.0 and out[-1] == 0.0 and s.window_s == 0.0
