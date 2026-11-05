"""CPU checks of the ratchet scorer: python -m pytest tests (no simulator)."""
import math

from ratchet_lowering.scoring import RatchetScore

HZ = 20
OPEN, HELD = [1.0, 1.0], [0.22, 0.15]
CLEAR, NEAR = [0.07, 0.07], [0.002, 0.002]


def run(states):
    s = RatchetScore(HZ)
    return s, [s.update(th, w, 3.0, g, c, 0.02) for th, w, g, c in states]


def lower(a, b, n, grip=HELD, clear=NEAR):
    return [(a + (b - a) * k / n, -8.0, grip, clear) for k in range(1, n + 1)]


def still(th, n, grip=OPEN, clear=CLEAR):
    return [(th, 0.0, grip, clear)] * n


FULL = lower(35.3, 5.3, 60) + still(5.3, 50) + lower(5.3, -28.0, 80) + still(-28.0, 50)


def test_untouched_start_scores_zero():
    s, out = run(still(35.3, 200))
    assert max(out) == 0.0


def test_full_sequence_succeeds():
    s, out = run(FULL)
    assert s.success and out[-1] == 1.0


def test_hold_needs_two_seconds_released():
    s, out = run(lower(35.3, 5.3, 60) + still(5.3, 30))
    assert out[-1] == 0.25 and not s.held
    s, out = run(lower(35.3, 5.3, 60) + still(5.3, 60, grip=HELD, clear=NEAR))  # still holding the crank
    assert out[-1] == 0.25


def test_hold_then_restart_partial():
    s, out = run(lower(35.3, 5.3, 60) + still(5.3, 50) + lower(5.3, -15.0, 30))
    assert out[-1] == 0.75


def test_landing_without_intermediate_hold_is_capped():
    s, out = run(lower(35.3, -28.0, 120) + still(-28.0, 100))
    assert out[-1] == 0.25 and not s.success


def test_runaway_latches_zero():
    states = lower(35.3, 20.0, 20) + [(10.0, -120.0, OPEN, CLEAR)] + FULL
    s, out = run(states)
    assert out[-1] == 0.0 and not s.success


def test_landed_but_still_held_is_not_success():
    s, out = run(lower(35.3, 5.3, 60) + still(5.3, 50) + lower(5.3, -28.0, 80) + still(-28.0, 100, HELD, NEAR))
    assert out[-1] == 0.75 and not s.success


def test_wrong_tooth_is_not_the_intermediate_hold():
    s, out = run(lower(35.3, 20.3, 30) + still(20.3, 100))
    assert out[-1] == 0.25 and not s.held


def test_nonfinite_is_invalid():
    s, out = run(lower(35.3, 5.3, 60) + [(math.nan, 0.0, OPEN, CLEAR)] + still(5.3, 50))
    assert out[-1] == 0.0


def test_supporting_another_part_is_not_a_released_hold():
    s = RatchetScore(HZ)
    for _ in range(80):
        score = s.update(5.3, 0.0, 3.2, OPEN, CLEAR, 0.0)
    assert score == 0.25 and not s.held


def test_open_pawl_is_not_a_pawl_supported_hold():
    s = RatchetScore(HZ)
    for _ in range(80):
        score = s.update(5.3, 0.0, 12.0, OPEN, CLEAR, 0.02)
    assert score == 0.25 and not s.held


def test_nonfinite_clearance_fails_closed():
    s = RatchetScore(HZ)
    assert s.update(5.3, 0.0, 3.2, OPEN, CLEAR, math.nan) == 0.0
    assert s.failed
