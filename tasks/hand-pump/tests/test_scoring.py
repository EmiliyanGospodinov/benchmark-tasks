"""CPU checks of the pump scorer: python -m pytest tests (no simulator)."""
import math

from hand_pump.scoring import PumpScore

HZ = 20
OPEN, HELD = [1.0, 1.0], [0.4, 0.45]
CLEAR, NEAR = [0.2, 0.2], [0.01, 0.01]


def run(states):
    s = PumpScore(HZ)
    return s, [s.update(*st) for st in states]


def test_untouched_start_scores_zero():
    s, out = run([(0.0, OPEN, CLEAR, True)] * 100)
    assert max(out) == 0.0 and not s.success


def test_partial_pressure():
    s, out = run([(30.0, HELD, NEAR, False)] * 10)
    assert out[-1] == 0.25


def test_in_band_still_held_is_partial():
    s, out = run([(60.0, HELD, NEAR, False)] * 100)
    assert out[-1] == 0.75 and not s.success


def test_released_in_band_succeeds_after_hold():
    s, out = run([(60.0, HELD, NEAR, False)] * 20 + [(60.0, OPEN, CLEAR, True)] * (2 * HZ))
    assert out[-1] == 1.0 and s.success and max(out[:20 + 2 * HZ - 1]) < 1.0


def test_released_while_lifted_is_not_success():
    s, out = run([(60.0, OPEN, CLEAR, False)] * 100)
    assert not s.success and out[-1] == 0.75


def test_below_band_after_release():
    s, out = run([(60.0, HELD, NEAR, False)] * 10 + [(45.0, OPEN, CLEAR, True)] * 100)
    assert out[-1] == 0.5 and not s.success


def test_overpressure_caps_score():
    s, out = run([(90.0, HELD, NEAR, False)] * 5 + [(60.0, OPEN, CLEAR, True)] * 100)
    assert out[-1] == 0.25 and not s.success


def test_nonfinite_is_invalid():
    s, out = run([(60.0, OPEN, CLEAR, True)] * 10 + [(math.nan, OPEN, CLEAR, True)])
    assert out[-1] == 0.0 and s.window_s == 0.0
