"""Exercise the whole controller with an inaccessible simulator and synthetic RGB."""
import sys
import types
import numpy as np
from hand_pump import policy
from inspect_robots_yam.packing import STATE_KEY, pack
from yam_arena import station
from yam_arena.constants import READY

class _Poison:
    def __getattr__(self, name):
        raise AssertionError(f"policy touched simulator: {name}")

def test_observation_policy_never_reads_the_simulator(monkeypatch):
    monkeypatch.setitem(sys.modules, 'hand_pump.embodiment', _Poison())
    pol = policy.PumpScriptedPolicy()
    pol._bases = dict(station.BASES)
    pol.reset(_Poison())
    state = pack(np.r_[READY, 1.0], np.r_[READY, 1.0])
    for i in range(950):
        obs = types.SimpleNamespace(state={STATE_KEY: state}, images={'top_cam': np.zeros((176, 320, 3), np.uint8)}, state_time=i * .05)
        pol.act(obs)
    assert not pol._stopped

def test_stop_requires_consecutive_readable_pumping_frames(monkeypatch):
    dr = policy.dr
    monkeypatch.setattr(dr, 'locate', lambda image: (1, 1, 1))
    monkeypatch.setattr(dr, 'read_v2', lambda image, dial: (None, None) if image is None else (0, 0))
    rule = dr.PressureStop()
    assert rule.update(1, True) == (False, 60.)
    assert rule.update(None, True) == (False, None)
    assert rule.update(1, True) == (False, 60.)
    assert rule.update(1, True) == (True, 60.)
    assert rule.update(1, False) == (False, 60.)
