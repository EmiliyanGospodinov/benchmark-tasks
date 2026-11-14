"""Run the whole controller with the simulator, embodiment and scorer inaccessible, on observations only."""
import sys
import types

import numpy as np
from inspect_robots_yam.packing import STATE_KEY, pack, split
from yam_arena import station
from yam_arena.constants import READY

from mechanical_puller import policy, shaft_view


class _Poison:
    def __getattr__(self, name):
        raise AssertionError(f"policy touched simulator/scorer: {name}")


def _frame(collar_on_shaft=False, occluded=False):
    img = np.full((176, 320, 3), (232, 220, 191), np.uint8)  # table
    img[shaft_view.WINDOW] = (150, 150, 152)  # grey flange
    if collar_on_shaft:
        img[100:106, 146:153] = (200, 90, 30)
    if occluded:
        img[shaft_view.WINDOW] = (25, 27, 29)  # dark arm link over the shaft
    return img


def _run(monkeypatch, frame, steps=2000):
    for name in ("mechanical_puller.embodiment", "mechanical_puller.scoring"):
        monkeypatch.setitem(sys.modules, name, _Poison())
    pol = policy.PullerScriptedPolicy()
    pol._bases = dict(station.BASES)
    pol.reset(_Poison())
    state = pack(np.r_[READY, 1.0], np.r_[READY, 1.0])
    labels, grips, stopped = [], [], None
    for i in range(steps):
        obs = types.SimpleNamespace(state={STATE_KEY: state}, images={"top_cam": frame}, state_time=i * 0.05)
        chunk = pol.act(obs)
        state = np.asarray(chunk.actions[0].data, float)  # stub robot: measured joints follow the command
        left, right = split(state)
        grips.append((float(left[6]), float(right[6])))
        labels.append(tuple(s.segs[s.index].label if 0 <= s.index < len(s.segs) else "done"
                            for s in pol._scripts.values()))
        if chunk.actions[0].meta.get("request_stop"):
            stopped = i
            break
    return pol, labels, grips, stopped


def test_full_plan_runs_on_observations_only(monkeypatch):
    pol, labels, grips, stopped = _run(monkeypatch, _frame())
    flat = {x for pair in labels for x in pair}
    for needed in ("supported screw extraction", "stroke 1 turn", "lift clear together", policy.GATE,
                   "lower onto cradle", "spread jaws to release collar", "withdraw holder", "done"):
        assert needed in flat, needed
    assert pol._gate == "passed"
    assert stopped is not None and stopped < 2000  # fits the 2,000-step limit
    assert min(grips[-1]) > 0.99  # both grippers open at the end
    # The complete bounded plan is used: wrist slip makes wrist turns insufficient as an early stop.
    assert pol.mem.expected_rise() >= policy.RISE_DONE
    assert "stroke 11 turn" in flat


def test_unconfirmed_collar_is_never_released(monkeypatch):
    for frame in (_frame(collar_on_shaft=True), _frame(occluded=True), None):
        pol, labels, grips, stopped = _run(monkeypatch, frame, steps=1700)
        assert pol._gate == "failed"
        flat = {x for pair in labels for x in pair}
        assert "lower onto cradle" not in flat and "spread jaws to release collar" not in flat
        assert grips[-1][1] < 0.5  # the holder keeps the tool


def test_shaft_view_reads_and_refuses():
    assert shaft_view.read(_frame()) == "gone"
    assert shaft_view.read(_frame(collar_on_shaft=True)) == "on"
    assert shaft_view.read(_frame(occluded=True)) is None
    assert shaft_view.read(None) is None
    assert shaft_view.read(np.zeros((10, 10, 3), np.uint8)) is None
    c = shaft_view.ClearConfirm(need=3)
    assert [c.update(r) for r in ("gone", "gone", None, "gone", "gone", "gone")] == [False] * 5 + [True]


def test_policy_source_has_no_privileged_reads():
    import inspect
    import re
    for module in (policy, shaft_view):
        code = inspect.getsource(module).split('"""', 2)[-1]  # after the module docstring
        code = "\n".join(line.split("#")[0] for line in code.splitlines())
        for pattern in (r"\.poses\(", r"\._base\b", r"\.scene\b", r"embodiment", r"ACTIVE", r"_policy_scripts",
                        r"scor"):
            assert not re.search(pattern, code), (module.__name__, pattern)
