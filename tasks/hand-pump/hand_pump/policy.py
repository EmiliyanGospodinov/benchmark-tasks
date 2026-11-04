"""Policy ``hand_pump_scripted``: privileged demonstration (reads simulator part poses and the gas state).

The right gripper takes the barrel, the left the handle; together they lift the pump off the mat. The left
gripper then strokes the handle while the right counterholds the barrel, until the chamber reaches the
manual's 60 kPa: it watches the pressure (privileged gas read, the value the dial shows) and stops the
stroke as soon as the target is reached. Both put the pump down and release.
Outputs ordinary 14-D joint targets through the guide's IK; never sets object poses.

``-P mode=idle`` keeps both arms at the ready pose (passive stability check); ``-P mode=nohold`` runs the
pumping arm alone (the barrel is not held).
"""
import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy

from . import design as d
from .motion import Script, Seg

BODY_SQUEEZE = -.012
HANDLE_SQUEEZE = -.017
LIFT = .040
TCP_PAD_OFFSET = .013
MAX_STROKES = 10
STOP_MARGIN_KPA = 1.0  # stop the stroke once within this of the target
PUSH_THROUGH = 0.06  # command the handle past the end stop: the compliant arms then press about 5 N


def _barrel_point(height=0.0):
    pos = d.BARREL_POS + np.array([0.0, 0.0, TCP_PAD_OFFSET + height])
    return lambda sim: (pos.copy(), grip_rotation(0.0))


def _handle_point(height=0.0, extension=d.STROKE):
    """Handle grip at a given extension, relative to the barrel's actual position (read on entry)."""
    def f(sim):
        b = sim.poses()["barrel"][:3] if sim is not None else d.BARREL_POS
        x = extension(sim) if callable(extension) else extension
        pos = np.array([b[0] + d.MIN_HANDLE_OFFSET + x, d.BARREL_POS[1], d.BARREL_POS[2] + TCP_PAD_OFFSET + height])
        return pos, grip_rotation(0.0)
    return f


def plans():
    left = [Seg(2.0, _handle_point(.06), FINGER_OPEN, "approach"),
            Seg(1.5, _handle_point(), FINGER_OPEN, "descend"),
            Seg(.8, _handle_point(), HANDLE_SQUEEZE, "grasp"),
            Seg(1.5, _handle_point(LIFT), HANDLE_SQUEEZE, "lift together")]
    right = [Seg(2.0, _barrel_point(.06), FINGER_OPEN, "approach"),
             Seg(1.5, _barrel_point(), FINGER_OPEN, "descend"),
             Seg(.8, _barrel_point(), BODY_SQUEEZE, "grasp"),
             Seg(1.5, _barrel_point(LIFT), BODY_SQUEEZE, "lift together")]
    for k in range(1, MAX_STROKES + 1):
        left += [Seg(1.5, _handle_point(LIFT, -PUSH_THROUGH), HANDLE_SQUEEZE, f"compress {k}"),
                 Seg(.3, None, HANDLE_SQUEEZE, "pause"),
                 Seg(1.5, _handle_point(LIFT), HANDLE_SQUEEZE, f"draw {k}"),
                 Seg(.3, None, HANDLE_SQUEEZE, "pause")]
        right += [Seg(3.6, _barrel_point(LIFT), BODY_SQUEEZE, "counterhold")]
    left += [Seg(1.5, _handle_point(), HANDLE_SQUEEZE, "lower together"),
             Seg(.7, None, FINGER_OPEN, "release"),
             Seg(1.3, _handle_point(.07), FINGER_OPEN, "retreat")]
    right += [Seg(1.5, _barrel_point(), BODY_SQUEEZE, "lower together"),
              Seg(.7, None, FINGER_OPEN, "release"),
              Seg(1.3, _barrel_point(.07), FINGER_OPEN, "retreat")]
    return {"left": left, "right": right}


def _start_of(segs, label):
    t = 0.0
    for s in segs:
        if s.label == label:
            return t
        t += s.duration
    raise KeyError(label)


def _label_at(segs, t):
    acc = 0.0
    for s in segs:
        if t < acc + s.duration:
            return s.label, acc
        acc += s.duration
    return "done", acc


class PumpScriptedPolicy(YamScriptedPolicy):
    def __init__(self, mode="coop", **kwargs):
        super().__init__(**kwargs)
        self.mode = str(mode)
        assert self.mode in ("coop", "idle", "nohold"), self.mode

    def reset(self, scene):
        p = plans()
        if self.mode == "idle":
            p = {}
        elif self.mode == "nohold":
            del p["right"]
        self._plans = p
        self._scripts = {arm: Script(segs, self._bases[arm]) for arm, segs in p.items()}
        self._offset = 0.0
        self._lower = _start_of(p["left"], "lower together") if p else 0.0
        self._end = (max(s.duration for s in self._scripts.values()) if p else 8.0) + 3.0
        self._printed = None

    def act(self, observation):
        from . import embodiment
        sim = embodiment.ACTIVE
        t = float(observation.state_time) + self._dt
        if "left" in self._plans:
            label, start = _label_at(self._plans["left"], t + self._offset)
            if label.startswith("compress") and sim.gas.gauge_kpa >= d.TARGET_KPA - STOP_MARGIN_KPA:
                self._offset = self._lower - t  # target reached: go straight to putting the pump down
                print(f"[pump-demo] t={t:.2f} chamber at {sim.gas.gauge_kpa:.1f} kPa: stop pumping", flush=True)
        actions, labels = [], []
        for arm in ("left", "right"):
            if arm in self._scripts:
                a, label = self._scripts[arm].act(sim, t + self._offset)
            else:
                a, label = np.r_[READY, 1.0], "idle"
            actions.append(a)
            labels.append(label)
        if tuple(labels) != self._printed:
            print(f"[pump-demo] t={t:.2f} phases={labels} "
                  f"ik_worst_mm={max([s.worst_mm for s in self._scripts.values()] or [0]):.1f}", flush=True)
            self._printed = tuple(labels)
        meta = {"request_stop": True} if t + self._offset > self._end + self.hold_s else {}
        return ActionChunk(actions=[Action(data=pack(*actions), meta=meta)], control_hz=1 / self._dt)
