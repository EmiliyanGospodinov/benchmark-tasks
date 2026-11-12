"""Observation-only hand-pump demo: native overhead RGB and exposed robot joint state.

Known station calibration and robot forward/inverse kinematics support Cartesian
motion. Pressure decisions use only visible dial pixels. No simulator handles.
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
PUSH_THROUGH = 0.06  # command the handle past the end stop: the compliant arms then press about 5 N


def _barrel_point(height=0.0):
    pos = d.BARREL_POS + np.array([0.0, 0.0, TCP_PAD_OFFSET + height])
    return lambda sim: (pos.copy(), grip_rotation(0.0))


def _barrel_x(ctx):
    """Estimate barrel x from the grasping robot's observed joints and fixed setup."""
    return ctx.barrel_pos()


def _handle_point(height=0.0, extension=d.STROKE):
    """Handle grip at a given extension, relative to the barrel's position (read on entry)."""
    def f(sim):
        b = _barrel_x(sim)
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


class _PumpMotion(YamScriptedPolicy):
    """Initialize motion plans without any scene access."""
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

# --- observation-only demonstration (REV-A6) ----------------------------------------------------------------
from inspect_robots_yam.packing import STATE_KEY, split  # noqa: E402
from yam_arena import ik  # noqa: E402

from . import dial_reader as dr  # noqa: E402




class ObsContext:
    """Everything the observation-only demo knows: the robot's own joint state and the overhead RGB image."""

    def __init__(self, bases):
        self.bases, self.q, self.image, self.grasped = bases, None, None, False

    def update(self, observation):
        left, right = split(np.asarray(observation.state[STATE_KEY], float))
        self.q = {"left": left, "right": right}
        self.image = observation.images["top_cam"]

    def tcp(self, arm):
        return ik.tcp_pose(self.q[arm][:6], self.bases[arm])[0]

    def barrel_pos(self):
        if not self.grasped:
            return d.BARREL_POS  # documented fixed start layout (task setup), not a simulator read
        t = self.tcp("right")
        return np.array([t[0], d.BARREL_POS[1], d.BARREL_POS[2]])


class PumpScriptedPolicy(_PumpMotion):
    """Observation-only demonstration: robot joint state + native 320x176 overhead RGB. No simulator access."""

    def reset(self, scene):
        super().reset(scene)
        self.ctx = ObsContext(self._bases)
        self.reader = dr.PressureStop()
        self._stopped = False

    def act(self, observation):
        self.ctx.update(observation)
        t = float(observation.state_time) + self._dt
        if "right" in self._plans:
            rl, _ = _label_at(self._plans["right"], t + self._offset)
            self.ctx.grasped = self.ctx.grasped or rl in ("lift together", "counterhold")
        if "left" in self._plans and not self._stopped:
            label, start = _label_at(self._plans["left"], t + self._offset)
            if label.startswith("compress"):
                stop, est = self.reader.update(self.ctx.image, pumping=True)
                if stop:
                    self._offset = self._lower - t
                    self._stopped = True
                    print(f"[pump-demo] t={t:.2f} dial reads {est:.1f} kPa (overhead RGB): stop pumping", flush=True)
            else:
                self.reader.update(self.ctx.image, pumping=False)
        actions, labels = [], []
        for arm in ("left", "right"):
            if arm in self._scripts:
                a, label = self._scripts[arm].act(self.ctx, t + self._offset)
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
