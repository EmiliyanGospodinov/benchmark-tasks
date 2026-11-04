"""Policy ``mount_scripted``: a privileged feasibility demonstration (reads simulator part poses).

The right gripper takes the inspection head and swings the arm up to the requested aim; the left
gripper tightens the clamp knob in regrasp strokes while the right one keeps the aim; then both
release. It outputs ordinary 14-D joint targets through the guide's IK and never sets object poses.
``-P mode=nolock`` aims and releases without tightening; ``-P mode=prelock`` tightens the knob
first, then tries to aim (the one-arm shortcut comparisons).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy

from . import design as d

HEAD_SQUEEZE = -0.012  # finger target inside the 40 mm head: the pads stall on it
KNOB_GRIP_Z = d.KNOB_CENTER[2] + d.KNOB_H / 2 - 0.006
KNOB_SQUEEZE = -0.010
YAW_START, YAW_END = math.radians(40), math.radians(-90)  # one tightening stroke (clockwise)
LIFT_STEPS = 4
AIM_BIAS_DEG = 4.0  # command above the mark: the position-controlled arm sags under the load


@dataclass
class Seg:
    duration: float
    target: Callable  # sim -> (pos, rot)
    finger: float
    label: str
    freeze: bool = True  # evaluate the target once, on entering the segment


class Script:
    def __init__(self, segs, base):
        self.segs, self.base = segs, np.asarray(base, float)
        self.q = np.array(READY, float)
        self.prev = ik.tcp_pose(self.q, self.base)
        self.prev_finger = FINGER_OPEN
        self.index, self.goal = -1, None
        self.worst_mm = 0.0

    @property
    def duration(self):
        return sum(s.duration for s in self.segs)

    def act(self, sim, t):
        acc = 0.0
        for i, s in enumerate(self.segs):
            if t < acc + s.duration:
                break
            acc += s.duration
        else:
            i, s = len(self.segs), None
        if i != self.index:
            if self.goal is not None:
                self.prev, self.prev_finger = self.goal[:2], self.goal[2]
            self.index = i
            self.goal = None if s is None else (*(self.prev if s.target is None else s.target(sim)), s.finger)
        if s is None:
            pos, rot, finger, label = *self.prev, self.prev_finger, "done"
        else:
            if not s.freeze and s.target is not None:
                self.goal = (*s.target(sim), s.finger)
            u = ik.smoothstep((t - acc) / s.duration)
            pos = self.prev[0] + u * (self.goal[0] - self.prev[0])
            rot = ik.slerp_matrix(self.prev[1], self.goal[1], u)
            finger = self.prev_finger + u * (s.finger - self.prev_finger)
            label = s.label
        q, pe, _ = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q = q
        self.worst_mm = max(self.worst_mm, pe * 1000)
        return np.r_[q, float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))], label


def _head_point(phi_deg, back=0.0):
    def f(sim):
        # vertical pinch on the head's side faces, across the hinge axis: the head may pivot in the
        # pads, and its centre position on the arc sets the arm angle
        pos = d.head_center(math.radians(phi_deg)) + np.array([0.0, 0.0, back])
        return pos, grip_rotation(math.pi / 2)
    return f


def _knob_point(dz, yaw):
    def f(sim):
        return np.array([d.KNOB_CENTER[0], d.KNOB_CENTER[1], KNOB_GRIP_Z + dz]), grip_rotation(yaw)
    return f


def holder_plan(start, hold_until):
    """Right gripper: take the head at the rest pose, swing it to the aim, hold, release."""
    rest, aim = d.PHI_REST_DEG, -d.TARGET_ELEV_DEG - AIM_BIAS_DEG
    segs = [Seg(max(0.1, start), None, FINGER_OPEN, "wait"),  # stay at the ready pose
            Seg(3.0, _head_point(rest, 0.06), FINGER_OPEN, "head approach"),
            Seg(1.0, _head_point(rest), FINGER_OPEN, "head descend"),
            Seg(0.7, _head_point(rest), HEAD_SQUEEZE, "head grip")]
    for k in range(1, LIFT_STEPS + 1):
        segs.append(Seg(1.0, _head_point(rest + (aim - rest) * k / LIFT_STEPS), HEAD_SQUEEZE, f"aim {k}"))
    t = sum(s.duration for s in segs)
    segs += [Seg(max(0.1, hold_until - t), _head_point(aim), HEAD_SQUEEZE, "hold aim"),
             Seg(0.6, _head_point(aim), FINGER_OPEN, "head release"),
             Seg(1.0, _head_point(aim, 0.12), FINGER_OPEN, "head retreat")]
    return segs


def holder_ready_time():
    return 0.1 + 3.0 + 1.0 + 0.7 + LIFT_STEPS * 1.0


def _knob_turn(limit_deg):
    """Turn by at most one stroke, and no further than limit_deg of knob rotation (privileged read)."""
    def f(sim):
        need = min(math.degrees(YAW_START - YAW_END), limit_deg - sim.tightened_deg)
        return _knob_point(0.0, YAW_START - math.radians(max(10.0, need)))(sim)
    return f


def _lift(dz):
    """Straight up from the knob, keeping the current wrist yaw."""
    def f(sim):
        script = sim._policy_scripts["left"]
        pos, rot = script.prev
        return np.array([d.KNOB_CENTER[0], d.KNOB_CENTER[1], KNOB_GRIP_Z + dz]), rot
    return f


SNUG_DEG = d.ENGAGE_DEG - 30.0  # first stroke: take up the knob's free travel, well short of meshing


def _stroke(k, limit_deg, last):
    segs = [
        Seg(1.0, _knob_point(0.0, YAW_START), FINGER_OPEN, f"stroke {k} descend"),
        Seg(0.6, _knob_point(0.0, YAW_START), KNOB_SQUEEZE, f"stroke {k} grip"),
        Seg(2.0, _knob_turn(limit_deg), KNOB_SQUEEZE, f"stroke {k} turn"),
        Seg(0.5, None, FINGER_OPEN, f"stroke {k} open"),
        Seg(0.4, _lift(0.02), FINGER_OPEN, f"stroke {k} clear"),
    ]
    if not last:
        segs.append(Seg(1.0, _knob_point(0.02, YAW_START), FINGER_OPEN, f"stroke {k} return"))
    return segs


def turner_plan(final_start, snug_deg=SNUG_DEG):
    """Left gripper: snug the knob (short of meshing) while the arm is being raised, then, once the
    head is at the aim (final_start), tighten it to its stop; withdraw."""
    segs = [Seg(3.5, _knob_point(0.05, YAW_START), FINGER_OPEN, "knob approach")]  # snug turn overlaps the lift
    segs += _stroke(1, snug_deg, last=False)
    t = sum(s.duration for s in segs)
    segs.append(Seg(max(0.1, final_start - t), _knob_point(0.02, YAW_START), FINGER_OPEN, "wait for aim"))
    segs += _stroke(2, d.KNOB_STOP_DEG + 3.0, last=True)
    segs.append(Seg(1.0, _lift(0.10), FINGER_OPEN, "knob retreat"))
    return segs


class MountScriptedPolicy(YamScriptedPolicy):
    def __init__(self, mode="coop", **kwargs):
        super().__init__(**kwargs)
        self.mode = str(mode)
        assert self.mode in ("coop", "nolock", "prelock"), self.mode

    def reset(self, scene):
        plans = {}
        if self.mode == "coop":
            ready = holder_ready_time() + 0.3
            plans["left"] = turner_plan(ready)
            plans["right"] = holder_plan(0.0, sum(s.duration for s in plans["left"][:-1]) + 0.5)
        elif self.mode == "nolock":
            plans["right"] = holder_plan(0.0, holder_ready_time() + 1.0)
        else:  # prelock: tighten first, then try to aim with the same scene
            plans["left"] = turner_plan(0.0, snug_deg=d.KNOB_STOP_DEG + 3.0)  # tighten fully first
            t = sum(s.duration for s in plans["left"])
            plans["right"] = holder_plan(t, t + holder_ready_time() + 1.0)
        self._scripts = {a: Script(s, self._bases[a]) for a, s in plans.items()}
        from . import embodiment
        if embodiment.ACTIVE is not None:
            embodiment.ACTIVE._policy_scripts = self._scripts
        self._end = max(s.duration for s in self._scripts.values()) + 5.0  # watch the released arm
        self._printed = None

    def act(self, observation):
        from . import embodiment

        sim = embodiment.ACTIVE
        t = float(observation.state_time) + self._dt
        out, labels = [], []
        for a in ("left", "right"):
            if a in self._scripts:
                act, label = self._scripts[a].act(sim, t)
            else:
                act, label = np.r_[READY, 1.0], "-"
            out.append(act)
            labels.append(label)
        if tuple(labels) != self._printed:
            print(f"[mount-demo] t={t:5.2f} left={labels[0]} right={labels[1]} "
                  f"ik_worst_mm={max(s.worst_mm for s in self._scripts.values()):.1f}", flush=True)
            self._printed = tuple(labels)
        meta = {"request_stop": True} if t > self._end + self.hold_s else {}
        return ActionChunk(actions=[Action(data=pack(*out), meta=meta)], control_hz=1.0 / self._dt)
