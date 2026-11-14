"""Policy ``mechanical_puller_scripted``: scripted demonstration that uses only what a deployed policy observes.

Inputs:
  * proprioception ``observation.state[STATE_KEY]``: both arms' measured joints and gripper openings;
  * the overhead camera image ``observation.images["top_cam"]`` (native 320x176), read by ``shaft_view``;
  * calibrated robot geometry (``yam_arena.station.BASES`` and the guide's forward kinematics ``ik.tcp_pose``);
  * the published, fixed start layout (the prepared puller on the shaft axis at ``design.FIX``, fitted height, level;
    the tray and rest position) and the published screw lead;
  * elapsed time and the policy's own memory (grasp transforms captured from measured joints).
It never reads simulator part poses or joint data, the embodiment object or the scorer.

Body pose: before the right gripper grasps, the published start layout; afterwards, right-hand FK of the measured
joints composed with the grasp transform captured from the measured joints at grasp time; after the right hand lets
go, the last estimate. Arms are commanded by position only; there is no integral correction (an integrated
commanded-minus-measured error winds up while a hand is in contact and pushes the tool).

Screw: wrist rotation estimates the screw advance for holder following. Slip makes it an imperfect estimate;
all 11 bounded turning strokes are executed rather than using wrist motion to assert extraction.
The later camera gate must confirm the collar left before release.

Camera gate: after the tool has been carried over the tray, the overhead camera must show the shaft flange
readable and free of the yellow collar for 5 consecutive frames before the tool is lowered and the jaws are spread. Until then
both arms hold still; if it is not confirmed within 4 s the policy keeps holding and does not release anything.

``-P stage=idle`` keeps both arms at READY.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import STATE_KEY, pack, split
from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy

from . import design as d
from . import shaft_view

TILT = math.radians(20)
HOLD_TILT = math.radians(60)
BOSS_SQUEEZE = -0.012
BAR_OPEN = FINGER_OPEN
KNOB_SQUEEZE = -0.012
YAW_START, YAW_END = math.radians(40), math.radians(-90)
MAX_STROKES = 11  # bounded full stroke sequence; no wrist-derived early extraction claim
RISE_DONE = d.COLLAR_CLEAR + 0.004  # holder rise (m) that shows the collar bore has left the shaft
BODY0 = np.r_[d.FIX, d.TOP + d.FIT_Z]  # published start: body origin on the shaft axis, fitted height, level
ARMS = ("left", "right")
GATE = "confirm collar left the shaft"  # both arms hold over the tray until the camera confirms
GATE_TIMEOUT = 4.0


def front_grip(tilt):
    z = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
    x = np.array([0.0, -1.0, 0.0])
    return np.c_[x, np.cross(z, x), z]


def side_grip(tilt, from_side):
    z = np.array([0.0, -from_side * math.cos(tilt), -math.sin(tilt)])
    x = np.array([-1.0, 0.0, 0.0])
    return np.c_[x, np.cross(z, x), z]


class Memory:
    """Everything the policy knows: measured joints this step, FK, and what it remembered."""

    def __init__(self):
        self.q = {a: np.array(READY, float) for a in ARMS}
        self.grip = {a: 1.0 for a in ARMS}
        self.grasp = {}  # arm -> (offset in body frame, hand rotation in body frame), from measured joints
        self.body_frozen = (BODY0.copy(), np.eye(3))  # body pose while no hand holds it
        self.holder_z0 = None  # holder TCP height when the extraction starts (measured)
        self.holder_floor = None  # one-way follow: lowest commanded holder height
        self.turns = 0.0  # knob turns made, from the left wrist's MEASURED rotation while it grips the knob
        self._q5 = None

    def expected_rise(self):
        """Tool rise expected from the screw: published lead x turns measured at the left wrist."""
        return d.LEAD * self.turns

    def tcp(self, arm):
        return ik.tcp_pose(self.q[arm], station.BASES[arm])

    def body(self):
        """Body pose estimate: from the holding right hand while it holds, else the last known pose."""
        if "right" in self.grasp:
            hp, hr = self.tcp("right")
            off, rot = self.grasp["right"]
            r = hr @ rot.T
            return hp - r @ off, r
        return self.body_frozen

    def capture(self, arm):
        """Remember the grasp transform from the measured joints and the current body estimate."""
        bp, br = self.body()
        hp, hr = self.tcp(arm)
        self.grasp[arm] = (br.T @ (hp - bp), br.T @ hr)

    def release(self, arm):
        if arm == "right" and "right" in self.grasp:
            self.body_frozen = self.body()
        self.grasp.pop(arm, None)


@dataclass
class Seg:
    duration: float
    target: Callable  # mem -> (pos, rot)
    finger: float
    label: str
    freeze: bool = True
    on_end: Callable = None  # mem -> None, run once when the segment ends (capture/release grasp memory)


class Script:
    def __init__(self, arm, segs):
        self.arm, self.segs, self.base = arm, segs, np.asarray(station.BASES[arm], float)
        self.q = np.array(READY, float)
        self.prev = ik.tcp_pose(self.q, self.base)
        self.prev_finger = FINGER_OPEN
        self.index, self.goal = -1, None
        self.worst_mm = self.worst_deg = 0.0

    @property
    def duration(self):
        return sum(s.duration for s in self.segs)

    def act(self, mem, t):
        acc = 0.0
        for i, s in enumerate(self.segs):
            if t < acc + s.duration:
                break
            acc += s.duration
        else:
            i, s = len(self.segs), None
        if i != self.index:
            for k in range(max(self.index, 0), i):  # segments just finished, in order
                if self.index >= 0 and self.segs[k].on_end is not None:
                    self.segs[k].on_end(mem)
            if self.goal is not None:
                self.prev, self.prev_finger = self.goal[:2], self.goal[2]
            self.index = i
            self.goal = None if s is None else (*(self.prev if s.target is None else s.target(mem)), s.finger)
        if s is None:
            pos, rot, finger = *self.prev, self.prev_finger
        else:
            if not s.freeze and s.target is not None:
                self.goal = (*s.target(mem), s.finger)
            u = ik.smoothstep((t - acc) / s.duration)
            pos = self.prev[0] + u * (self.goal[0] - self.prev[0])
            rot = ik.slerp_matrix(self.prev[1], self.goal[1], u)
            finger = self.prev_finger + u * (s.finger - self.prev_finger)
        label = "done" if s is None else s.label
        pos = np.asarray(pos, float)
        q, pe, ae = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q = q
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(ae))
        return np.r_[q, float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))], label


def _body_point(local, rot, back=0.0, lift=0.0, at=None):
    """TCP at ``local`` in the estimated body frame (or with the body placed at ``at`` = (x, y, height))."""
    def f(mem):
        if at is None:
            origin, r = mem.body()
        else:
            origin, r = np.array([at[0], at[1], d.TOP + at[2]]), np.eye(3)
        return origin + r @ np.asarray(local, float) + np.array([0, 0, lift]) - back * rot[:, 2], rot
    return f


def _placed(arm, at):
    """Hand pose that puts the body at ``at`` (level), through the remembered grasp transform of ``arm``."""
    def f(mem):
        off, grot = mem.grasp[arm]
        goal = np.array([at[0], at[1], d.TOP + at[2]])
        return goal + off, grot
    return f


def _follow_up(mem):
    """Counterhold target: hold the remembered fitted pose, but follow the tool up one-way as the collar comes off
    (from the holder's measured TCP height; the tool cannot be pulled down, only allowed to rise)."""
    off, grot = mem.grasp["right"]
    hp, _ = mem.tcp("right")
    base = BODY0 + off
    if mem.holder_z0 is None:
        mem.holder_z0 = hp[2]
    # the tool slides up through the pinch if the holder only reacts to being pushed (rev-001: hand +7 mm while the
    # body rose 24 mm), so the holder also rises with the screw advance computed from the measured knob turns
    rise = max(0.0, hp[2] - mem.holder_z0, mem.expected_rise())
    mem.holder_floor = max(mem.holder_floor or 0.0, rise)
    return base + np.array([0, 0, mem.holder_floor]), grot


BOSS = np.array([0.0, -d.BOSS[1] / 2 - d.HANDLE[1] - d.GRIP_BLOCK[1] / 2 + 0.005, d.GRIP_Z])
KNOB = np.array([0.0, 0.0, d.KNOB_Z + 0.004])
JAWPOINT = np.array([0.0, 0.0, -0.035])


def complete_plans():
    hold, close = front_grip(HOLD_TILT), side_grip(TILT, +1)
    right = [Seg(2.5, _body_point(BOSS, hold, back=.09), BAR_OPEN, "support approach"),
             Seg(1.5, _body_point(BOSS, hold, back=.013), BAR_OPEN, "support reach"),
             Seg(.8, _body_point(BOSS, hold, back=.013), BOSS_SQUEEZE, "support grasp",
                 on_end=lambda m: m.capture("right"))]
    knob = lambda yaw, dz=0.: _body_point(KNOB, grip_rotation(yaw), lift=dz)  # noqa: E731
    left = [Seg(5., None, FINGER_OPEN, "wait for support"), Seg(1.5, knob(YAW_START, .05), FINGER_OPEN, "knob approach")]
    for k in range(MAX_STROKES):
        left += [Seg(.8, knob(YAW_START), FINGER_OPEN, f"stroke {k + 1} descend"),
                 Seg(.5, knob(YAW_START), KNOB_SQUEEZE, f"stroke {k + 1} grip"),
                 Seg(1.5, knob(YAW_END), KNOB_SQUEEZE, f"stroke {k + 1} turn"),
                 Seg(.4, None, FINGER_OPEN, f"stroke {k + 1} open"),
                 Seg(.3, knob(YAW_END, .02), FINGER_OPEN, f"stroke {k + 1} clear"),
                 Seg(.8, knob(YAW_START, .02), FINGER_OPEN, f"stroke {k + 1} return")]
    left += [Seg(1., knob(YAW_START, .04), FINGER_OPEN, "knob retreat")]
    right += [Seg(sum(v.duration for v in left) - sum(v.duration for v in right), _follow_up, BOSS_SQUEEZE,
                  "supported screw extraction", freeze=False)]
    tx, ty = d.TRAY_XY  # aim x +2 mm, y 0: rev-003 settled at aim x -3.6, y -1.4 mm, +1.5 mm y on release
    left += [Seg(1., _body_point(KNOB, grip_rotation(YAW_START)), FINGER_OPEN, "regrasp transport knob"),
             Seg(.6, None, KNOB_SQUEEZE, "grasp transport knob", on_end=lambda m: m.capture("left"))]
    right += [Seg(1.6, None, BOSS_SQUEEZE, "wait for two-hand carry")]
    for x, y, z, duration, label in [(d.FIX[0], d.FIX[1], .125, 2.5, "lift clear together"),
                                     (tx + .002, ty, .125, 3.5, "carry together over cradle"),
                                     (tx + .002, ty, .125, .5, GATE),
                                     (tx + .002, ty, d.PARK_Z, 2.5, "lower onto cradle"),
                                     (tx + .002, ty, d.PARK_Z, 1.5, "settle on cradle")]:
        for side, finger in (("right", BOSS_SQUEEZE), ("left", KNOB_SQUEEZE)):
            (right if side == "right" else left).append(Seg(duration, _placed(side, (x, y, z)), finger, label))
    left += [Seg(.6, None, FINGER_OPEN, "release transport knob", on_end=lambda m: m.release("left")),
             Seg(1., _body_point(KNOB, grip_rotation(YAW_START), lift=.05), FINGER_OPEN, "clear knob"),
             Seg(1.5, _body_point(JAWPOINT, close, back=.08), -.010, "approach inside jaws"),
             Seg(1.2, _body_point(JAWPOINT, close), -.010, "insert fingers between jaws"),
             Seg(1.2, None, FINGER_OPEN, "spread jaws to release collar"),
             Seg(.6, None, FINGER_OPEN, "let collar settle"),
             Seg(2., _body_point(JAWPOINT, close, back=.10), FINGER_OPEN, "withdraw from jaws")]
    right += [Seg(8.1, None, BOSS_SQUEEZE, "steady resting tool"),
              Seg(.8, None, FINGER_OPEN, "release parked tool", on_end=lambda m: m.release("right")),
              Seg(2., _body_point(BOSS, hold, back=.10), FINGER_OPEN, "withdraw holder"),
              Seg(2.5, None, FINGER_OPEN, "verify stable release")]
    return {"left": left, "right": right}


def _start_of(segs, label):
    t = 0.0
    for s in segs:
        if s.label == label:
            return t
        t += s.duration
    raise KeyError(label)


class PullerScriptedPolicy(YamScriptedPolicy):
    def __init__(self, stage="complete", duration=8.0, **kwargs):
        super().__init__(**kwargs)
        self.stage, self.duration = str(stage), float(duration)

    def reset(self, scene):
        self.mem = Memory()
        p = {} if self.stage in ("idle", "probe") else complete_plans()
        self._scripts = {a: Script(a, s) for a, s in p.items()}
        self._offset = 0.0
        self._after_strokes = _start_of(p["left"], "regrasp transport knob") - 1.0 if p else 0.0
        self._end = max([s.duration for s in self._scripts.values()] or [self.duration]) + 2.0
        self._printed = None
        self.log = []
        self._confirm = shaft_view.ClearConfirm()
        self._gate = "pending"  # pending -> passed | failed
        self._gate_wait = 0.0
        self._t_hold = None

    def _observe(self, observation):
        left, right = split(observation.state[STATE_KEY])
        for a, v in (("left", left), ("right", right)):
            self.mem.q[a], self.mem.grip[a] = np.asarray(v[:6], float), float(v[6])

    def _count_turns(self):
        s = self._scripts.get("left")
        q5 = float(self.mem.q["left"][5])
        if s is not None and 0 <= s.index < len(s.segs) and s.segs[s.index].label.endswith(" turn") \
                and self.mem._q5 is not None:
            self.mem.turns += max(0.0, q5 - self.mem._q5) / (2 * math.pi)  # tightening (clockwise from above) raises joint 6
        self.mem._q5 = q5

    def act(self, observation):
        self._observe(observation)
        self._count_turns()
        t = float(observation.state_time) + self._dt + self._offset
        # Execute the bounded extraction strokes. Wrist turns cannot prove ram
        # advance when the knob slips; only the later camera gate permits release.
        if "left" in self._scripts and self._gate != "passed":
            s = self._scripts["left"]
            label = s.segs[s.index].label if 0 <= s.index < len(s.segs) else ""
            if label == GATE or self._gate == "failed":
                reading = shaft_view.read((getattr(observation, "images", None) or {}).get("top_cam"))
                if self._gate == "pending" and self._confirm.update(reading):
                    self._gate = "passed"
                    print(f"[puller-demo] top camera: shaft flange readable and free of the yellow collar, collar left the shaft",
                          flush=True)
                else:
                    if self._gate == "pending":
                        self._gate_wait += self._dt
                        if self._gate_wait > GATE_TIMEOUT:
                            self._gate = "failed"
                            print("[puller-demo] top camera did not confirm the collar left the shaft: holding, "
                                  "nothing is released", flush=True)
                    if self._t_hold is None:
                        self._t_hold = t
                    self._offset += self._t_hold - t  # hold: time in the plan does not advance
                    t = self._t_hold
        out, labels = [], []
        for a in ARMS:
            if a in self._scripts:
                act, label = self._scripts[a].act(self.mem, t)
            else:
                act, label = np.r_[READY, 1.0], "-"
            out.append(act)
            labels.append(label)
        if self._scripts and tuple(labels) != self._printed:
            print(f"[puller-demo] t={t:5.2f} left={labels[0]} right={labels[1]} "
                  f"ik_worst_mm={max(s.worst_mm for s in self._scripts.values()):.1f}", flush=True)
            self._printed = tuple(labels)
        meta = {"request_stop": True} if t > self._end else {}
        return ActionChunk(actions=[Action(data=pack(*out), meta=meta)], control_hz=1 / self._dt)
