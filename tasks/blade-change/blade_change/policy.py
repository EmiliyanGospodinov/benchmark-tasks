"""Policy ``blade_change_scripted``: a privileged demonstration (reads simulator part poses).

The left gripper pinches the collar's lever tab and turns the collar to release, holding it there;
the right gripper withdraws the worn blade, drops it in the rest tray, takes the replacement from its
rack, inserts it to the shoulder; the left lets the collar spring back (the pin drops into the notch);
the right tugs outward (the spring carriage extends) and lets go. Ordinary 14-D joint targets through
the guide's IK; object poses are never set. Comparison modes (``-P mode=...``):
``closed``  the right gripper pulls the installed blade with the collar untouched;
``early``   the left lets the collar return while the tang is still half inside;
``push``    the collar is released before inserting: the replacement is pushed against the pin.
"""

from __future__ import annotations

import math

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy

from . import design as d
from .motion import Script, Seg

LEVER_SQUEEZE = -0.004  # 8 mm target on the 12 mm tab (across its travel direction)
BLADE_SQUEEZE = -0.007  # 14 mm target on the 20 mm grip block
LEVER_GRIP_R = d.LEVER_R - 0.008  # TCP on the tab, from the axis
TURN_DEG = -37.0  # short of the -40 stop
ALONG_X = grip_rotation(math.pi / 2)  # straight down, fingers closing along x (across the blade)
ALONG_Y = grip_rotation(0.0)  # straight down, fingers closing along y (across the lever tab)
WITHDRAW = d.TANG_L + 0.012
TRAY_GRIP_DY = d.GRIP_FROM_SHOULDER - (d.BODY_L - d.TANG_L) / 2  # grip block to the blade's middle, along -y


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _lever(deg, up=0.0):
    a = math.radians(deg)
    def f(sim):
        o = np.array([d.AXIS_X, d.COLLAR_Y, d.AXIS_Z])
        return o + rot_y(a) @ np.array([0.0, 0.0, LEVER_GRIP_R + up]), rot_y(a) @ ALONG_X  # pads push the tab
    return f


def _on_axis(shoulder_y, dz=0.0, x=d.AXIS_X):
    """TCP on a blade held at its grip point, with its shoulder at shoulder_y on the axis line."""
    return lambda sim: (np.array([x, shoulder_y - d.GRIP_FROM_SHOULDER, d.AXIS_Z + dz]), ALONG_X)


_HAND = {}


def _tang_to(shoulder_y, mode="hold"):
    """TCP that puts the held replacement's shoulder on the axis at shoulder_y (privileged).
    mode "hand": measure where the blade sits in the hand (offset from the measured TCP);
    mode "correct": add the remaining measured shoulder error (arm tracking/sag) to the command;
    mode "hold": reuse both."""
    def f(sim):
        pose = sim.poses()["new_blade"]
        rb, shoulder = d.quat_to_matrix(pose[3:]), pose[:3]
        if mode == "hand" or "offset" not in _HAND:
            _HAND["offset"] = shoulder - (sim.tcp("right") + rb @ np.array([0.0, d.GRIP_FROM_SHOULDER, 0.0]))
            _HAND["bias"] = np.zeros(3)
        if mode == "correct":
            err = np.array([d.AXIS_X, shoulder[1], d.AXIS_Z]) - shoulder
            _HAND["bias"] = _HAND["bias"] + err * [1.0, 0.0, 1.0]
        nominal = np.array([d.AXIS_X, shoulder_y - d.GRIP_FROM_SHOULDER, d.AXIS_Z])
        return nominal - _HAND["offset"] + _HAND["bias"], ALONG_X
    return f


def _fixed(pos, rot=ALONG_X):
    return lambda sim: (np.asarray(pos, float), rot)


def collar_plan(turn_at, release_at, turn=TURN_DEG):
    segs = [Seg(max(0.1, turn_at - 5.6), None, FINGER_OPEN, "wait"),  # turned at turn_at
            Seg(2.5, _lever(0.0, 0.05), FINGER_OPEN, "lever approach"),
            Seg(1.0, _lever(0.0), FINGER_OPEN, "lever descend"),
            Seg(0.6, _lever(0.0), LEVER_SQUEEZE, "lever grip"),
            Seg(1.5, _lever(turn), LEVER_SQUEEZE, "collar turn")]
    t = sum(s.duration for s in segs)
    segs += [Seg(max(0.1, release_at - t), None, LEVER_SQUEEZE, "collar hold"),
             Seg(1.0, _lever(0.0), LEVER_SQUEEZE, "collar return"),
             Seg(0.5, None, FINGER_OPEN, "lever release"),
             Seg(1.0, _lever(0.0, 0.07), FINGER_OPEN, "lever retreat")]
    return segs


def removal_segs(start, withdraw=WITHDRAW):
    n = d.NOSE_Y
    tx, ty = d.TRAY_XY
    return [Seg(max(0.1, start - 4.1), None, FINGER_OPEN, "wait"),  # withdraw starts at start
            Seg(2.5, _on_axis(n, 0.05), FINGER_OPEN, "old approach"),
            Seg(1.0, _on_axis(n), FINGER_OPEN, "old descend"),
            Seg(0.6, _on_axis(n), BLADE_SQUEEZE, "old grip"),
            Seg(2.0, _on_axis(n - withdraw), BLADE_SQUEEZE, "old withdraw"),
            Seg(1.0, _on_axis(n - withdraw, 0.05), BLADE_SQUEEZE, "old lift"),
            Seg(2.0, _fixed((tx, ty - TRAY_GRIP_DY, d.TOP + 0.10)), BLADE_SQUEEZE, "old carry"),
            Seg(1.0, _fixed((tx, ty - TRAY_GRIP_DY, d.TOP + 0.045)), BLADE_SQUEEZE, "old lower"),
            Seg(0.5, None, FINGER_OPEN, "old release"),
            Seg(0.8, _fixed((tx, ty - TRAY_GRIP_DY, d.TOP + 0.10)), FINGER_OPEN, "old clear")]


def insertion_segs():
    rest = d.blade_start(False)
    rx, ry, rz = rest[:3]
    gy = ry - d.GRIP_FROM_SHOULDER
    n = d.NOSE_Y
    segs = [Seg(2.0, _fixed((rx, gy, rz + 0.05)), FINGER_OPEN, "new approach"),
            Seg(1.0, _fixed((rx, gy, rz)), FINGER_OPEN, "new descend"),
            Seg(0.6, _fixed((rx, gy, rz)), BLADE_SQUEEZE, "new grip"),
            Seg(1.0, _fixed((rx, gy, rz + 0.045)), BLADE_SQUEEZE, "new lift"),
            Seg(2.0, _tang_to(n - 0.06, "hand"), BLADE_SQUEEZE, "new align"),
            Seg(0.8, _tang_to(n - 0.06, "correct"), BLADE_SQUEEZE, "new realign"),
            Seg(0.6, _tang_to(n - 0.06, "correct"), BLADE_SQUEEZE, "new realign 2"),
            Seg(2.0, _tang_to(n + 0.0015), BLADE_SQUEEZE, "new insert"),
            Seg(0.5, None, BLADE_SQUEEZE, "new seat hold")]
    return segs


def tug_segs():
    n = d.NOSE_Y
    return [Seg(1.0, _on_axis(n - 0.016), BLADE_SQUEEZE, "tug"),
            Seg(1.2, None, BLADE_SQUEEZE, "tug hold"),
            Seg(0.8, _on_axis(n), BLADE_SQUEEZE, "tug relax"),
            Seg(0.5, None, FINGER_OPEN, "blade release"),
            Seg(1.0, _on_axis(n, 0.08), FINGER_OPEN, "blade retreat")]


class BladeScriptedPolicy(YamScriptedPolicy):
    def __init__(self, mode="coop", **kwargs):
        super().__init__(**kwargs)
        self.mode = str(mode)
        assert self.mode in ("coop", "closed", "early", "push"), self.mode

    def reset(self, scene):
        turned = 0.1 + 4.1 + 1.5  # collar_plan reaches the turned pose at this time
        if self.mode == "closed":
            plans = {"right": removal_segs(turned)[:5]}
        elif self.mode == "early":
            right = removal_segs(turned)
            grip_done = sum(s.duration for s in right[:4])
            plans = {"right": right[:6], "left": collar_plan(turned, grip_done + 0.9)}  # tang half out
        else:
            right = removal_segs(turned) + insertion_segs()
            seated_at = sum(s.duration for s in right)
            if self.mode == "push":
                removed = sum(s.duration for s in removal_segs(turned)[:5]) + 0.5
                plans = {"left": collar_plan(turned, removed), "right": right}
            else:
                plans = {"left": collar_plan(turned, seated_at), "right": right + [Seg(1.6, None, BLADE_SQUEEZE, "wait")]
                         + tug_segs()}
        self._scripts = {a: Script(s, self._bases[a]) for a, s in plans.items()}
        self._end = max(s.duration for s in self._scripts.values()) + 3.0
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
            print(f"[blade-demo] t={t:5.2f} left={labels[0]} right={labels[1]} "
                  f"ik_worst_mm={max(s.worst_mm for s in self._scripts.values()):.1f}", flush=True)
            self._printed = tuple(labels)
        meta = {"request_stop": True} if t > self._end + self.hold_s else {}
        return ActionChunk(actions=[Action(data=pack(*out), meta=meta)], control_hz=1.0 / self._dt)
