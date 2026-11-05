"""Policy ``mechanical_puller_scripted``: privileged scripted demonstration (reads simulator part poses).

Draft: the right gripper takes the puller body from the tool rest with a low side grip (A4) and holds it over
the shaft with the hooks in the gap under the collar rim; the left gripper squeezes both jaws closed, the right
lets go, and the left turns the knob clockwise in regrasp strokes (A4/A5). Motion segments copied from our A5
task (f7e6030). Outputs ordinary 14-D joint targets through the guide's IK; never sets object poses.
``-P stage=idle`` keeps both arms at READY (passive stability check).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena import ik, station
from scipy.spatial.transform import Rotation
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy

from . import design as d

TILT = math.radians(20)  # jaw-closing side grip (reach at the lower jaw arms)
HOLD_TILT = math.radians(60)  # handle-end hold from the robot side
BOSS_SQUEEZE = -0.012  # 30 mm end block, pinched across y
BAR_OPEN = FINGER_OPEN  # front pinch on the end block opens fully
JAW_CLOSE = -0.027  # squeeze both jaw arms to 56 mm outside
KNOB_SQUEEZE = -0.012
YAW_START, YAW_END = math.radians(40), math.radians(-90)  # clockwise from above: tightening
STROKES = 9


def front_grip(tilt):
    """Approach from the robot side (+x), tilted down by ``tilt``; fingers close along world y on the boss. The tool
    is pinned across y by this hold and across x by the jaws (y drift: draft-003; thin-bar pinch slipped: draft-005)."""
    z = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
    x = np.array([0.0, -1.0, 0.0])  # equivalent pinch, avoids joint6=-120deg limit
    return np.c_[x, np.cross(z, x), z]


def bar_grip(tilt, from_side):
    """Side approach as A4's, but the fingers close vertically (top and bottom of a horizontal bar): the held part
    cannot pivot about a horizontal pad axis (the boss pinch along x let the tool tilt and slide 8 mm, draft-003)."""
    z = np.array([0.0, -from_side * math.cos(tilt), -math.sin(tilt)])
    x = np.array([0.0, -from_side * math.sin(tilt), math.cos(tilt)])
    return np.c_[x, np.cross(z, x), z]


def side_grip(tilt, from_side):
    """A4 holding grip: approach horizontally from -y (from_side=-1) or +y (+1), tilted down by tilt."""
    z = np.array([0.0, -from_side * math.cos(tilt), -math.sin(tilt)])
    x = np.array([-1.0, 0.0, 0.0])  # symmetric pinch avoids left wrist roll limit
    return np.c_[x, np.cross(z, x), z]


@dataclass
class Seg:
    duration: float
    target: Callable  # sim -> (pos, rot)
    finger: float
    label: str
    freeze: bool = True  # evaluate the target once, on entering the segment
    servo: object = None  # sim -> goal xyz of the body; the measured error is integrated into the TCP (privileged)


class Script:
    def __init__(self, segs, base):
        self.segs, self.base = segs, np.asarray(base, float)
        self.q = np.array(READY, float)
        self.prev = ik.tcp_pose(self.q, self.base)
        self.prev_finger = FINGER_OPEN
        self.index, self.goal = -1, None
        self.worst_mm = 0.0
        self.worst_deg = 0.0

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
        if s is not None and s.servo is not None:
            body=sim.poses()["body"]
            err=np.asarray(s.servo(sim),float)-body[:3]
            self.correction=np.clip(getattr(self,"correction",np.zeros(3))+.10*err,-.025,.025)
            delta=-.08*Rotation.from_quat(body[3:]).as_rotvec()
            delta*=min(1.,math.radians(1.)/max(np.linalg.norm(delta),1e-9))
            correction=Rotation.from_rotvec(delta).as_matrix()
            accumulated=Rotation.from_matrix(correction@getattr(self,"rot_correction",np.eye(3))).as_rotvec()
            accumulated*=min(1.,math.radians(20.)/max(np.linalg.norm(accumulated),1e-9))
            self.rot_correction=Rotation.from_rotvec(accumulated).as_matrix()
        pos=np.asarray(pos,float)+getattr(self,"correction",np.zeros(3))
        rot=getattr(self,"rot_correction",np.eye(3))@rot
        q, pe, ae = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q = q
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(ae))
        return np.r_[q, float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))], label


def _placed(local, rot, at, script_key="right", follow_z=False):
    """Measured rigid grasp transform; correction for arm compliance is separate."""
    def f(sim):
        from yam_arena.inspect_embodiment import ARMS
        script=sim._policy_scripts[script_key]
        b=sim.poses()["body"]
        if not hasattr(script,"local_grasp"):
            arm=ARMS[0 if script_key=="left" else 1]
            q=sim._base.scene[arm].data.joint_pos.torch[0][sim._ids[arm][0]].cpu().numpy()
            hp,hr=ik.tcp_pose(q,station.BASES[script_key]);rb=_rot(b)
            script.local_grasp=(rb.T@(hp-b[:3]),rb.T@hr)
        offset,grasp_rot=script.local_grasp
        goal=np.array([at[0],at[1],b[2] if follow_z else d.TOP+at[2]])
        return goal+offset,grasp_rot
    return f

def _body_point(local, rot, back=0.0, at=None, lift=0.0):
    """TCP at ``local`` in the live body frame (or the body placed at ``at`` = (x, y, body height))."""
    def f(sim):
        if at is None:
            b = sim.poses()["body"]
            origin, r = b[:3], _rot(b)
        else:
            origin, r = np.array([at[0], at[1], d.TOP + at[2]]), np.eye(3)
        return origin + r @ np.asarray(local, float) + np.array([0, 0, lift]) - back * rot[:, 2], rot
    return f


def _rot(pose):
    x, y, z, w = pose[3:]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


BOSS = np.array([0.0, -d.BOSS[1] / 2 - d.HANDLE[1] - d.GRIP_BLOCK[1] / 2 + 0.005, d.GRIP_Z])  # end block, 13 cm off the axis
JAWS = np.array([0.0, 0.0, -0.035])  # lower jaw arms, 47 mm below the hold (at -20 mm the hands came within 29 mm, draft-008)
KNOB = np.array([0.0, 0.0, d.KNOB_Z + 0.004])


def plans():
    hold = front_grip(HOLD_TILT)  # fingers across y on the end block
    close = side_grip(TILT, +1)
    fx, fy = d.FIX
    above = (fx, fy, d.FIT_Z + 0.035)
    fitted = (fx, fy, d.FIT_Z)
    right = [Seg(2.5, _body_point(BOSS, hold, back=0.09), BAR_OPEN, "body approach"),
             Seg(1.5, _body_point(BOSS, hold, back=.013), BAR_OPEN, "body reach"),
             Seg(0.6, _body_point(BOSS, hold, back=.013), BOSS_SQUEEZE, "body grasp"),
             Seg(1.0, _body_point(BOSS, hold, back=.013, lift=0.04), BOSS_SQUEEZE, "body lift"),
             Seg(2.5, _placed(BOSS, hold, above), BOSS_SQUEEZE, "carry over the shaft"),
             Seg(2.,None,BOSS_SQUEEZE,"level supported tool",servo=lambda sim:np.r_[d.FIX,d.TOP+above[2]]),
             Seg(1.5, _placed(BOSS, hold, fitted), BOSS_SQUEEZE, "lower onto the collar"),
             Seg(1.5,None,BOSS_SQUEEZE,"align seated tool",servo=lambda sim:np.r_[d.FIX,d.TOP+d.FIT_Z])]
    t_fit = sum(s.duration for s in right)
    jaws = [Seg(max(0.1, t_fit - 3.0), None, FINGER_OPEN, "wait"),
            Seg(2.0, _body_point(JAWS, close, back=0.08), FINGER_OPEN, "jaws approach"),  # on the measured body
            Seg(1.0, _body_point(JAWS, close), FINGER_OPEN, "jaws reach"),
            Seg(1.2, _body_point(JAWS, close), JAW_CLOSE, "close the jaws"),
            Seg(0.5, None, JAW_CLOSE, "hold the jaws"),
            Seg(0.5, None, FINGER_OPEN, "jaws let go"),
            Seg(0.8, _body_point(JAWS, close, back=0.08), FINGER_OPEN, "jaws back off")]
    t_set = sum(s.duration for s in jaws)
    # The right gripper keeps holding the body while the screw is turned: the turning reaction otherwise spins the
    # whole puller on the collar (draft-002: 59 deg over two strokes). It follows the body up as the collar comes off.
    hold_body = Seg(0.1, _placed(BOSS, hold, fitted, follow_z=True), BOSS_SQUEEZE, "counterhold", freeze=False,
                    servo=lambda sim: np.r_[d.FIX,sim.poses()["body"][2]])
    right_tail = (hold_body, t_set - t_fit)
    knob = lambda yaw, dz=0.0: _body_point(KNOB, grip_rotation(yaw), lift=dz)  # noqa: E731  (live body)
    left = jaws + [Seg(1.5, knob(YAW_START, 0.05), FINGER_OPEN, "knob approach")]
    for k in range(STROKES):
        left += [Seg(0.8, knob(YAW_START), FINGER_OPEN, f"stroke {k + 1} descend"),
                 Seg(0.5, knob(YAW_START), KNOB_SQUEEZE, f"stroke {k + 1} grip"),
                 Seg(1.5, knob(YAW_END), KNOB_SQUEEZE, f"stroke {k + 1} turn"),
                 Seg(0.4, None, FINGER_OPEN, f"stroke {k + 1} open"),
                 Seg(0.3, knob(YAW_END, 0.02), FINGER_OPEN, f"stroke {k + 1} clear"),
                 Seg(0.8, knob(YAW_START, 0.02), FINGER_OPEN, f"stroke {k + 1} return")]
    left += [Seg(1.0, knob(YAW_START, 0.04), FINGER_OPEN, "knob retreat")]
    hold_body.duration = max(0.1, sum(s.duration for s in left) - t_fit)
    right += [hold_body]  # keeps holding: releasing before the tool is set down dropped it (draft-008)
    return {"left": left, "right": right}


def prepared_plans(supported=True):
    hold=front_grip(HOLD_TILT)
    right=[Seg(2.5,_body_point(BOSS,hold,back=.09),BAR_OPEN,"support approach"),
           Seg(1.5,_body_point(BOSS,hold,back=.013),BAR_OPEN,"support reach"),
           Seg(.8,_body_point(BOSS,hold,back=.013),BOSS_SQUEEZE,"support grasp")]
    knob=lambda yaw,dz=0.:_body_point(KNOB,grip_rotation(yaw),lift=dz)
    left=[Seg(5.,None,FINGER_OPEN,"wait for support"),
          Seg(1.5,knob(YAW_START,.05),FINGER_OPEN,"knob approach")]
    for k in range(8):
        left += [Seg(.8,knob(YAW_START),FINGER_OPEN,f"stroke {k+1} descend"),
                 Seg(.5,knob(YAW_START),KNOB_SQUEEZE,f"stroke {k+1} grip"),
                 Seg(1.5,knob(YAW_END),KNOB_SQUEEZE,f"stroke {k+1} turn"),
                 Seg(.4,None,FINGER_OPEN,f"stroke {k+1} open"),
                 Seg(.3,knob(YAW_END,.02),FINGER_OPEN,f"stroke {k+1} clear"),
                 Seg(.8,knob(YAW_START,.02),FINGER_OPEN,f"stroke {k+1} return")]
    left += [Seg(1.,knob(YAW_START,.04),FINGER_OPEN,"knob retreat")]
    right += [Seg(sum(v.duration for v in left)-sum(v.duration for v in right),
                  _placed(BOSS,hold,(*d.FIX,d.FIT_Z),follow_z=True),BOSS_SQUEEZE,
                  "supported screw extraction",freeze=False,
                  servo=lambda sim:np.r_[d.FIX,sim.poses()["body"][2]])]
    return {"left":left,**({"right":right} if supported else {})}


def complete_plans():
    """Two-hand carry to a cradle, then physical jaw spreading over its tray."""
    p=prepared_plans(True)
    hold=front_grip(HOLD_TILT);close=side_grip(TILT,+1)
    fx,fy=d.FIX;tx,ty=d.TRAY_XY
    # Regrasp knob after extraction so the side grip does not carry the lever alone.
    p["left"] += [Seg(1.,_body_point(KNOB,grip_rotation(YAW_START)),FINGER_OPEN,"regrasp transport knob"),
                  Seg(.6,None,KNOB_SQUEEZE,"grasp transport knob")]
    p["right"] += [Seg(1.6,None,BOSS_SQUEEZE,"wait for two-hand carry")]
    for x,y,z,duration,label in [(fx,fy,.125,2.5,"lift clear together"),
                                (tx+.002,ty+.005,.125,3.5,"carry together over cradle"),
                                (tx+.002,ty+.005,d.PARK_Z,2.5,"lower onto cradle"),
                                (tx+.002,ty+.005,d.PARK_Z,1.5,"settle on cradle")]:
        for side,local,rot,finger in [("right",BOSS,hold,BOSS_SQUEEZE),("left",KNOB,grip_rotation(YAW_START),KNOB_SQUEEZE)]:
            p[side].append(Seg(duration,_placed(local,rot,(x,y,z),script_key=side),finger,label))
    p["left"] += [Seg(.6,None,FINGER_OPEN,"release transport knob"),
                  Seg(1.,_body_point(KNOB,grip_rotation(YAW_START),lift=.05),FINGER_OPEN,"clear knob")]
    jawpoint=np.array([0.,0.,-.035])
    p["left"] += [Seg(1.5,_body_point(jawpoint,close,back=.08),-.010,"approach inside jaws"),
                   Seg(1.2,_body_point(jawpoint,close),-.010,"insert fingers between jaws"),
                   Seg(1.2,None,FINGER_OPEN,"spread jaws to release collar"),
                   Seg(.6,None,FINGER_OPEN,"let collar settle"),
                   Seg(2.,_body_point(jawpoint,close,back=.10),FINGER_OPEN,"withdraw from jaws")]
    p["right"] += [Seg(8.1,None,BOSS_SQUEEZE,"steady resting tool"),
                   Seg(.8,None,FINGER_OPEN,"release parked tool"),
                   Seg(2.,_body_point(BOSS,hold,back=.10),FINGER_OPEN,"withdraw holder"),
                   Seg(2.5,None,FINGER_OPEN,"verify stable release")]
    return p


class PullerScriptedPolicy(YamScriptedPolicy):
    def __init__(self, stage="complete", duration=8.0, **kwargs):
        super().__init__(**kwargs)
        self.stage, self.duration = str(stage), float(duration)

    def reset(self, scene):
        p = complete_plans() if self.stage=="complete" else {} if self.stage in ("idle", "probe") else (prepared_plans(self.stage=="preparedhold") if self.stage in ("preparedhold","preparednohold") else plans())
        if self.stage=="fit":
            p["left"]=p["left"][:7]+[Seg(2.,None,FINGER_OPEN,"observe engagement")]
            p["right"][-1].duration=sum(v.duration for v in p["left"])-sum(v.duration for v in p["right"][:-1])
        if self.stage=="pickup":
            p={"right":p["right"][:4]+[Seg(2.,None,BOSS_SQUEEZE,"hold lifted tool")]}
        self._scripts = {a: Script(s, self._bases[a]) for a, s in p.items()}
        for s in self._scripts.values():
            s.grip_dz = 0.0
        from . import embodiment
        if embodiment.ACTIVE is not None:
            embodiment.ACTIVE._policy_scripts = self._scripts
        self._end = max([s.duration for s in self._scripts.values()] or [self.duration]) + 2.0
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
        if self._scripts and tuple(labels) != self._printed:
            print(f"[puller-demo] t={t:5.2f} left={labels[0]} right={labels[1]} "
                  f"ik_worst_mm={max(s.worst_mm for s in self._scripts.values()):.1f} ik_worst_deg={max(s.worst_deg for s in self._scripts.values()):.1f}", flush=True)
            self._printed = tuple(labels)
        meta = {"request_stop": True} if t > self._end else {}
        return ActionChunk(actions=[Action(data=pack(*out), meta=meta)], control_hz=1 / self._dt)
