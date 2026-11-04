"""Policy ``tipper_scripted``: a privileged feasibility demonstration, not a model of manual reading.

It reads simulator part poses (declared privileged input) to place its Cartesian targets and
outputs ordinary 14-D joint targets through the guide's IK, like ``yam_scripted``. It never sets
object poses. ``stage`` selects the plan:

- ``bed``: the diagnostic of the critical interaction on a preassembled base (run the embodiment with
  ``-E preassembled=brick,cab,hinge``): seat the bed on the hinge, load the payload, tip it out.
- ``full``: the whole build from loose parts (see ``full_plan``).

``-P counterhold=false`` leaves the right arm at its ready pose (the one-arm comparison).
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

TRAVEL = d.TOP + 0.13
BED_GRASP = np.array([0.030, -0.030, 0.028])  # TCP in the bed frame: pads on the front wall and the rear face (y = 4 mm)
# of the right side wall; lower grasps put the open front finger inside the cab (bed-022 lifted the truck 21 mm)
# (fingers closing along the bed's length). A pinch across the width pivots the bed about the hinge-axis
# direction between the pads; it cocked 35 deg on the truck while lowering (bed-012).
TIP_GRASP = np.array([0.0, -0.048, 0.031])  # nearer the front for leverage
CAB_GRASP = np.array([0.0, -0.046, 0.030])  # TCP in the cab frame: side grip from the front, pads on the cabin sides 18 mm behind its front
PAYLOAD_IN_BED = np.array([0.0, -0.024, 0.0076 + d.PAYLOAD_SIZE / 2 + 0.003])
TIP_ANGLE = math.radians(55)
HOLD_TILT = math.radians(30)  # counterhold side grip: approach from -y, 30 degrees below horizontal (A4)


def side_grip(tilt):
    """A4's holding grip: tool z (approach) toward +y and down by ``tilt``, fingers closing along world x."""
    z = np.array([0.0, math.cos(tilt), -math.sin(tilt)])
    x = np.array([1.0, 0.0, 0.0])
    return np.c_[x, np.cross(z, x), z]
BED_CLEAR = 0.075  # bed lift above its closed pose while carried: its floor clears the cabin roof
PRESS_POINT = np.array([0.0, -0.0075, 0.0076])  # bed floor top just in front of the clip, over the pin
PRESS_ABOVE = 0.012  # TCP above that surface when the closed fingertips touch it (approximate)
PRESS_PAST_SNAP = 0.0080  # TCP above the closed floor at the press; 11.0 mm gave 0.85 N (bed-018)
SEAT_TIP = math.radians(10)  # bed onto the pin tipped open (manual step 3): front 2 mm clear of the headboard, rear lip
# 1.4 mm above the hinge ears (at 40 deg the lip hit the ears and the clip stopped 6.5 mm short: bed-016)
WIDE = -0.0295  # finger target closing on the 64 mm bed / cab width
LONG = -0.0330  # finger target closing on the 72 mm bed length
BED_LET_GO = -0.040  # 80 mm: clears the bed without swinging the front finger into the cab
SMALL = -0.006  # finger target closing on the 20 mm payload
INBED_OPEN = -0.020  # 40 mm opening: releases the payload without touching the bed's side walls


@dataclass
class Seg:
    duration: float
    target: Callable  # (sim, s) -> (pos, rot) ; s in [0, 1] for arcs, ignored for moves
    finger: float
    label: str
    arc: bool = False  # target is a path in s rather than an end point
    servo: str | None = None  # part whose measured pose error is fed back (privileged pose servo)
    servo_local: object = None  # that part's goal: callable(sim) -> world position of its frame
    freeze: bool = False  # evaluate the target once, on entering the segment (a fixed hold pose)


def _xform(pose, local):
    r = d.quat_to_matrix(pose[3:])
    return pose[:3] + r @ local, r


class ArmScript:
    def __init__(self, segs, base, yaw):
        self.segs, self.base = segs, np.asarray(base, float)
        self.q = np.array(READY, float)
        self.start = ik.tcp_pose(self.q, self.base)
        self.start_finger = FINGER_OPEN
        self.index, self.t0 = -1, 0.0
        self.yaw = yaw
        self.worst_mm = 0.0
        self.correction = np.zeros(3)

    @property
    def duration(self):
        return sum(s.duration for s in self.segs)

    def label(self, t):
        acc = 0.0
        for s in self.segs:
            if t < acc + s.duration:
                return s.label
            acc += s.duration
        return "done"

    def act(self, sim, t, q_now):
        acc, idx = 0.0, len(self.segs)
        for i, s in enumerate(self.segs):
            if t < acc + s.duration:
                idx = i
                break
            acc += s.duration
        if idx != self.index:  # entering a segment: it starts from the previous command
            if self.index >= 0 and self.index < len(self.segs):
                self.start, self.start_finger = self._last, self._last_finger
            self.index = idx
            self.frozen = None
        if idx >= len(self.segs):
            pos, rot, finger = (*self._last, self._last_finger) if hasattr(self, "_last") else (*self.start, FINGER_OPEN)
        else:
            seg = self.segs[idx]
            s = ik.smoothstep((t - acc) / seg.duration)
            if seg.arc:
                pos, rot = seg.target(sim, s)
            else:
                if seg.freeze:
                    if self.frozen is None:
                        self.frozen = self._last if hasattr(self, "_last") else self.start
                    goal, goal_rot = self.frozen
                else:
                    goal, goal_rot = seg.target(sim, 1.0)
                pos = self.start[0] + s * (goal - self.start[0])
                rot = ik.slerp_matrix(self.start[1], goal_rot, s)
            finger = self.start_finger + s * (seg.finger - self.start_finger)
        if idx < len(self.segs) and self.segs[idx].servo:
            seg = self.segs[idx]
            actual = sim.poses()[seg.servo][:3]
            goal = seg.servo_local(sim)
            self.correction = np.clip(self.correction + 0.4 * (goal - actual), -0.008, 0.008)
            self.correction[2] = 0.0  # height follows the plan; contact sets the seat
        elif idx < len(self.segs) and self.segs[idx].label.endswith(("release", "retreat")):
            pass  # keep the correction while letting go
        else:
            self.correction = np.zeros(3)
        pos = np.asarray(pos, float) + self.correction
        self._last, self._last_finger = (np.asarray(pos, float) - self.correction, rot), finger
        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q = q
        self.worst_mm = max(self.worst_mm, pe * 1000)
        return np.r_[q, float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))]


class TipperScriptedPolicy(YamScriptedPolicy):
    def __init__(self, stage="bed", counterhold=True, tip_deg=60.0, idle_s=6.0, **kwargs):
        super().__init__(**kwargs)
        self.stage = str(stage)
        self.counterhold = str(counterhold).lower() not in ("false", "0", "no")
        self.tip = math.radians(float(tip_deg))
        self.idle_s = float(idle_s)

    def reset(self, scene):
        if self.stage == "idle":
            plans = {"left": [Seg(self.idle_s, lambda sim, s: ik.tcp_pose(np.array(READY, float), np.array(self._bases["left"])), FINGER_OPEN, "idle")]}
        elif self.stage == "hold":  # diagnostic: the right gripper's cab counterhold alone, left arm idle
            plans = {"right": counterhold_segments(side_grip(HOLD_TILT), 1.0, self.idle_s),
                     "left": [Seg(1.0, lambda sim, s: ik.tcp_pose(np.array(READY, float), np.array(self._bases["left"])), FINGER_OPEN, "idle")]}
        elif self.stage == "bed":
            plans = bed_plan(self.counterhold, self.tip)
        elif self.stage == "full":
            plans = full_plan(self.counterhold, self.tip)
        else:
            raise ValueError(f"stage must be 'bed' or 'full', got {self.stage!r}")
        self._scripts = {arm: ArmScript(segs, self._bases[arm], 0.0) for arm, segs in plans.items()}
        self._end = max(s.duration for s in self._scripts.values())
        self._printed = None

    def act(self, observation):
        from . import embodiment

        sim = embodiment.ACTIVE
        t = float(observation.state_time) + self._dt
        per_arm = [self._scripts[a].act(sim, t, None) if a in self._scripts else np.r_[READY, 1.0]
                   for a in ("left", "right")]
        labels = tuple(self._scripts[a].label(t) if a in self._scripts else "-" for a in ("left", "right"))
        if labels != self._printed:
            print(f"[tipper-demo] t={t:5.2f} left={labels[0]} right={labels[1]} "
                  f"ik_worst_mm={max(s.worst_mm for s in self._scripts.values()):.1f}", flush=True)
            self._printed = labels
        meta = {"request_stop": True} if t > self._end + self.hold_s else {}
        if labels[0] == "tip approach" and not sim._joined["bed"]:
            print("[tipper-demo] bed is not on its hinge; ending the demonstration before tipping", flush=True)
            meta = {"request_stop": True}
        return ActionChunk(actions=[Action(data=pack(*per_arm), meta=meta)], control_hz=1.0 / self._dt)


# -- plans -----------------------------------------------------------------------------------
def _fixed(pos, rot):
    pos = np.asarray(pos, float)
    return lambda sim, s: (pos, rot)


def _live(name, local, rot, lift=0.0):
    """TCP at ``local`` in part ``name``'s live frame, raised by ``lift``."""
    def f(sim, s):
        p, _ = _xform(sim.poses()[name], np.asarray(local, float))
        return p + np.array([0, 0, lift]), rot
    return f


def _live_back(name, local, rot, back):
    """TCP at ``local`` in part ``name``'s live frame, backed off ``back`` along the approach axis."""
    def f(sim, s):
        p, _ = _xform(sim.poses()[name], np.asarray(local, float))
        return p - back * rot[:, 2], rot
    return f


def _closed_bed(sim):
    """Where the bed frame is when seated and closed, from the live hinge plate pose."""
    return d.world_pose(sim.poses()["hinge"], d.BED_IN_HINGE)


def _at_closed_bed(local, rot, lift=0.0):
    def f(sim, s):
        p, _ = _xform(_closed_bed(sim), np.asarray(local, float))
        return p + np.array([0, 0, lift]), rot
    return f


def _tip_arc(local, rot, a0, a1):
    """Grasp point on the bed swept about the live hinge axis from angle a0 to a1."""
    def f(sim, s):
        frame = _closed_bed(sim)
        rc = d.quat_to_matrix(frame[3:])
        angle = a0 + s * (a1 - a0)
        turn = rc @ d.rot_x(-angle) @ rc.T
        return frame[:3] + turn @ (rc @ np.asarray(local, float)), turn @ rot
    return f


def _tipped(local, rot, angle, lift=0.0):
    """Grasp point with the bed opened by ``angle`` on the hinge, offset ``lift`` along the tipped bed's z."""
    def f(sim, s):
        frame = _closed_bed(sim)
        rc = d.quat_to_matrix(frame[3:])
        turn = rc @ d.rot_x(-angle) @ rc.T
        return frame[:3] + turn @ (rc @ (np.asarray(local, float) + np.array([0, 0, lift]))), turn @ rot
    return f


def _tcp_point(sim, name, local, rot):
    p, _ = _xform(sim.poses()[name], np.asarray(local, float))
    return p, rot


def bed_seat_segments(rot_x_grip):
    g = rot_x_grip
    return [
        Seg(1.5, _live("bed", BED_GRASP, g, lift=0.06), FINGER_OPEN, "bed approach"),
        Seg(1.2, _live("bed", BED_GRASP, g), FINGER_OPEN, "bed descend"),
        Seg(0.7, _live("bed", BED_GRASP, g), LONG, "bed grasp"),
        Seg(1.2, lambda sim, s: (np.r_[_tcp_point(sim, "bed", BED_GRASP, g)[0][:2], TRAVEL], g), LONG, "bed lift"),
        # Tipped open by SEAT_TIP above the hinge, down onto the pin along the tipped bed's own axis; let go
        # (the bed swings closed on the pin under its own weight), then push the clip home with the closed
        # fingertips on the floor right above the pin. A pinch slips at about 1 N on the 26 g bed (bed-014).
        Seg(2.0, _tipped(BED_GRASP, g, SEAT_TIP, lift=0.045), LONG, "bed carry"),
        Seg(2.0, _tipped(BED_GRASP, g, SEAT_TIP, lift=0.002), LONG, "bed lower",
            servo="bed", servo_local=lambda sim: _closed_bed(sim)[:3]),
        Seg(0.6, _tipped(BED_GRASP, g, SEAT_TIP, lift=0.002), BED_LET_GO, "bed let go"),
        Seg(0.8, _tipped(BED_GRASP, g, SEAT_TIP, lift=0.040), BED_LET_GO, "bed clear"),
        Seg(1.2, _at_closed_bed(PRESS_POINT, g, lift=PRESS_ABOVE + 0.015), 0.0, "press approach"),
        # Snap observed with the TCP 13.4 mm above the floor (bed-015); pressing 5 mm deeper sank the light bed
        # 1.9 mm into the pin's contact. Press to 1 mm past the snap point and hold.
        Seg(2.0, _at_closed_bed(PRESS_POINT, g, lift=PRESS_PAST_SNAP), 0.0, "bed press"),
        Seg(1.0, _at_closed_bed(PRESS_POINT, g, lift=PRESS_PAST_SNAP), 0.0, "press hold"),
        Seg(1.0, _at_closed_bed(PRESS_POINT, g, lift=0.07), FINGER_OPEN, "bed retreat"),
    ]


def payload_segments(rot_y_grip):
    above = lambda lift: _at_closed_bed(PAYLOAD_IN_BED + [0, 0, 0.008], rot_y_grip, lift=lift)  # noqa: E731
    return [
        Seg(1.5, _live("payload", (0, 0, 0.008), rot_y_grip, lift=0.06), FINGER_OPEN, "payload approach"),
        Seg(1.0, _live("payload", (0, 0, 0.008), rot_y_grip), FINGER_OPEN, "payload descend"),
        Seg(0.6, _live("payload", (0, 0, 0.008), rot_y_grip), SMALL, "payload grasp"),
        Seg(1.0, lambda sim, s: (np.r_[_tcp_point(sim, "payload", (0, 0, 0.008), rot_y_grip)[0][:2], TRAVEL],
                                 rot_y_grip), SMALL, "payload lift"),
        Seg(1.6, above(0.05), SMALL, "payload carry"),
        # Set it down 4 mm above the floor and let go: the block rides lower in the pads than nominal, and
        # pushing it 3 mm into the floor broke the bed's hinge connection (bed-019).
        Seg(1.0, above(0.004), SMALL, "payload lower"),
        # Open only to 40 mm inside the bed (56 mm inner width): a full opening pushed the side walls and
        # broke the hinge connection (bed-020). Open fully once clear.
        Seg(0.5, above(0.004), INBED_OPEN, "payload release"),
        Seg(0.8, above(0.06), INBED_OPEN, "payload retreat"),
        Seg(0.3, above(0.06), FINGER_OPEN, "payload open"),
    ]


def tip_segments(rot_x_grip, tip):
    """Tip the bed on its hinge, hold it while the load slides out of the open rear, then let go and move up
    out of its swing: the bed settles back on the hinge by itself (closing it is not part of the task)."""
    return [
        Seg(1.5, _at_closed_bed(TIP_GRASP, rot_x_grip, lift=0.05), FINGER_OPEN, "tip approach"),
        Seg(1.0, _at_closed_bed(TIP_GRASP, rot_x_grip), FINGER_OPEN, "tip descend"),
        Seg(0.6, _at_closed_bed(TIP_GRASP, rot_x_grip), WIDE, "tip grasp"),
        Seg(3.0, _tip_arc(TIP_GRASP, rot_x_grip, 0.0, tip), WIDE, "tip open", arc=True),
        Seg(2.5, _tip_arc(TIP_GRASP, rot_x_grip, tip, tip), WIDE, "tip hold", arc=True),
        Seg(0.5, _tip_arc(TIP_GRASP, rot_x_grip, tip, tip), FINGER_OPEN, "tip release", arc=True),
        Seg(1.0, _lift_from_tip(TIP_GRASP, rot_x_grip, tip, 0.10), FINGER_OPEN, "tip retreat"),
    ]


def _lift_from_tip(local, rot, angle, up):
    """Straight up from the tipped grasp point (clear of the bed's swing back down)."""
    def f(sim, s):
        pos, r = _tip_arc(local, rot, angle, angle)(sim, 1.0)
        return pos + np.array([0.0, 0.0, up]), r
    return f


def counterhold_segments(rot_x_grip, wait, hold):
    return [
        Seg(wait, lambda sim, s: ik.tcp_pose(np.array(READY, float), np.array(d_bases()["right"]))[0:2], 1.0 * FINGER_OPEN,
            "wait"),
        Seg(2.0, _live_back("cab", CAB_GRASP, rot_x_grip, 0.09), FINGER_OPEN, "cab approach"),
        Seg(1.5, _live("cab", CAB_GRASP, rot_x_grip), FINGER_OPEN, "cab reach"),
        Seg(0.6, None, WIDE, "cab grasp", freeze=True),  # hold still where the cab is: no tracking
        Seg(hold, None, WIDE, "counterhold", freeze=True),
        Seg(0.6, None, FINGER_OPEN, "cab release", freeze=True),
        Seg(1.0, _live_back("cab", CAB_GRASP, rot_x_grip, 0.10), FINGER_OPEN, "cab retreat"),
    ]


def d_bases():
    from yam_arena import station
    return station.BASES


def _dur(segs):
    return sum(s.duration for s in segs)


def bed_plan(counterhold, tip):
    # A1/r3 layout (truck along y): the left gripper works top-down with its fingers closing across the
    # truck (world x), its wrist camera toward its own side (+y); the right gripper counterholds the cab
    # front with A4's low side grip from -y. The wrists work from opposite sides of the truck.
    left_x = grip_rotation(math.pi / 2)
    right_x = side_grip(HOLD_TILT)
    left_y = grip_rotation(math.pi / 2)
    left_len = grip_rotation(0.0)  # fingers along y: the bed's length
    seat, load, tipping = bed_seat_segments(left_len), payload_segments(left_y), tip_segments(left_x, tip)
    left = seat + load + tipping + [Seg(4.0, None, FINGER_OPEN, "left wait", freeze=True)]
    plans = {"left": left}
    if counterhold:
        # The right gripper holds the cab from before the bed is lowered onto the hinge until the
        # left gripper has let go of the tipped-and-closed bed.
        grasped_by = _dur(seat[:5]) - 0.3
        released_after = _dur(seat + load + tipping) + 1.0  # after the bed has settled
        reach = 2.0 + 1.5 + 0.6
        plans["right"] = counterhold_segments(right_x, grasped_by - reach, released_after - grasped_by)
    return plans


# -- full build from loose parts ------------------------------------------------------------
BRICK_GRASP = np.array([0.0, 0.0, 0.003])  # TCP in the brick frame (body centre): pads on two side faces
HINGE_GRASP = np.array([0.0, 0.020, 0.002])  # TCP in the plate frame: lengthwise pinch, front edge to the ears' backs (72 mm)
# through its centre of mass; a pinch across the ears 36 mm off the mass centre flipped the plate (full-003)
CAB_PRESS = np.array([0.0, 0.030, d.CAB_DECK_Z])  # top of the cab's rear deck (before the hinge plate goes on)
BRICK_TOP = np.array([0.0, 0.0, d.BRICK_H / 2])
PLATE_TOP = np.array([0.0, 0.0, 0.0])
PART_SQUEEZE = -0.012  # finger target closing on the 32 mm brick / plate
LET_GO = -0.025  # 50 mm opening to let go of a placed part


def _goal(parent, offset):
    """World pose of a child frame at ``offset`` in the live parent frame (where the manual puts it)."""
    return lambda sim: d.world_pose(sim.poses()[parent], np.asarray(offset, float))


def _at_goal(goal, local, rot, lift=0.0, back=0.0):
    def f(sim, s):
        p, _ = _xform(goal(sim), np.asarray(local, float))
        return p + np.array([0, 0, lift]) - back * rot[:, 2], rot
    return f


def _lift_here(name, local, rot, height):
    """Straight up from the part's live grasp point to an absolute height."""
    def f(sim, s):
        p, _ = _xform(sim.poses()[name], np.asarray(local, float))
        return np.r_[p[:2], height], rot
    return f


def _wait(duration, label="wait"):
    return Seg(max(0.1, duration), None, FINGER_OPEN, label, freeze=True)


def place_segments(name, goal, grasp, rot, squeeze, top, travel=TRAVEL):
    """Pick a loose part top-down, carry it above its place, lower it there (servoing the part's measured
    position onto the goal), let go and snap it home with a fingertip press on its top."""
    return [
        Seg(1.8, _live(name, grasp, rot, lift=0.06), FINGER_OPEN, f"{name} approach"),
        Seg(1.2, _live(name, grasp, rot), FINGER_OPEN, f"{name} descend"),
        Seg(0.6, _live(name, grasp, rot), squeeze, f"{name} grasp"),
        Seg(1.0, _lift_here(name, grasp, rot, travel), squeeze, f"{name} lift"),
        Seg(2.0, _at_goal(goal, grasp, rot, lift=0.04), squeeze, f"{name} carry"),
        Seg(1.8, _at_goal(goal, grasp, rot, lift=0.002), squeeze, f"{name} lower",
            servo=name, servo_local=lambda sim: goal(sim)[:3]),
        Seg(0.5, _at_goal(goal, grasp, rot, lift=0.002), LET_GO, f"{name} let go"),
        Seg(0.6, _at_goal(goal, grasp, rot, lift=0.03), LET_GO, f"{name} clear"),
        Seg(1.0, _at_goal(goal, top, rot, lift=PRESS_ABOVE + 0.015), 0.0, f"{name} press approach"),
        Seg(2.0, _at_goal(goal, top, rot, lift=PRESS_PAST_SNAP), 0.0, f"{name} press"),
        Seg(1.0, _at_goal(goal, top, rot, lift=PRESS_PAST_SNAP), 0.0, f"{name} press hold"),
        Seg(0.8, _at_goal(goal, top, rot, lift=0.07), FINGER_OPEN, f"{name} retreat"),
    ]


def full_plan(counterhold, tip):
    """The whole build from the loose parts, then the load-and-dump check (steps of manual.txt).

    Right gripper: step 1 brick onto the base; step 2 carries the cab over the base with the low side grip
    and keeps holding it (the counterhold) through steps 3-5. Left gripper: presses the cab down onto the
    base, places and presses the hinge plate, then the bed, payload and tipping sequences of the diagnostic.
    """
    down = grip_rotation(math.pi / 2)  # fingers closing across the truck (world x)
    side = side_grip(HOLD_TILT)
    brick_goal = _goal("chassis", d.BRICK_IN_CHASSIS)
    cab_goal = _goal("chassis", d.CAB_IN_CHASSIS)
    hinge_goal = _goal("cab", d.HINGE_IN_CAB)

    right = place_segments("brick", brick_goal, BRICK_GRASP, down, PART_SQUEEZE, BRICK_TOP)
    cab_carry = [
        # Around the loose cab, not through it: beside its front corner first (x clear of the cab), then in
        # along +x behind its front face to the approach pose (full-002 knocked the cab 20 cm on a straight path).
        Seg(1.5, _fixed((0.36, -0.31, 0.83), side), FINGER_OPEN, "cab via"),
        Seg(1.5, _live_back("cab", CAB_GRASP, side, 0.09), FINGER_OPEN, "cab approach"),
        Seg(1.5, _live("cab", CAB_GRASP, side), FINGER_OPEN, "cab reach"),
        Seg(0.6, _live("cab", CAB_GRASP, side), WIDE, "cab grasp"),
        Seg(1.2, _lift_here("cab", CAB_GRASP, side, d.TOP + 0.10), WIDE, "cab lift"),  # 23 mm above the brick; 0.17 was out of reach
        Seg(2.5, _at_goal(cab_goal, CAB_GRASP, side, lift=0.05), WIDE, "cab carry"),
        Seg(2.0, _at_goal(cab_goal, CAB_GRASP, side, lift=0.002), WIDE, "cab lower",
            servo="cab", servo_local=lambda sim: cab_goal(sim)[:3]),
    ]
    # Let go once the cab rests on the base (servoed to 0.4 mm): gripping it through the press pulled it 1.7 mm
    # off its seat (full-003). The right gripper takes it again afterwards as the counterhold.
    cab_carry += [Seg(0.5, None, FINGER_OPEN, "cab let go", freeze=True),
                  Seg(0.8, _at_goal(cab_goal, CAB_GRASP, side, back=0.05), FINGER_OPEN, "cab back off")]
    right += cab_carry
    cab_down = _dur(right)

    press_cab = [
        Seg(1.2, _at_goal(cab_goal, CAB_PRESS, down, lift=PRESS_ABOVE + 0.015), 0.0, "cab press approach"),
        Seg(2.0, _at_goal(cab_goal, CAB_PRESS, down, lift=PRESS_PAST_SNAP), 0.0, "cab press"),
        Seg(1.0, _at_goal(cab_goal, CAB_PRESS, down, lift=PRESS_PAST_SNAP), 0.0, "cab press hold"),
        Seg(0.8, _at_goal(cab_goal, CAB_PRESS, down, lift=0.07), FINGER_OPEN, "cab press retreat"),
    ]
    hinge = place_segments("hinge", hinge_goal, HINGE_GRASP, grip_rotation(0.0), LONG, PLATE_TOP)
    left_len = grip_rotation(0.0)
    left_y = grip_rotation(math.pi / 2)
    seat, load, tipping = bed_seat_segments(left_len), payload_segments(left_y), tip_segments(down, tip)
    left = [_wait(cab_down - 1.2 + 0.3, "wait for the cab")] + press_cab + hinge + seat + load + tipping
    left += [Seg(4.0, None, FINGER_OPEN, "left wait", freeze=True)]
    release_at = _dur(left) - 3.0  # after the bed has settled
    regrip_at = cab_down + _dur(press_cab)
    right += [Seg(max(0.1, regrip_at - _dur(right)), None, FINGER_OPEN, "wait for the cab press", freeze=True),
              Seg(1.2, _live("cab", CAB_GRASP, side), FINGER_OPEN, "cab reach"),
              Seg(0.6, None, WIDE, "cab grasp", freeze=True)]
    right += [Seg(max(0.1, release_at - _dur(right)), None, WIDE, "counterhold", freeze=True),
              Seg(0.6, None, FINGER_OPEN, "cab release", freeze=True),
              Seg(1.0, _live_back("cab", CAB_GRASP, side, 0.10), FINGER_OPEN, "cab retreat")]
    if not counterhold:
        raise ValueError("the full build uses the right gripper for the brick and cab; use stage=bed for the "
                         "no-counterhold comparison")
    return {"left": left, "right": right}
