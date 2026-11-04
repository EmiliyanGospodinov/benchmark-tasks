"""Outcome scorer for the tipper truck. NumPy only, so it can be tested without Isaac Sim.

Milestones are cumulative and describe the current state of the truck, not the
trajectory or which arm did what:

- 0.25: brick and cab both joined at their manual positions;
- 0.50: additionally the hinge plate joined on the deck, knuckles at the rear;
- 0.75: additionally the bed seated on the hinge and retained by it;
- 1.00: additionally a valid unload has happened (below), the payload rests wholly in
  the tray, the truck stands on its wheels with every link intact, and both grippers
  are open and clear; this state must hold for ``HOLD_S``.

A valid unload: the payload rested in the bed (closed within 10 degrees, payload
inside the bed and on its floor, nothing gripping it) for ``LOADED_S``; later, with
the bed still on its hinge and opened at least ``EXIT_TIP``, the payload crossed the
bed's rear opening without a gripper within ``HAND_CLEAR`` of it. Lifting the
payload out, tipping the whole truck (chassis tilt above ``TRUCK_TILT``) or a lost
bed connection invalidates that unload; the payload must then be reloaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import design as d

LOADED_S = 0.5
HOLD_S = 1.0
EXIT_TIP = math.radians(20)
HAND_CLEAR = 0.035  # m, TCP or pad tip to payload centre
PART_CLEAR = 0.045  # m, TCP to any truck part origin at the end
TRUCK_TILT = math.radians(12)
UPRIGHT_TILT = math.radians(5)
REST_SPEED = 0.01  # m/s
REST_SPIN = 0.3  # rad/s
FLOOR_TOP = 0.0076  # bed floor top in the bed frame
BED_INNER = (0.028, -0.060, 0.004)  # |x| inside the side walls, front wall, rear posts (bed frame y)
CUTOUT = 0.020  # half-width of the rear opening


def tilt(pose) -> float:
    r = d.quat_to_matrix(np.asarray(pose, float)[3:])
    return float(math.acos(np.clip(r[2, 2], -1, 1)))


def payload_in_bed(bed_pose, payload_pose) -> bool:
    offset, _ = d.relative(bed_pose, payload_pose)
    h = d.PAYLOAD_SIZE / 2
    return bool(abs(offset[0]) <= BED_INNER[0] - h + 0.002
                and BED_INNER[1] + h - 0.002 <= offset[1] <= BED_INNER[2] - h + 0.002
                and abs(offset[2] - (FLOOR_TOP + h)) <= 0.004)


def payload_in_tray(payload_pose, tray_center) -> bool:
    """Every corner of the cube inside the tray walls and its bottom on the tray floor."""
    p = np.asarray(payload_pose, float)
    r = d.quat_to_matrix(p[3:])
    h = d.PAYLOAD_SIZE / 2
    corners = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)]) @ r.T + p[:3]
    lo = np.array([tray_center[0] - d.TRAY_INNER[0] / 2, tray_center[1] - d.TRAY_INNER[1] / 2])
    hi = np.array([tray_center[0] + d.TRAY_INNER[0] / 2, tray_center[1] + d.TRAY_INNER[1] / 2])
    inside = bool((corners[:, :2] >= lo).all() and (corners[:, :2] <= hi).all())
    on_floor = abs(corners[:, 2].min() - (d.TOP + d.TRAY_FLOOR)) <= 0.003
    return inside and on_floor


@dataclass
class Observed:
    """One control step of simulator state for the scorer."""

    poses: dict  # name -> [x,y,z,qx,qy,qz,qw] for chassis, brick, cab, hinge, bed, payload
    speeds: dict  # name -> (linear m/s, angular rad/s)
    joined: dict  # brick, cab, hinge, bed -> retention joint enabled and intact
    hands: list  # world points of both TCPs and pad tips
    grippers_open: bool
    dt: float


@dataclass
class Scorer:
    tray_center: tuple = d.TRAY_CENTER
    preassembled: tuple = ()  # parts that start joined earn no credit
    loaded_s: float = 0.0
    loaded: bool = False
    unloaded: bool = False
    hold_s: float = 0.0
    invalid_reason: str | None = None
    events: list = field(default_factory=list)

    def reset(self):
        self.__init__(tray_center=self.tray_center, preassembled=self.preassembled)

    def connections(self, o: Observed) -> dict:
        p = o.poses
        return {
            "brick": o.joined["brick"] and d.fixed_seat(p["chassis"], p["brick"], d.BRICK_IN_CHASSIS)["seated"],
            "cab": o.joined["cab"] and d.fixed_seat(p["chassis"], p["cab"], d.CAB_IN_CHASSIS)["seated"],
            "hinge": o.joined["hinge"] and d.fixed_seat(p["cab"], p["hinge"], d.HINGE_IN_CAB)["seated"],
            "bed": o.joined["bed"] and d.hinge_seat(p["hinge"], p["bed"])["seated"],
        }

    def update(self, o: Observed) -> tuple[float, dict]:
        p = o.poses
        links = self.connections(o)
        bed = d.hinge_seat(p["hinge"], p["bed"])
        payload = np.asarray(p["payload"], float)
        hand_gap = min(float(np.linalg.norm(np.asarray(h) - payload[:3])) for h in o.hands)
        lin, ang = o.speeds["payload"]
        truck_tilt = tilt(p["chassis"])
        if self.loaded and not self.unloaded:
            reason = None
            if truck_tilt > TRUCK_TILT:
                reason = "truck tipped"
            elif not links["bed"]:
                reason = "bed connection lost"
            elif hand_gap < HAND_CLEAR and not payload_in_bed(p["bed"], payload):
                reason = "payload taken out of the bed by hand"
            if reason:
                self.loaded, self.loaded_s, self.invalid_reason = False, 0.0, reason
                self.events.append(reason)
        resting_in_bed = (links["bed"] and bed["tip"] < math.radians(10) and payload_in_bed(p["bed"], payload)
                          and hand_gap >= HAND_CLEAR and lin < REST_SPEED and ang < REST_SPIN)
        self.loaded_s = self.loaded_s + o.dt if resting_in_bed and not self.unloaded else 0.0
        if self.loaded_s + 1e-9 >= LOADED_S and not self.loaded:
            self.loaded = True
            self.events.append("loaded")
        if self.loaded and not self.unloaded:
            offset, _ = d.relative(p["bed"], payload)
            if (offset[1] > BED_INNER[2] and abs(offset[0]) < CUTOUT and bed["tip"] >= EXIT_TIP
                    and hand_gap >= HAND_CLEAR and links["bed"] and truck_tilt <= TRUCK_TILT):
                self.unloaded = True
                self.events.append("unloaded through rear opening")
        in_tray = payload_in_tray(payload, self.tray_center)
        upright = truck_tilt <= UPRIGHT_TILT and abs(float(p["chassis"][2]) - d.CHASSIS_Z) <= 0.004
        parts = [p[n] for n in ("chassis", "brick", "cab", "hinge", "bed")]
        clear = o.grippers_open and all(
            float(np.linalg.norm(np.asarray(h) - np.asarray(q)[:3])) >= PART_CLEAR for h in o.hands for q in parts)
        still = all(o.speeds[n][0] < REST_SPEED and o.speeds[n][1] < REST_SPIN for n in o.speeds)
        complete = all(links.values()) and self.unloaded and in_tray and upright and clear and still
        self.hold_s = self.hold_s + o.dt if complete else 0.0
        success = self.hold_s + 1e-9 >= HOLD_S
        if set(self.preassembled) >= {"brick", "cab", "hinge"}:
            # Bed task: the truck starts built; credit only for the bed attached on an intact truck,
            # the cube loaded, and the full functional check.
            base = links["brick"] and links["cab"] and links["hinge"]
            score = 0.0
            if base and links["bed"]:
                score = 1.0 if success else (0.75 if (self.loaded or self.unloaded) else 0.5)
        elif links["brick"] and links["cab"]:
            score = 0.25
            if links["hinge"]:
                score = 0.5
                if links["bed"]:
                    score = 1.0 if success else 0.75
        else:
            score = 0.0
        info = {"score": score, "success": success, "links": links, "loaded": self.loaded,
                "unloaded": self.unloaded, "payload_in_tray": in_tray, "upright": upright,
                "hands_clear": clear, "bed_tip_deg": math.degrees(bed["tip"]) if math.isfinite(bed["tip"]) else None,
                "truck_tilt_deg": math.degrees(truck_tilt), "hand_to_payload_m": hand_gap,
                "still": still, "speeds": {k: [round(v[0], 4), round(v[1], 3)] for k, v in o.speeds.items()},
                "complete": complete, "hold_s": self.hold_s,
                "invalidated": self.invalid_reason}
        return score, info
