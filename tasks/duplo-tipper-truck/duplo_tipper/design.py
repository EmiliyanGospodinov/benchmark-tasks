"""Geometry, layout and connection tolerances for the DUPLO tipper truck. NumPy only.

Units are metres, radians and kilograms; quaternions are xyzw. Every part frame is
the LDraw part frame converted to Z-up metres ((x, y, z) = (X, Z, -Y) * 0.4 mm),
except where noted. The truck is built along +y: the cab is at -y and the
tipping bed's hinge, tow pin and rear opening face +y.

Sources: LDraw 15314 (car base), 3437 (2x2 brick), 65084 (hinge plate) and 14094
(tipper bed) give the mating geometry. The cab 15454pb06 and the factory-fitted
wheels of 15314c01 have no LDraw model; their geometry here is an original
approximation sized from BrickLink catalogue data (cab 8.5 x 4 x 4.5 studs,
59.25 g; complete base 74.9 g, packed height 35.8 mm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TOP = 0.7475  # station table top (yam_arena.task_file.TABLE_TOP["station"])
STUD = 0.016  # DUPLO stud pitch
BRICK_H = 0.0192
PLATE_H = 0.0096

# -- chassis 15314c01 (frame origin: centre of the deck top, below the studs) -----------------
CHASSIS_SIZE = (0.064, 0.128, 0.0192)  # body below the deck; arches and tow pin are visual only
AXLE_Y = 0.032  # wheel centres at y = +-32 mm (LDraw 15314s01 arch centres)
AXLE_Z = -0.0122
TIRE_RADIUS = 0.016  # approximation: arch inner radius is 18 mm; packed height 35.8 mm
TIRE_WIDTH = 0.012
TIRE_X = 0.028
# Coulomb friction of each wheel bearing, N m: a 0.1 N push at the tyres starts the truck rolling
# (both axles); a light free-rolling toy. Uncalibrated.
BEARING_TORQUE = 0.05 * TIRE_RADIUS
CHASSIS_Z = TOP - (AXLE_Z - TIRE_RADIUS)  # chassis origin height when standing on its wheels

MASS = {  # kg; catalogue weights, wheels split from the 74.9 g base
    "chassis": 0.0599, "axle_front": 0.0075, "axle_rear": 0.0075,
    "brick": 0.0124, "cab": 0.05925, "hinge": 0.00832, "bed": 0.02623,
    "payload": 0.0050,
}
PAYLOAD_SIZE = 0.020  # 20 mm cube of light hardwood (about 0.6 g/cm^3)
TRAY_INNER = (0.100, 0.090)  # world x (along the truck), y inside the walls
TRAY_WALL = (0.004, 0.012)  # thickness, height above the 4 mm floor
TRAY_FLOOR = 0.004

# -- assembly targets: each child's frame expressed in its parent's frame ---------------------
# Step 1: the blue 2x2 brick on the two centre stud columns, rows 2-3 from the front.
# Brick frame: body centre (LDraw origin shifted down by half a brick height).
BRICK_IN_CHASSIS = np.array([0.0, -0.040, BRICK_H / 2])
# Step 2: the cab over the whole base, flush with its rear (cab frame = chassis frame).
CAB_IN_CHASSIS = np.array([0.0, 0.0, 0.0])
CAB_DECK_Z = BRICK_H  # top of the cab's rear deck above the chassis deck
# Step 3a: hinge plate on the cab deck, hinge knuckles at the rear (plate frame = LDraw 65084:
# origin at the plate top over its 2x2 studs; hinge axis along x at (0, 48, +2) mm).
HINGE_IN_CAB = np.array([0.0, 0.009, CAB_DECK_Z + PLATE_H])
HINGE_AXIS_IN_PLATE = np.array([0.0, 0.048, 0.002])
# Step 3b: bed origin on the hinge axis (LDraw 14094 !HELP: plate at (0, 5, -120) LDU).
BED_IN_HINGE = HINGE_AXIS_IN_PLATE
BED_TIP_LIMIT = math.radians(65)  # bed rotates about -x from 0 (closed) to this angle (open)

# -- connection surrogate tolerances (uncalibrated; written before the demonstration) ---------
SEAT_XY = 0.0015  # lateral position error, m
SEAT_Z = 0.0010  # seating height error, m
SEAT_ANGLE = math.radians(4)  # residual tilt / yaw
HINGE_AXIS_POS = 0.0015  # bed origin to hinge axis
HINGE_AXIS_ANGLE = math.radians(5)  # bed rotation must be about the plate's x axis
PRESS_FORCE = 2.0  # N of measured child-to-parent contact force
PRESS_DWELL = 0.10  # s
SLOW = 0.03  # m/s relative speed
BREAK = {  # (force N, torque N m): provisional break limits of the retention joints
    "brick": (25.0, 0.5), "cab": (40.0, 1.5), "hinge": (20.0, 0.4), "bed": (15.0, 0.3),
}
LOST_POS = 0.004  # a connection whose bodies drift this far from the joined pose is lost
LOST_ANGLE = 0.15

# -- start layout (world) ---------------------------------------------------------------------
# A1/r3: the truck lies across the table (frame +y = world +y): the cab toward the right arm (-y),
# the hinge, rear opening and tray toward the left arm (+y). The right gripper counterholds the cab
# front with a low side grip from -y (as A4's bottle hold) and the left places the bed top-down at
# the hinge, so the two wrists work from opposite sides (r2's along-x layout put both wrists
# above the 13 cm truck: bed-008..011).
TRUCK_XY = (0.30, 0.0)
TRUCK_YAW = 0.0
TRUCK_QUAT = np.array([0.0, 0.0, math.sin(TRUCK_YAW / 2), math.cos(TRUCK_YAW / 2)])
TRAY_GAP = 0.001  # tray placed against the truck rear: the load drops off the hinge nearly vertically (bed-023)
TRAY_CENTER = (TRUCK_XY[0], TRUCK_XY[1] + 0.064 + TRAY_GAP + TRAY_WALL[0] + TRAY_INNER[1] / 2)


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not n > 0:
        return np.full((3, 3), np.nan)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def relative(parent, child) -> tuple[np.ndarray, np.ndarray]:
    """Child pose [x,y,z,qx,qy,qz,qw] in the parent's frame: (offset, rotation matrix)."""
    parent, child = np.asarray(parent, float), np.asarray(child, float)
    rp, rc = quat_to_matrix(parent[3:]), quat_to_matrix(child[3:])
    return rp.T @ (child[:3] - parent[:3]), rp.T @ rc


def angle_of(r: np.ndarray) -> float:
    return float(math.acos(np.clip((np.trace(r) - 1) / 2, -1, 1)))


def fixed_seat(parent, child, target) -> dict:
    """Seating error of a stud connection at its manual position (no rotation allowed)."""
    offset, r = relative(parent, child)
    if not (np.isfinite(offset).all() and np.isfinite(r).all()):
        return {"xy": math.inf, "z": math.inf, "angle": math.inf, "seated": False}
    error = offset - np.asarray(target)
    out = {"xy": float(np.hypot(error[0], error[1])), "z": float(abs(error[2])), "angle": angle_of(r)}
    out["seated"] = out["xy"] <= SEAT_XY and out["z"] <= SEAT_Z and out["angle"] <= SEAT_ANGLE
    return out


def hinge_seat(plate, bed) -> dict:
    """Bed on the hinge axis, rotated only about the plate's x axis, within the hinge range.

    ``tip`` is the opening angle (0 closed, positive as the front rises toward the rear).
    """
    offset, r = relative(plate, bed)
    if not (np.isfinite(offset).all() and np.isfinite(r).all()):
        return {"axis_pos": math.inf, "axis_angle": math.inf, "tip": math.nan, "seated": False}
    tip = -math.atan2(r[2, 1], r[1, 1])  # rotation about -x
    off_axis = angle_of(rot_x(tip) @ r)  # residual after removing the hinge rotation
    pos = float(np.linalg.norm(offset - BED_IN_HINGE))
    seated = (pos <= HINGE_AXIS_POS and off_axis <= HINGE_AXIS_ANGLE
              and -math.radians(3) <= tip <= BED_TIP_LIMIT + math.radians(3))
    return {"axis_pos": pos, "axis_angle": off_axis, "tip": tip, "seated": seated}


@dataclass
class Gate:
    """Consecutive seated, slow, pressed contact before a retention joint is enabled."""

    pressed_s: float = 0.0

    def update(self, seated: bool, force: float, speed: float, dt: float) -> bool:
        ok = seated and force >= PRESS_FORCE and speed < SLOW
        self.pressed_s = self.pressed_s + dt if ok else 0.0
        return self.pressed_s + 1e-9 >= PRESS_DWELL


def world_pose(parent_pose, offset, rotation=np.eye(3)) -> np.ndarray:
    """World [x,y,z,qx,qy,qz,qw] of a frame given in the parent's frame."""
    parent_pose = np.asarray(parent_pose, float)
    rp = quat_to_matrix(parent_pose[3:])
    r = rp @ rotation
    return np.r_[parent_pose[:3] + rp @ np.asarray(offset), matrix_to_quat(r)]


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def assembled_poses(chassis_pose=None) -> dict[str, np.ndarray]:
    """World poses of every truck part when correctly assembled (bed closed)."""
    if chassis_pose is None:
        chassis_pose = np.r_[TRUCK_XY[0], TRUCK_XY[1], CHASSIS_Z, TRUCK_QUAT]
    cab = world_pose(chassis_pose, CAB_IN_CHASSIS)
    hinge = world_pose(cab, HINGE_IN_CAB)
    return {
        "chassis": np.asarray(chassis_pose, float),
        "axle_front": world_pose(chassis_pose, (0, -AXLE_Y, AXLE_Z)),
        "axle_rear": world_pose(chassis_pose, (0, AXLE_Y, AXLE_Z)),
        "brick": world_pose(chassis_pose, BRICK_IN_CHASSIS),
        "cab": cab,
        "hinge": hinge,
        "bed": world_pose(hinge, BED_IN_HINGE),
    }
BEARING_DAMPING = 1.0e-5  # N m s / deg, light viscous part of the bearing
# Static (stick-slip) bearing friction, elasto-plastic: each axle is held by an angular spring to an
# anchor angle; when the spring torque would exceed BEARING_TORQUE the anchor slips along, so the axle
# turns against a constant Coulomb torque and sticks below it (no creep). The capped damper used before
# allowed steady creep (idle-013..015: 1.0-1.6 mm/s).
BEARING_STIFFNESS = BEARING_TORQUE / 0.5  # N m / deg: breakaway after 0.5 deg of elastic wind-up
