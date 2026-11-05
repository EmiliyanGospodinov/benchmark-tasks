"""Geometry and layout of the modular retrieval tool task. NumPy only; original nominal design, uncalibrated.

Coordinates: x forward from the line between the arm bases, y left, z up; metres, kilograms, radians.

Parts (all rigid; contact-only shaft/head attachment, spring-return release control):
- shaft: handle block + flat rod lying along +x; at the rod tip a vertical bayonet pin with two lugs (along
  the shaft's y axis). Frame origin: pin axis at the table surface (bottom of the rod).
- head: a plate with a keyed square hole (slots along the head's x axis) under a closed square bell that
  encloses the lugs, plus a hook arm along +x_head ending in a downward tooth. Frame origin: pin axis at the
  plate's bottom face. Seated and turned so x_head = x_shaft, the lugs lie over the solid plate inside the bell
  and stop blocks end the quarter turn.
- cartridge: a block with a front lip and a pocket behind it (the pull eye). Frame origin: front-bottom centre.
- hood (fixed): closed sides, roof and rear; one 5 cm wide, 10 cm tall mouth facing the robots.
- tray (fixed) and head stand (fixed).
"""
from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # station table top

# shaft
ROD_L, ROD_W, ROD_T = 0.160, 0.016, 0.012
HANDLE_L, HANDLE_W, HANDLE_H = 0.080, 0.024, 0.034  # tall enough for the pads (fingertips reach ~24 mm below the TCP)
PIN_R = 0.005
PLATE_T, LUG_GAP, LUG_T, LUG_W, LUG_R = 0.006, 0.001, 0.004, 0.004, 0.013
# head
PLATE_HALF, HOLE_HALF = 0.018, 0.008  # 16 mm hole on the 10 mm pin: 3 mm clearance
SLOT_HALF_W, SLOT_END = 0.004, 0.0145  # 8 mm slots for the 4 mm lugs
BELL_WALL, BELL_CAVITY, BELL_ROOF = 0.002, 0.010, 0.003  # bell above the plate encloses the lugs and release gate
NECK = (0.022, 0.022, 0.020)  # grip neck on the bell: fingertips stay above the rod when seated
STOP_ANGLE = math.radians(-124.0)  # stop blocks (head frame, r 11.5 mm) end the quarter turn just past locked
STOP_R = 0.0115
ARM_X0, ARM_X1, ARM_W, ARM_H = PLATE_HALF, 0.075, 0.012, 0.016
TOOTH_X0, TOOTH_DROP = 0.065, 0.010  # tooth spans x 0.065-0.075, down to 10 mm below the plate bottom
# cartridge
CART_L, CART_W, CART_H = 0.060, 0.040, 0.018  # CART_H: front lip and pocket rails
CART_BODY_H = 0.036  # body behind the pocket: tall enough to pinch with the YAM fingers
CART_BASE_T = 0.006
POCKET_X0, POCKET_X1, POCKET_HALF_W = 0.008, 0.028, 0.014  # pull-eye pocket 20 x 28 mm
# hood: interior x from the mouth to the rear wall, |y| < half width, z up to the roof (table is the floor)
HOOD_MOUTH_X, HOOD_DEPTH, HOOD_HALF_W, HOOD_H, HOOD_WALL = 0.40, 0.190, 0.025, 0.100, 0.010
CART_DEPTH = 0.120  # cartridge front this far inside the mouth (beyond fingertip + bare-head reach)
# fixed accessories
HEAD_STAND = (0.026, 0.026, ROD_T)  # head stand block (x, y, z): same height as the rod top
TRAY_CENTER, TRAY_INNER, TRAY_RIM = (0.24, 0.21), (0.100, 0.080), (0.006, 0.012)  # rim (thickness, height)
TOOL_REST_TOL = 0.03  # m, the assembled tool back at its rest: pin within this of its start
MAT_T = 0.0  # the tool lies directly on the table (a mat edge would stop the cartridge being pulled out)
BRACKET = ((0.62, 0.0), (0.03, 0.12, 0.02))  # fixed hood mounting bracket behind the hood (task-file object)

# Spring-return squeeze release. The native guide runs along shaft y; a rigid
# linkage connects a handle trigger to a gate beside the pin. Closing the gripper
# on trigger and fixed handle retracts the gate 20 mm toward +y.
RELEASE_TRAVEL = 0.020
RELEASE_K, RELEASE_PRELOAD, RELEASE_DAMP = 100.0, 0.003, 2.0
RELEASE_MASS = 0.050
GATE_X, GATE_Y, GATE_Z = 0.013, 0.008, ROD_T + 0.010
GATE_W, GATE_T = 0.004, 0.003

MASS = {"shaft": 0.150, "head": 0.060, "cartridge": 0.120}  # handle-heavy shaft, die-cast head, plastic part
SHAFT_COM_X = -(ROD_L + 0.6 * HANDLE_L)  # most of the shaft mass is in the handle
HANDLE_FRICTION = (0.9, 0.8)  # rubber grip sleeve on the handle only
FRICTION = {"shaft": (0.30, 0.25), "head": (0.40, 0.35), "cartridge": (0.40, 0.35)}  # metal tool, plastic part

PIN_XY = (0.30, 0.0)  # shaft pin axis at the start (tool rest)
HEAD_XY = (0.30, 0.15)  # head stand (head turned -90 deg: hook arm toward -y, slots along the lugs)
HEAD_START_YAW = -math.pi / 2


def quat_to_matrix(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_quat(a):
    return np.array([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])


def yaw_of(r):
    return math.atan2(r[1, 0], r[0, 0])


def cart_start_x():
    return HOOD_MOUTH_X + CART_DEPTH


def start_poses(layout="loose"):
    """layout: 'loose' (task start), 'assembled' (diagnostic: head pre-seated and locked)."""
    shaft = np.r_[PIN_XY, TOP + MAT_T, 0, 0, 0, 1.0]
    if layout == "assembled":
        head = np.r_[PIN_XY, TOP + MAT_T + ROD_T, 0, 0, 0, 1.0]
    else:
        head = np.r_[HEAD_XY, TOP + HEAD_STAND[2], yaw_quat(HEAD_START_YAW)]
    cart = np.r_[cart_start_x(), 0.0, TOP, 0, 0, 0, 1.0]
    plunger = shaft.copy()
    hood = np.r_[HOOD_MOUTH_X, 0.0, TOP, 0, 0, 0, 1.0]
    tray = np.r_[TRAY_CENTER, TOP, 0, 0, 0, 1.0]
    stand = np.r_[HEAD_XY, TOP, 0, 0, 0, 1.0]
    return {"shaft": shaft, "head": head, "cartridge": cart, "hood": hood, "tray": tray, "stand": stand,
            "plunger": plunger}


def coupling_state(shaft_pose, head_pose):
    """Measured relative pose of the head on the pin: (seat height error m, radial offset m, yaw rad)."""
    rs = quat_to_matrix(shaft_pose[3:])
    rel = rs.T @ (np.asarray(head_pose[:3]) - np.asarray(shaft_pose[:3]))
    rh = rs.T @ quat_to_matrix(head_pose[3:])
    return float(rel[2] - ROD_T), float(math.hypot(rel[0], rel[1])), yaw_of(rh)


LOCK_MIN_YAW = math.radians(-9.0)  # retained quarter-turn orientation


def locked(shaft_pose, head_pose, seat_tol=0.002, radial_tol=0.0035):
    """Bayonet seating/orientation. Release return is checked separately by release_closed."""
    for pose in (shaft_pose, head_pose):
        if not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
            return False
    dz, dr, yaw = coupling_state(shaft_pose, head_pose)
    relative = quat_to_matrix(shaft_pose[3:]).T @ quat_to_matrix(head_pose[3:])
    return (abs(dz) <= seat_tol and dr <= radial_tol
            and relative[2, 2] >= math.cos(math.radians(10))
            and LOCK_MIN_YAW <= yaw <= math.radians(15))


def release_closed(shaft_pose, control_pose):
    """Measured control pose at its return stop, with a sound shaft-fixed guide."""
    for p in (shaft_pose, control_pose):
        if not np.isfinite(p).all() or np.linalg.norm(p[3:]) < 1e-8:
            return False
    rs = quat_to_matrix(shaft_pose[3:])
    rel = rs.T @ (np.asarray(control_pose[:3])-np.asarray(shaft_pose[:3]))
    rotation = rs.T @ quat_to_matrix(control_pose[3:])
    return bool(abs(rel[1]) <= 0.002 and np.linalg.norm(rel[[0,2]]) <= 0.002
                and (np.trace(rotation)-1)/2 >= math.cos(math.radians(5)))
