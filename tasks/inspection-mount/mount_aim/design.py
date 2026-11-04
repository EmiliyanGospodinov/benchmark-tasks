"""Geometry, masses and the clamp model of the inspection mount. NumPy only.

Units: metres, kilograms, radians unless named _DEG; quaternions xyzw; z up.

Fixture: a post fixed to the bench carries a horizontal pitch hinge (axis along world x). A 20 cm
arm hangs from the hinge toward -y with a dummy inspection head (camera block) at its end. A clamp
knob on top of the post (vertical axis) locks the hinge: turning it clockwise drives a toothed
rosette into mesh. The arm pitch angle PHI is the rotation about +x; positive PHI lowers the head.
Elevation of the head = -PHI.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # station table top

HINGE = np.array([0.25, 0.08, TOP + 0.12])  # pitch axis point (world)
ARM_L = 0.20  # hinge to head centre, along -y in the arm frame
BAR_W, BAR_T = 0.016, 0.012  # bar cross-section (x width, z thickness)
HUB_R = 0.016  # hinge hub radius (cylinder along x)
HEAD = np.array([0.040, 0.060, 0.035])  # head block size (x, y along the bar, z)
MASS = {"arm": 0.06, "head": 0.25, "knob": 0.12}  # head: dummy inspection camera
FRICTION = {"arm": (0.9, 0.8), "knob": (1.0, 0.9), "post": (0.6, 0.5)}

POST_XY = (HINGE[0] + 0.045, HINGE[1])  # post beside the hub (clear of the swinging bar)
POST_SIZE = (0.030, 0.040)
POST_TOP = TOP + 0.14
KNOB_LIFT = 0.012  # knob on a stem above the post top (fingertip clearance)
KNOB_R, KNOB_H = 0.020, 0.025
KNOB_CENTER = np.array([POST_XY[0], POST_XY[1], POST_TOP + KNOB_LIFT + KNOB_H / 2])

PHI_REST_DEG = 25.0  # lower stop: head 25 degrees below horizontal
PHI_UP_DEG = -45.0  # upper stop: head 45 degrees above horizontal
TARGET_ELEV_DEG = 15.0  # requested aim: head 15 degrees above horizontal (on the mark)

# Clamp (declared reduced-order model). Knob: revolute about z, tightened clockwise (negative),
# with viscous thread damping and stops at 0 (loose) and -KNOB_STOP_DEG (hand-tight).
# Rosette: TEETH teeth per turn. When the knob has been tightened past ENGAGE_DEG the teeth mesh
# at the tooth nearest the current pitch, modelled as a stiff angular spring (tooth compliance)
# whose torque saturates at LOCK_TORQUE (tooth shear capacity). Loosening below RELEASE_DEG
# separates the teeth again. Unmeshed, the hinge is free (no clamp friction).
KNOB_STOP_DEG = 170.0  # under half a turn: PhysX reports revolute angles in (-180, 180]
KNOB_DAMPING = 2.0e-4  # N m s / deg
KNOB_MAX_TORQUE = 0.5  # N m cap of that damping
ENGAGE_DEG, RELEASE_DEG = 150.0, 130.0
TEETH = 72  # 5 degree pitch
LOCK_STIFFNESS = 2.0  # N m / deg
LOCK_DAMPING = 0.02  # N m s / deg
LOCK_TORQUE = 20.0  # N m

TOOTH_DEG = 360.0 / TEETH


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return np.array([x, y, z, w])


def yaw_of(r: np.ndarray) -> float:
    return math.atan2(r[1, 0], r[0, 0])


def phi_of(r: np.ndarray) -> float:
    """Arm pitch about +x (radians); positive lowers the head."""
    return math.atan2(r[2, 1], r[1, 1])


def head_center(phi: float) -> np.ndarray:
    return HINGE + rot_x(phi) @ np.array([0.0, -ARM_L, 0.0])


def start_poses() -> dict[str, np.ndarray]:
    arm = np.array([*HINGE, *matrix_to_quat(rot_x(math.radians(PHI_REST_DEG)))])
    knob = np.array([*KNOB_CENTER, 0, 0, 0, 1.0])
    post = np.array([POST_XY[0], POST_XY[1], TOP, 0, 0, 0, 1.0])
    return {"post": post, "arm": arm, "knob": knob}
