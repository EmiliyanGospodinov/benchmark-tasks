"""Geometry, masses and conventions of the ratchet lowering trainer. NumPy only.

Units: metres, kilograms, radians unless named _DEG; quaternions xyzw. Station frame: x forward from the
line between the arm bases, y left, z up; table top at TOP.

An original low-energy rigid trainer (not a commercial winch): a horizontal shaft along y carries the
rotor: a load lever pointing forward (+x) with a weight, a 24-tooth ratchet wheel at the right end
(-y) and a crank at the left end (+y) whose handle sleeve spins freely on its pin. The weight is a 200 g aluminium block. A spring-returned pawl
on its own pivot engages the wheel; pressing its thumb tab down lifts it out. Gravity lowers the lever.
Lever elevation THETA is the lever's angle above horizontal; the rotor joint angle about +y is -THETA.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475

SHAFT = np.array([0.24, 0.0, TOP + 0.10])  # shaft axis point; the axis runs along y

# Rotor (one rigid body; frame origin on the shaft axis, lever along local +x)
LEVER_L = 0.13  # shaft axis to weight centre
LEVER_W, LEVER_T = 0.016, 0.012
WEIGHT = np.array([0.050, 0.040, 0.040])  # aluminium block on the lever end (0.20 kg)
WHEEL_Y, WHEEL_T = -0.080, 0.012
ROOT_R, TIP_R, TEETH = 0.040, 0.048, 24  # sawtooth wheel, 15-degree pitch
CRANK_Y, CRANK_R = 0.090, 0.070  # crank arm plane and handle pin radius
CRANK_LOCAL_DEG = 65.0  # pin direction in the rotor frame (world angle = this + THETA)
HANDLE_R, HANDLE_L = 0.010, 0.040  # free-spinning handle sleeve
HANDLE_Y = CRANK_Y + 0.006 + HANDLE_L / 2
MASS = {"rotor": 0.41, "handle": 0.02, "pawl": 0.03}
ROTOR_COM = (0.0713, -0.009, 0.0)  # weight 0.20, lever 0.05, wheel 0.08, shaft 0.05, crank 0.03
FRICTION = {"rotor": (0.6, 0.5), "handle": (1.0, 0.9), "pawl": (0.5, 0.4), "frame": (0.6, 0.5)}

# Pawl (frame origin at its pivot; axis along y). Positions relative to the shaft axis in (x, z).
PAWL_Y, PAWL_T = WHEEL_Y, 0.010
PAWL_PIVOT = np.array([0.050, 0.064])
PAWL_NOSE = np.array([-0.006, 0.044])  # centre of the nose end face (engaging the tooth's steep face)
NOSE = (0.010, 0.006)  # nose length along the pawl, thickness
TAB_END = np.array([0.090, 0.072])  # thumb tab centre, beyond the pivot
TAB = (0.022, 0.018, 0.006)  # tab pad (x, y, z)
PAWL_SPRING_TARGET_DEG = -4.0  # pressing the nose into the wheel
PAWL_SPRING = 0.004  # N m / deg (declared torsion spring)
PAWL_LIMITS_DEG = (-6.0, 20.0)  # positive lifts the nose

THETA_START_DEG = 35.0
THETA_REST_DEG = -28.0  # the weight's flat face lands on the receiving rest here
THETA_LIMIT_DEG = (-31.0, 40.0)
TOOTH_DEG = 360.0 / TEETH

# Procedure and scoring targets (lever angle above horizontal; a protractor on a real trainer)
HOLD_DEG, HOLD_BAND_DEG = 5.0, (1.0, 10.0)  # intermediate hold: the second tooth below the start
REST_TOL_DEG = 1.5
SPEED_LIMIT_DPS = 40.0  # controlled lowering; a free release passes this within a few degrees


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


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


def theta_of(r: np.ndarray) -> float:
    """Lever elevation (radians) from the rotor's rotation: angle of its local +x above horizontal."""
    return math.atan2(r[2, 0], r[0, 0])


def pawl_angle_of(r: np.ndarray) -> float:
    """Pawl rotation about +y (radians); positive lifts the nose."""
    return -math.atan2(r[2, 0], r[0, 0])


def tooth_phase_deg() -> float:
    """Local angle of tooth 0's steep face, so that a steep face meets the pawl nose at the start angle."""
    nose_deg = math.degrees(math.atan2(PAWL_NOSE[1], PAWL_NOSE[0]))
    return nose_deg - THETA_START_DEG + 0.4  # 0.4 degree of free settle onto the pawl


def hold_angles_deg():
    """Lever elevations at which the engaged pawl catches a tooth (every tooth pitch below the start)."""
    return [THETA_START_DEG - TOOTH_DEG * k for k in range(0, 6)]


def world(local_xz, y=0.0):
    return SHAFT + np.array([local_xz[0], y, local_xz[1]])


def handle_center(theta: float) -> np.ndarray:
    a = math.radians(CRANK_LOCAL_DEG) + theta
    return SHAFT + np.array([CRANK_R * math.cos(a), HANDLE_Y, CRANK_R * math.sin(a)])


def tab_center(pawl_angle: float = 0.0) -> np.ndarray:
    rel = TAB_END - PAWL_PIVOT
    r = rot_y(pawl_angle)
    v = r @ np.array([rel[0], 0.0, rel[1]])
    return world(PAWL_PIVOT, PAWL_Y) + v


def start_poses() -> dict[str, np.ndarray]:
    phi = -math.radians(THETA_START_DEG)
    q = matrix_to_quat(rot_y(phi))
    rotor = np.array([*SHAFT, *q])
    handle = np.array([*handle_center(math.radians(THETA_START_DEG)), *q])
    pawl = np.array([*world(PAWL_PIVOT, PAWL_Y), 0, 0, 0, 1.0])
    frame = np.array([SHAFT[0], 0.0, TOP, 0, 0, 0, 1.0])
    return {"frame": frame, "rotor": rotor, "handle": handle, "pawl": pawl}
