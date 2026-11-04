"""Original compact single-acting test pump; uncalibrated prototype dimensions.

Coordinates: x forward from the line between the arm bases, y left, z up; metres, kg, kPa gauge.
The pump lies along x on two flat moulded feet, outlet end toward -x, handle toward +x.
"""
import math

import numpy as np

TOP = 0.7475
BARREL_LENGTH = 0.180
BARREL_RADIUS = 0.020
BARREL_WALL = 0.003  # hollow tube: the rod runs in a real bore
RING_RADIUS = 0.023  # end rings (visual)
FOOT = (0.030, 0.036, 0.009)  # moulded flat feet under the barrel (stop it rolling)
FOOT_X = 0.065
AXIS_HEIGHT = 0.026  # barrel axis above the feet bottom (the mat surface)
MAT_T = 0.002  # rubber mat under the pump
MAT_CENTER, MAT_SIZE = (0.20, 0.0), (0.43, 0.14)  # x, y
MIN_HANDLE_OFFSET = 0.160
STROKE = 0.080
BORE_RADIUS = 0.005  # 10 mm bore: sized so the target needs about 5 N (YAM arms are compliant)
DEAD_VOLUME = 0.8e-6
RECEIVER_VOLUME = 30e-6  # test chamber (with hose), fixed volume
MASS = {"barrel": 0.20, "piston": 0.035}
FRICTION = {"barrel": (0.8, 0.6), "piston": (0.8, 0.6)}
BARREL_POS = np.array([0.12, 0.0, TOP + MAT_T + AXIS_HEIGHT])
SEAL_DAMPING = 6.0  # N s/m on the guide (seal drag), uncalibrated

# Test chamber with dial gauge, fixed on the table (hose to the pump outlet not simulated).
CHAMBER_XY = (0.43, -0.16)
CHAMBER_R, CHAMBER_H = 0.064, 0.040
DIAL_R = 0.060  # 12 cm test-gauge dial: about 32 px across in the 320 px top camera
GAUGE_MAX_KPA = 120.0
TICK_MINOR, TICK_MAJOR, TICK_BAND = 10, 20, 2
GAUGE_SWEEP_DEG = 270.0  # clockwise seen from above, 0 kPa at 135 deg, 60 kPa toward +x
TARGET_KPA, BAND_KPA = 60.0, 6.0  # manual: 60 kPa, acceptable 54-66
OVER_KPA = 80.0


def needle_angle(kpa):
    """Needle direction (radians, from +x, counter-clockwise) for a gauge reading."""
    k = min(max(float(kpa), 0.0), GAUGE_MAX_KPA)
    return math.radians(GAUGE_SWEEP_DEG / 2 - GAUGE_SWEEP_DEG * k / GAUGE_MAX_KPA)


def chamber_top():
    return TOP + CHAMBER_H



def quat_to_matrix(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def feet_on_mat(barrel_pose) -> bool:
    """Both feet's footprints (all eight corners) lie on the mat, judged from the barrel's measured pose."""
    r = quat_to_matrix(barrel_pose[3:])
    c = np.asarray(barrel_pose[:3], float)
    hx, hy = FOOT[0] / 2, FOOT[1] / 2
    for fx in (-FOOT_X, FOOT_X):
        for dx in (-hx, hx):
            for dy in (-hy, hy):
                p = c + r @ np.array([fx + dx, dy, -AXIS_HEIGHT])
                if abs(p[0] - MAT_CENTER[0]) > MAT_SIZE[0] / 2 or abs(p[1] - MAT_CENTER[1]) > MAT_SIZE[1] / 2:
                    return False
    return True


def yaw_quat(a):
    return np.array([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])


def start_poses():
    b = np.r_[BARREL_POS, 0., 0., 0., 1.]
    p = b.copy()
    p[0] += MIN_HANDLE_OFFSET + STROKE
    chamber = np.r_[CHAMBER_XY, TOP, 0., 0., 0., 1.]
    needle = np.r_[CHAMBER_XY, chamber_top() + 0.004, yaw_quat(needle_angle(0.0))]
    return {"barrel": b, "piston": p, "chamber": chamber, "needle": needle}
