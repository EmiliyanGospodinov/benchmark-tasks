"""Geometry, masses, springs and the starting layout of the blade holder. NumPy only.

Units: metres, kilograms, newtons, radians unless named _DEG; quaternions xyzw; z up.

Unpowered bench-held training holder (original design). The insertion axis runs along world y at
x = AXIS_X, AXIS_Z above the table; blades point toward -y out of the holder's nose (y = NOSE_Y).
- Holder: a channel for the tang, on a carriage rail along y. A spring (CARRIAGE_K) holds the
  carriage on its +y stop; pulling the blade outward extends it (the tug gauge, readable in mm).
- Collar: turns about the axis at the nose, spring-returned onto its 0 stop, travel to
  -COLLAR_TRAVEL_DEG; a radial lever tab on top is the grip. A cam peg on the collar's +x side
  lifts the locking pin's follower arm as the collar turns (contact, not a coupling).
- Locking pin: vertical, spring-loaded down onto its stop, in the notch of the seated tang. Its
  flat face blocks the tang both ways (no lead-in ramp: the blade cannot push it open).
- Blade: a flat training insert. Tang (16 mm tall) with a 6 mm notch in its top edge; the taller
  body (24 mm) forms the shoulder that seats on the nose face. The blunt "teeth" edge is the
  black lower edge; the notch is up when the blade is the right way up.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # station table top

AXIS_X, AXIS_Z = 0.30, TOP + 0.07
NOSE_Y = 0.02  # holder nose face (the shoulder seats here)
HOLDER_LEN = 0.06  # nose to back, along +y
HOLDER_HALF_W, HOLDER_HALF_H = 0.012, 0.016

# blade (training insert)
BLADE_T = 0.006  # thickness (x)
TANG_H, BODY_H = 0.016, 0.030  # heights (z), centred on the axis
TANG_L, BODY_L = 0.050, 0.170  # along y: tang inside the holder, body outside
NOTCH_Y, NOTCH_W, NOTCH_D = 0.030, 0.010, 0.006  # notch centre from the shoulder, width (y), depth (z)
SLOT_CLEAR = 0.0006  # channel clearance per side (x and z)
MOUTH = (0.005, 0.011, 0.008)  # entry chamfer: half-width (x), half-height (z) at the nose face, depth (y)

# locking pin (vertical, at the notch of a seated tang)
SEAT_Y = NOSE_Y - 0.0009  # where the shoulder actually bears: on the chamfer plates' front edges
PIN_Y = SEAT_Y + NOTCH_Y
PIN_W = 0.008  # square section (x, y)
PIN_TRAVEL = 0.014  # lift stop above the engaged stop
PIN_K, PIN_PRELOAD = 300.0, 0.9  # N/m, N (spring pushing the pin down onto its stop)
PIN_ENGAGE = TANG_H / 2 - NOTCH_D + 0.0005  # pin bottom above the axis when engaged
PIN_LEN = 0.024
FOLLOWER_X = 0.040  # follower arm reaches out over the cam peg (along +x)

# collar
COLLAR_R = 0.020  # visual ring radius
COLLAR_Y = NOSE_Y - 0.008  # collar centre (in front of the nose)
COLLAR_TRAVEL_DEG = 40.0
COLLAR_K_DEG = 0.0025  # N m / deg return spring
COLLAR_PRELOAD_DEG = 6.0  # spring set 6 deg beyond the stop (preload 0.015 N m)
LEVER_R = 0.050  # lever tab tip from the axis
LEVER = (0.012, 0.016, 0.028)  # tab (x, y, z extent radially)
CAM_R, CAM_PHI_DEG = 0.035, 60.0  # cam peg radius from the axis and angle from +z toward +x
PEG_R = 0.004  # half-size of the square cam peg

# carriage (tug gauge)
CARRIAGE_K = 150.0  # N/m
CARRIAGE_TRAVEL = 0.020

# a seated blade's tug check
TUG_MIN_EXT = 0.006  # m of carriage extension (1.2 N including 0.3 N preload) ...
TUG_HOLD_S = 0.5  # ... held this long with the blade seated and locked

MASS = {"holder": 0.40, "collar": 0.08, "pin": 0.04, "old_blade": 0.06, "new_blade": 0.06}
FRICTION = {"holder": (0.3, 0.25), "collar": (0.8, 0.7), "pin": (0.2, 0.15),
            "old_blade": (0.6, 0.5), "new_blade": (0.6, 0.5)}

GRIP_BLOCK = (0.020, 0.030, 0.030)  # moulded grip block at the far end of each training blade
GRIP_FROM_SHOULDER = BODY_L - GRIP_BLOCK[1] / 2  # blade grip point (block centre), from the shoulder

# layout of the loose parts
NEW_REST_X = AXIS_X + 0.10  # replacement blade lies in a two-fork rack, parallel to the axis
TRAY_XY = (0.20, -0.10)  # rest tray for the removed blade
TRAY = (0.09, 0.22, 0.020)  # inner x, y, wall height


def follower_bottom_rest() -> float:
    """Follower underside above the axis with the pin engaged: just clear of the cam peg at rest."""
    phi = math.radians(CAM_PHI_DEG)
    return CAM_R * math.cos(phi) + PEG_R + 0.0008


def cam_lift(collar_deg: float) -> float:
    """Peg top rise for a collar angle (negative = turned toward release)."""
    phi, d = math.radians(CAM_PHI_DEG), math.radians(collar_deg)
    x, z = CAM_R * math.sin(phi), CAM_R * math.cos(phi)
    return (-x * math.sin(d) + z * math.cos(d)) - z


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def roll_y(r: np.ndarray) -> float:
    """Rotation about +y (radians)."""
    return math.atan2(r[0, 2], r[2, 2])


def blade_start(seated: bool) -> np.ndarray:
    """Blade frame origin = shoulder centre on the axis; tang toward +y."""
    if seated:
        return np.array([AXIS_X, NOSE_Y, AXIS_Z, 0, 0, 0, 1.0])
    return np.array([NEW_REST_X, NOSE_Y - 0.01, AXIS_Z - 0.03, 0, 0, 0, 1.0])


def start_poses() -> dict[str, np.ndarray]:
    ident = [0, 0, 0, 1.0]
    return {"holder": np.array([AXIS_X, NOSE_Y, AXIS_Z, *ident]),
            "collar": np.array([AXIS_X, COLLAR_Y, AXIS_Z, *ident]),
            "pin": np.array([AXIS_X, PIN_Y, AXIS_Z, *ident]),
            "old_blade": blade_start(True), "new_blade": blade_start(False)}
