"""Geometry, masses and conventions of the enlarged clear pin-tumbler trainer. NumPy only.

Units: metres, kilograms, radians unless named _DEG; quaternions xyzw. Station frame: x forward from the line
between the arm bases, y left, z up; table top at TOP.

The trainer is about 3x a standard 5-pin cylinder, cut down to 3 pin stacks. The plug axis runs along x; the
keyway face looks toward the robot (-x). The pins stand on top (+z) of the plug.

PLUG ANGLE theta is the plug's rotation about +x (the plug's top moves toward -y as it opens), 0 at rest,
90 degrees at the open mark. Plug frame: origin on the axis at the keyway face, local x into the lock.
PIN-HOLE OFFSETS (the binding order): each plug pin hole is wider than its pin on the opening side by GAP[i];
under turning pressure the plug turns until the smallest-gap hole wall meets its driver pin, so only that pin
binds. Setting it lets the plug turn on to the next-smallest gap. This is the standard manufacturing-tolerance
explanation of binding order, with the offsets exaggerated (0.5/2.0/3.5 mm here vs a few hundredths of a mm
in a real lock) so the order is deterministic and the set ledge is wide enough to see.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475

FACE_X = 0.25  # keyway face (plug front)
AXIS_Z = TOP + 0.10  # plug axis height
AXIS = np.array([FACE_X, 0.0, AXIS_Z])
R = 0.020  # plug radius = shear line radius
R_BORE = 0.0208  # housing bore (the housing never touches the plug; the joint carries it)
PLUG_L = 0.090
HOUSING_L = 0.095
PIN_X = (0.035, 0.050, 0.065)  # pin stacks from the face: front, middle, back
PIN_NAMES = ("front", "middle", "back")
PIN_W = 0.008  # square pins
HOLE_X = 0.009  # hole length along x
GAP = (0.0035, 0.0005, 0.0020)  # binding gap on the opening (+y local) side -> order middle, back, front
HOLE_MINUS_Y = -0.005  # hole wall on the other side (never touched)
SLICE = 0.0055  # half length (x) of the plug surface collider under each driver

# Pin stack, heights above the axis in the plug frame (key pins) / station frame (drivers) at rest.
KEY_L, DRIVER_L = 0.016, 0.018
KEY_BOTTOM0 = -0.002  # key pin bottom at rest: 10 mm into the keyway roof region
KEY_TOP0 = KEY_BOTTOM0 + KEY_L  # 0.014: 6 mm below the shear line
DRIVER_BOTTOM0 = KEY_TOP0
KEY_LIMITS = (0.0, 0.009)  # key pin travel (prismatic, plug frame)
DRIVER_LIMITS = (-0.003, 0.012)  # driver travel (prismatic, housing): may drop 2 mm into the open detent
SPRING_K = 30.0  # N/m, driver springs
SPRING_PRELOAD = 0.3  # N at rest (commercial pin-tumbler springs push a few tenths of a newton)
SPRING_TARGET = -SPRING_PRELOAD / SPRING_K

# Keyway (plug frame, y-z): a 9 mm slot, floor at -16 mm, roof at +8 mm, blind end at 65 mm.
KEYWAY_HALF_W = 0.0045
KEYWAY_FLOOR, KEYWAY_ROOF, KEYWAY_DEPTH = -0.016, 0.008, 0.075

# Open detent: a 2 mm deep, 9 mm wide notch where each driver sits at 90 degrees.
NOTCH_DEPTH, NOTCH_HALF = 0.002, 0.0045
THETA_OPEN_DEG = 90.0
THETA_LIMITS_DEG = (0.0, 92.0)
# Plug return spring (a cylinder tailpiece spring): torque toward 0, preload at rest + rate.
RETURN_PRELOAD = 0.040  # N m at theta 0 (outweighs the wrench's own weight, ~0.012 N m, and set-pin friction)
RETURN_STIFF_DEG = 0.00025  # N m / deg
RETURN_TARGET_DEG = -RETURN_PRELOAD / RETURN_STIFF_DEG  # -160 deg (PhysX wraps drive targets beyond +-180)

# Tension wrench: a spring-steel lever fitted to the plug face, pivoting with the plug. Its flex relative to
# the plug (a declared torsion spring) is the applied turning torque; a pointer on the plug face shows zero.
LEVER_X = FACE_X - 0.006  # lever plane
LEVER_DIR0_DEG = 45.0  # direction at theta 0: up-and-out toward -y, 45 degrees above horizontal
LEVER_R0, LEVER_L = 0.012, 0.100  # bar from radius 12 mm to the sleeve pin at 100 mm
FLEX_STIFF_DEG = 0.010  # N m / deg
FLEX_LIMIT_DEG = 25.0  # the wrench bottoms out here (hard stop)
TORQUE_CAP = 0.15  # N m: score 0 beyond this (15 degrees of flex)
SLEEVE_R, SLEEVE_L = 0.009, 0.030
SLEEVE_X = LEVER_X - 0.006 - SLEEVE_L / 2  # sleeve centre x (on a pin along -x)

# Pick: steel shaft along x with an upturned hook at the tip, a dog-leg to +y and an upright aluminium post.
PICK_SHAFT = (0.130, 0.004, 0.004)  # length x, width y, height z
PICK_HOOK = (0.004, 0.004, 0.007)  # tip tooth above the shaft
PICK_LEG_Y = 0.075  # dog-leg offset of the handle
POST = (0.010, 0.030, 0.040)  # flat upright handle plate: thickness x (gripped across), width y, height z
POST_GRIP_LOCAL = (-PICK_SHAFT[0] + PICK_SHAFT[1] / 2, PICK_LEG_Y, -PICK_SHAFT[2] / 2 + 0.027)  # grip point, pick frame
PICK_START = np.array([0.28, 0.12, TOP + 0.002])  # pick frame (shaft tip centre) lying on the table

MASS = {"plug": 0.11, "lever": 0.03, "sleeve": 0.01, "driver": 0.0098, "key": 0.0087, "pick": 0.050}
FRICTION = {"pin": (0.25, 0.2), "plug": (0.25, 0.2), "pick": (0.4, 0.35), "lever": (0.8, 0.7),
            "sleeve": (1.0, 0.9), "housing": (0.4, 0.3)}

# Scoring thresholds
SET_DRIVER_MIN = R - 0.0008  # driver bottom at or above this height above the axis
SET_KEY_MAX = R - 0.0010  # its key pin top below this (not lifted into the shear line)


def pin_world_x(i: int) -> float:
    return FACE_X + PIN_X[i]


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def quat_x(a: float) -> np.ndarray:
    return np.array([math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)])


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def angle_about_x(rot: np.ndarray) -> float:
    """Rotation about +x of a body whose frame starts aligned with the station (radians)."""
    return math.atan2(rot[2, 1], rot[1, 1])


def lever_dir(phi: float) -> np.ndarray:
    """Unit (y, z) of the lever at lever angle phi (= theta + flex)."""
    a = math.radians(LEVER_DIR0_DEG)
    d0 = np.array([-math.cos(a), math.sin(a)])
    c, s = math.cos(phi), math.sin(phi)
    return np.array([c * d0[0] - s * d0[1], s * d0[0] + c * d0[1]])


def sleeve_center(phi: float) -> np.ndarray:
    yz = lever_dir(phi) * LEVER_L
    return np.array([SLEEVE_X, yz[0], AXIS_Z + yz[1]])


def start_poses() -> dict[str, np.ndarray]:
    """Body frame poses (x y z qx qy qz qw). Plug, lever and housing frames sit on the axis."""
    ident = [0.0, 0.0, 0.0, 1.0]
    p = {"housing": np.r_[AXIS, ident], "plug": np.r_[AXIS, ident], "lever": np.r_[LEVER_X, AXIS[1:], ident],
         "sleeve": np.r_[sleeve_center(0.0), ident], "pick": np.r_[PICK_START, ident]}
    for i in range(3):
        x = pin_world_x(i)
        p[f"key{i}"] = np.r_[x, 0.0, AXIS_Z + KEY_BOTTOM0 + KEY_L / 2, ident]
        p[f"driver{i}"] = np.r_[x, 0.0, AXIS_Z + DRIVER_BOTTOM0 + DRIVER_L / 2, ident]
    return p


def binding_order() -> list[int]:
    return sorted(range(3), key=lambda i: GAP[i])


def bind_angle_deg(i: int) -> float:
    return math.degrees(GAP[i] / R)

# Coloured caps on the driver tops (visual), seen through the clear housing.
CAP = (0.011, 0.011, 0.004)

# Tension-hand grasp: fingers across the sleeve, the gripper rolled 30 deg about x and pitched 40 deg about y so
# that its wrist camera looks through the clear housing at the three pin stacks from about 14 cm.
WRENCH_GRIP_TILT_DEG = (30.0, -40.0)
