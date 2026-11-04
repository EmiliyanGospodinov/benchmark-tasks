"""On the station table: an open cardboard carton between the arms, and six plain boxes (products) beside it.
The carton is fixed to the table; its opening is 24 cm deep (away from the arms), 18 cm wide and 6.5 cm tall
inside. Four of the boxes go into the carton, packed tightly as the packing plan shows; the other two stay out.

Products (loose, rigid, painted cardboard; x (depth) by y (width) by z (height)):
- blue box 16 x 6 x 3.5 cm, 0.20 kg, in front of the left arm
- red box 7 x 6 x 3.5 cm, 0.08 kg, in front of the left arm, far
- yellow box 9 x 6 x 2.5 cm, 0.06 kg, in front of the left arm, near the carton
- purple box 7 x 6 x 5 cm, 0.12 kg, beside the carton on the left, near
- green box 12 x 5.5 x 3.5 cm, 0.15 kg, in front of the right arm
- orange box 7 x 6 x 5 cm, 0.12 kg, in front of the right arm, far

The packing plan is printed on two sheets lying on the table behind the carton, at half scale and in the carton's
orientation (the far edge of a sheet is the carton's far wall): the left sheet shows the bottom layer, the right
sheet the top layer, each box as a rectangle in its colour inside the outline of the carton's opening. Boxes not on
the sheets stay out of the carton. The plan specifies 3 mm clearances at adjacent bottom-layer box faces and
the far/left reference walls; the yellow box rests directly on the blue box.

Run the scripted demo:  inspect-robots run --embodiment yam_arena --policy yam_scripted \\
    -E task=tasks/box_packing/box_packing.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 \\
    -E cam_height=176 --instruction "pack the carton as the packing plan shows" --scorer task_score \\
    --max-steps 2400 --store-frames --log-dir runs/box-packing-demo
"""

import numpy as np

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# -- carton (fixed to the table) -----------------------------------------------------------------------------
CARTON_XY = (0.26, 0.0)
INNER = (0.24, 0.18, 0.065)  # x (depth), y (width), z (height) inside
WALL, FLOOR = 0.008, 0.006
CARDBOARD = (0.76, 0.60, 0.42)
FLOOR_TOP = TOP + FLOOR
CX, CY = CARTON_XY
IX, IY, IZ = INNER
_wall_z, _wall_h = TOP + (FLOOR + IZ) / 2, FLOOR + IZ
CARTON = [
    Box("carton_floor", pos=(CX, CY, TOP + FLOOR / 2), size=(IX + 2 * WALL, IY + 2 * WALL, FLOOR), color=CARDBOARD),
    Box("carton_far", pos=(CX + IX / 2 + WALL / 2, CY, _wall_z), size=(WALL, IY + 2 * WALL, _wall_h), color=CARDBOARD),
    Box("carton_near", pos=(CX - IX / 2 - WALL / 2, CY, _wall_z), size=(WALL, IY + 2 * WALL, _wall_h), color=CARDBOARD),
    Box("carton_left", pos=(CX, CY + IY / 2 + WALL / 2, _wall_z), size=(IX, WALL, _wall_h), color=CARDBOARD),
    Box("carton_right", pos=(CX, CY - IY / 2 - WALL / 2, _wall_z), size=(IX, WALL, _wall_h), color=CARDBOARD),
]
INNER_LO = np.array([CX - IX / 2, CY - IY / 2, FLOOR_TOP])
INNER_HI = np.array([CX + IX / 2, CY + IY / 2, FLOOR_TOP + IZ])


# -- products --------------------------------------------------------------------------------------------------
def _product(name, xy, size, mass, color):
    return Box(name, pos=(xy[0], xy[1], TOP + size[2] / 2), size=size, mass=mass, color=color)


PRODUCTS = [
    _product("blue_box", (0.26, 0.27), (0.16, 0.06, 0.035), 0.20, (0.15, 0.30, 0.85)),
    _product("red_box", (0.42, 0.29), (0.07, 0.06, 0.035), 0.08, (0.85, 0.12, 0.12)),
    _product("yellow_box", (0.36, 0.19), (0.09, 0.06, 0.025), 0.06, (0.95, 0.80, 0.10)),
    _product("purple_box", (0.10, 0.17), (0.07, 0.06, 0.05), 0.12, (0.55, 0.25, 0.70)),
    _product("green_box", (0.26, -0.27), (0.12, 0.055, 0.035), 0.15, (0.10, 0.65, 0.20)),
    _product("orange_box", (0.42, -0.25), (0.07, 0.06, 0.05), 0.12, (0.95, 0.50, 0.10)),
]
OBJECTS = [*CARTON, *PRODUCTS]
START = {o.name: np.array(o.pos) for o in PRODUCTS}
SIZE = {o.name: np.array(o.size) for o in PRODUCTS}

# -- the packing: each packed box's goal centre (world), tight into the far-left corner ------------------------
GAP = 0.003
_far, _left = INNER_HI[0] - GAP, INNER_HI[1] - GAP  # far and left faces of the corner boxes


def _goal(far_face, left_face, bottom, name):
    x, y, z = SIZE[name]
    return np.array([far_face - x / 2, left_face - y / 2, bottom + z / 2])


GOALS = {"blue_box": _goal(_far, _left, FLOOR_TOP, "blue_box")}
GOALS["green_box"] = _goal(_far, GOALS["blue_box"][1] - SIZE["blue_box"][1] / 2 - GAP, FLOOR_TOP, "green_box")
GOALS["red_box"] = _goal(GOALS["blue_box"][0] - SIZE["blue_box"][0] / 2 - GAP, _left, FLOOR_TOP, "red_box")
GOALS["yellow_box"] = _goal(_far, _left, FLOOR_TOP + SIZE["blue_box"][2], "yellow_box")
SUPPORT = {"blue_box": FLOOR_TOP, "green_box": FLOOR_TOP, "red_box": FLOOR_TOP}
SUPPORT["yellow_box"] = FLOOR_TOP + SIZE["blue_box"][2]
DISTRACTORS = ("purple_box", "orange_box")

# -- the printed packing plan: one sheet per layer on the table behind the carton, bottom layer on the left -----
# Each sheet shows the carton's opening (dark outline) and every box of that layer as a coloured rectangle, at half
# scale and in the same orientation as the carton (far edge of the sheet = far wall). Thin fixed boxes, 1 mm thick.
PLAN_SCALE = 0.5
PLAN_X = CX + IX / 2 + WALL + 0.09  # sheet centres: 9 cm behind the carton's far wall ...
PLAN_YS = (0.08, -0.08)  # ... left sheet: bottom layer, right sheet: top layer
PAPER, INK = (0.95, 0.95, 0.93), (0.15, 0.15, 0.15)
SHEET_MARGIN, LINE = 0.015, 0.003
LAYERS = [
    [n for n in GOALS if SUPPORT[n] == FLOOR_TOP],
    [n for n in GOALS if SUPPORT[n] > FLOOR_TOP],
]


def _plan_sheet(i, names):
    """Sheet ``i`` (0 = bottom layer) with the carton outline and the boxes ``names``."""
    cy, s, t = PLAN_YS[i], PLAN_SCALE, 0.001
    ox, oy = IX * s, IY * s  # the opening, scaled
    parts = [
        Box(f"plan{i}_sheet", pos=(PLAN_X, cy, TOP + t / 2),
            size=(ox + 2 * SHEET_MARGIN, oy + 2 * SHEET_MARGIN, t), color=PAPER),  # fmt: skip
        Box(f"plan{i}_far", pos=(PLAN_X + ox / 2 + LINE / 2, cy, TOP + 1.5 * t), size=(LINE, oy + 2 * LINE, t), color=INK),
        Box(f"plan{i}_near", pos=(PLAN_X - ox / 2 - LINE / 2, cy, TOP + 1.5 * t), size=(LINE, oy + 2 * LINE, t), color=INK),
        Box(f"plan{i}_left", pos=(PLAN_X, cy + oy / 2 + LINE / 2, TOP + 1.5 * t), size=(ox, LINE, t), color=INK),
        Box(f"plan{i}_right", pos=(PLAN_X, cy - oy / 2 - LINE / 2, TOP + 1.5 * t), size=(ox, LINE, t), color=INK),
    ]
    color = {o.name: o.color for o in PRODUCTS}
    for n in names:
        gx, gy, _ = GOALS[n]
        sx, sy, _ = SIZE[n]
        parts.append(
            Box(f"plan{i}_{n}", pos=(PLAN_X + (gx - CX) * s, cy + (gy - CY) * s, TOP + 1.5 * t),
                size=(sx * s, sy * s, t), color=color[n])  # fmt: skip
        )
    return parts


PLAN = [part for i, names in enumerate(LAYERS) for part in _plan_sheet(i, names)]
OBJECTS = [*OBJECTS, *PLAN]
TOLERANCE = 0.005  # every face of a packed box's bounding box within 5 mm of where the packing puts it


def placed(name, s) -> bool:
    goal, half = GOALS[name], SIZE[name] / 2
    return bool(np.all(np.abs(s.low - (goal - half)) < TOLERANCE) and np.all(np.abs(s.high - (goal + half)) < TOLERANCE))


def in_carton(s) -> bool:
    """Any part of the box is inside the carton's opening, below its rim."""
    return bool(np.all(s.high > INNER_LO) and np.all(s.low < INNER_HI))


def success(objects):
    """Every packed box in place and still, both distractors out of the carton."""
    return all(placed(n, objects[n]) and objects[n].resting() for n in GOALS) and not any(
        in_carton(objects[n]) for n in DISTRACTORS
    )


_calls = [0]


def _debug(objects):
    """With BOX_PACKING_DEBUG=N set, print every N-th step how far each packed box's faces are from its goal (mm)."""
    import os

    every = int(os.environ.get("BOX_PACKING_DEBUG", "0") or 0)
    _calls[0] += 1
    if every and _calls[0] % every == 0:
        rows = []
        for n in GOALS:
            s, goal, half = objects[n], GOALS[n], SIZE[n] / 2
            lo, hi = (s.low - (goal - half)) * 1000, (s.high - (goal + half)) * 1000
            rows.append(f"{n.split('_')[0]} lo{np.round(lo, 1)} hi{np.round(hi, 1)}")
        print(f"[box_packing step {_calls[0]}] " + " | ".join(rows), flush=True)


def score(objects):
    """0.25 per packed box in place; 0 if a distractor is in the carton."""
    _debug(objects)
    if any(in_carton(objects[n]) for n in DISTRACTORS):
        return 0.0
    return sum(placed(n, objects[n]) for n in GOALS) / len(GOALS)


def distance(objects):
    """Summed distance (m) of the packed boxes from their goals."""
    return float(sum(np.linalg.norm(objects[n].pos - GOALS[n]) for n in GOALS))


# -- demo: place each box offset towards open space (room for the fingers), then push it home: left, then far ---
# The closed fingertips are 4 cm along x and 2 cm along y up to 2.5 cm above the TCP, wider above; pushing with the
# TCP 2 cm above the support keeps the wider parts above the 3.5 cm boxes and the 6.5 cm walls; boxes are set down
# at most 15 mm from their goal so the pusher's 4.6 cm wide middle section stays clear of the walls.
TRAVEL_Z = TOP + 0.16
CLEAR_Z = FLOOR_TOP + IZ + 0.03  # TCP height that clears the walls and every packed box
PUSH_ABOVE = 0.020  # TCP above the support while pushing: the fingertips meet a box near mid-height
TIP_Y, TIP_X = 0.006, 0.015  # closed fingertips' half extents (grip yaw 0: fingers close along y)
OFFSET = (0.010, 0.015)  # where a box is let go: this much nearer and further right than its goal
RELEASE_GAP = 0.004  # let go this far above the support
SLACK = 0.004  # the fingers open only this much wider than the box to let go, and back off this much after a push
OVERSHOOT = 0.030  # pushes are commanded this far past the goal: the arm gives up to about 2 cm under a sliding box,
# and every push ends against a wall or a packed box


def _wp(t, pos, finger, label):
    return Waypoint(t, tuple(float(v) for v in pos), finger, label)


def pack(name, push_far=True):
    """Pick ``name``, set it down offset from its goal, then push it left and (``push_far``) far into place.
    A box with no room in front of it for a far push is set down at its goal depth and only pushed left."""
    (px, py, pz), (gx, gy, gz) = START[name], GOALS[name]
    sx, sy, _ = SIZE[name]
    qx, qy, qz = gx - (OFFSET[0] if push_far else 0.0), gy - OFFSET[1], gz + RELEASE_GAP
    let_go = -(sy / 2 + SLACK)  # finger joint target: the jaws just wider than the box
    zp = SUPPORT[name] + PUSH_ABOVE
    ys, yf = qy - sy / 2 - TIP_Y - 0.006, gy - sy / 2 - TIP_Y + OVERSHOOT
    xs, xf = qx - sx / 2 - TIP_X - 0.006, gx - sx / 2 - TIP_X + OVERSHOOT
    plan = [
        _wp(1.5, (px, py, TRAVEL_Z), FINGER_OPEN, f"{name}: to it"),
        _wp(1.5, (px, py, pz), FINGER_OPEN, "descend"),
        _wp(0.8, (px, py, pz), FINGER_CLOSED, "grip"),
        _wp(1.2, (px, py, TRAVEL_Z), FINGER_CLOSED, "lift"),
        _wp(1.8, (qx, qy, TRAVEL_Z), FINGER_CLOSED, "carry"),
        _wp(1.5, (qx, qy, qz), FINGER_CLOSED, "lower"),
        _wp(0.6, (qx, qy, qz), let_go, "let go"),
        _wp(0.8, (qx, qy, CLEAR_Z), let_go, "up"),
        _wp(0.4, (qx, qy, CLEAR_Z), FINGER_CLOSED, "close"),
        _wp(1.0, (qx, ys, CLEAR_Z), FINGER_CLOSED, "to its right side"),
        _wp(1.0, (qx, ys, zp), FINGER_CLOSED, "down"),
        _wp(2.0, (qx, yf, zp), FINGER_CLOSED, "push left"),
        _wp(0.5, (qx, yf - OVERSHOOT, zp), FINGER_CLOSED, "back off"),
        _wp(0.8, (qx, yf - OVERSHOOT, CLEAR_Z), FINGER_CLOSED, "up"),
    ]
    if push_far:
        plan += [
            _wp(1.0, (xs, gy, CLEAR_Z), FINGER_CLOSED, "to its near side"),
            _wp(1.0, (xs, gy, zp), FINGER_CLOSED, "down"),
            _wp(2.0, (xf, gy, zp), FINGER_CLOSED, "push far"),
            _wp(0.5, (xf - OVERSHOOT, gy, zp), FINGER_CLOSED, "back off"),
        ]
    last = plan[-1].pos
    return plan + [_wp(1.0, (last[0], last[1], TRAVEL_Z), FINGER_CLOSED, "up")]


def park(arm):
    """Over the arm's own side of the table, clear of the carton and the other arm."""
    return [_wp(2.0, (0.20, 0.30 if arm == "left" else -0.30, TRAVEL_Z), FINGER_OPEN, "park")]


def wait(arm, seconds):
    return [_wp(seconds, (0.20, 0.30 if arm == "left" else -0.30, TRAVEL_Z), FINGER_OPEN, "wait")]


def _duration(waypoints):
    return sum(w.duration for w in waypoints)


def demo():
    """One arm at a time: left packs blue, right packs green, left packs red and yellow."""
    left1 = pack("blue_box") + park("left")
    right = pack("green_box") + park("right")
    left2 = pack("red_box", push_far=False) + pack("yellow_box") + park("left")
    return {
        "left": left1 + wait("left", _duration(right)) + left2,
        "right": wait("right", _duration(left1)) + right,
    }
