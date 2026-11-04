"""Scorer state for the blade change. Pure Python (CPU-testable).

Fed once per control step with the measured state from the embodiment (holder-relative blade poses,
collar angle, pin lift, carriage extension, part speeds, gripper openings and TCP distances); never
the policy's commands or phase.

Tiers (current state, except the retention check which is a measured event):
    0.25  the old blade is out of the holder and resting in the tray
    0.50  and the replacement is seated the right way up (shoulder on the nose, notch up)
    0.75  and locked: pin down in the notch, collar back on its stop
    1.00  success: and the retention check has been passed (carriage extended by at least
          TUG_MIN_EXT for TUG_HOLD_S while seated and locked, one gripper holding the blue end
          block and the other open and clear of the mechanism), with both grippers open and away and
          every part at rest, continuously for HOLD_S
Non-finite input scores 0 for that step and restarts the windows. The untouched start scores 0.
"""

from __future__ import annotations

import math

from . import design as d

SEAT_MIN_MM, SEAT_MAX_MM = -2.5, 1.0  # shoulder along the axis: bears at -0.9; the notch gives 1 mm of play
AXIS_TOL_MM = 1.2  # tang off the axis (x, z)
ANGLE_TOL_DEG = 8.0
PIN_DOWN_MM = 1.0
COLLAR_HOME_DEG = -3.0
REST_SPEED = 0.01
OPEN_MIN = 0.8
CLEAR_MIN_M = 0.06
HOLD_S = 2.0
TUG_GRIP_DIST_M = 0.02
TUG_GRIP_OPEN_MIN, TUG_GRIP_OPEN_MAX = 0.1, 0.5


def seated(b) -> bool:
    x, y, z = b["rel_mm"]
    return (SEAT_MIN_MM <= y <= SEAT_MAX_MM and abs(x) <= AXIS_TOL_MM and abs(z) <= AXIS_TOL_MM
            and abs(b["roll_deg"]) <= ANGLE_TOL_DEG and abs(b["yaw_deg"]) <= ANGLE_TOL_DEG
            and abs(b["pitch_deg"]) <= ANGLE_TOL_DEG)


def in_tray(b) -> bool:
    x, y, z = b["world"]
    tx, ty = d.TRAY_XY
    return abs(x - tx) <= d.TRAY[0] / 2 and abs(y - ty) <= d.TRAY[1] / 2 and z <= d.TRAY[2] + 0.02


def locked(st) -> bool:
    return st["pin_lift_mm"] <= PIN_DOWN_MM and st["collar_deg"] >= COLLAR_HOME_DEG


def blade_tug_grip(st, grip_open) -> bool:
    """Measured geometric grasp proxy; a direct load on the holder is not a blade tug.

    One TCP must be at the blue blade's end block with partly closed fingers; the other
    gripper must be open and clear of the holder, collar and pin. Either arm may pull.
    This uses measured joints/poses, not policy targets or a scripted phase.
    """
    return any(
        TUG_GRIP_OPEN_MIN <= grip_open[i] <= TUG_GRIP_OPEN_MAX
        and st["new_grip_dist_m"][i] <= TUG_GRIP_DIST_M
        and grip_open[1 - i] >= OPEN_MIN
        and st["mechanism_clear_m"][1 - i] >= CLEAR_MIN_M
        for i in range(2)
    )


def _numbers(st):
    yield from (st["carriage_ext_mm"], st["collar_deg"], st["pin_lift_mm"])
    for n in ("old_blade", "new_blade"):
        b = st[n]
        yield from (*b["rel_mm"], b["roll_deg"], b["yaw_deg"], b["pitch_deg"], *b["world"])
    yield from st["speeds"].values()
    yield from st["new_grip_dist_m"]
    yield from st["mechanism_clear_m"]


class BladeScore:
    def __init__(self, control_hz: float):
        self.dt = 1.0 / float(control_hz)
        self.tug_s, self.tug_passed, self.window_s, self.success, self.reason = 0.0, False, 0.0, False, "start"

    def update(self, st, grip_open, tcp_clear) -> float:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [*_numbers(st), *grip_open, *tcp_clear]):
            self.tug_s = self.window_s = 0.0
            self.tug_passed = False
            self.reason = "invalid state"
            return 0.0
        stowed = in_tray(st["old_blade"]) and not seated(st["old_blade"])
        new_ok = stowed and seated(st["new_blade"])
        is_locked = new_ok and locked(st)
        if not is_locked:
            self.tug_passed = False  # a later installation must receive its own retention check
        if is_locked and blade_tug_grip(st, grip_open) and st["carriage_ext_mm"] / 1000 >= d.TUG_MIN_EXT:
            self.tug_s += self.dt
            if self.tug_s >= d.TUG_HOLD_S - 1e-9:
                self.tug_passed = True
        else:
            self.tug_s = 0.0
        released = all(g >= OPEN_MIN for g in grip_open) and all(c >= CLEAR_MIN_M for c in tcp_clear)
        at_rest = all(v <= REST_SPEED for v in st["speeds"].values())
        if is_locked and self.tug_passed and released and at_rest:
            self.window_s += self.dt
            if self.window_s >= HOLD_S - 1e-9:
                self.success = True
        else:
            self.window_s = 0.0
        if self.success:
            self.reason = "replaced, locked and retention-checked"
            return 1.0
        score = 0.75 if is_locked else 0.5 if new_ok else 0.25 if stowed else 0.0
        self.reason = ("locked, tug passed" if is_locked and self.tug_passed else "locked" if is_locked else
                       "seated" if new_ok else "old blade stowed" if stowed else "not done")
        return score
