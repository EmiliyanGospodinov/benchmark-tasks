"""Scorer for the clear pin-tumbler trainer. Pure Python (CPU-testable).

Fed once per control step (20 Hz) with measured state only:
  theta        plug angle, degrees (0 at rest, open mark at 90); a protractor/marks on the trainer face
  flex         tension-wrench flex relative to the plug, degrees; torque = flex x 0.010 N m/deg (face pointer)
  driver_bottom, key_top   per pin stack, metres above the plug axis (seen through the clear housing)
  pick_tip     the pick's tip position (is it out of the keyway)
  grip_open, tcp_clear     grippers' opening fractions; each TCP's distance to the wrench sleeve / keyway mouth
  housing_moved            displacement of the clamped trainer body, metres

A pin is SET when its driver's bottom is at the shear line (19.2..20.8 mm above the axis, i.e. within 0.8 mm of it
where the round plug surface carries it) while its key pin's top is below the shear line (<= 19.0 mm): a visible
gap separates the stack at the shear line, the driver resting on the plug ledge, not on the pick.

Credit:
  0.25 x (most pins set at the same time, held for 0.5 s)       partial, latched
  1.0  success: plug at the open mark (theta >= 85 deg), wrench unloaded (|flex| <= 3 deg; its own weight flexes it up to ~2 deg), pick out of the
       keyway (tip 1 cm in front of the face or 3 cm from the axis), both grippers >= 80 % open and >= 7 cm from the sleeve
       and the keyway mouth, all for 2 s. The trial ends.
Zero, latched: wrench flex beyond 15 deg (0.15 N m, forcing the plug), housing moved more than 2 mm, or any
non-finite input. The first crossing beyond the 15-degree binding region must have all three pins set;
after that, pins may drop into the open detents. Returning to the binding region requires a fresh crossing.
"""

from __future__ import annotations

import math

from . import design as d

SET_HOLD_S = 0.5
OPEN_DEG = 85.0
BINDING_LIMIT_DEG = 15.0
SET_DRIVER_MAX = d.R + (d.R - d.SET_DRIVER_MIN)
FLEX_CAP_DEG = d.TORQUE_CAP / d.FLEX_STIFF_DEG  # 15
FLEX_RELEASED_DEG = 3.0  # the unheld wrench sags up to ~2 deg under its own weight
OPEN_MIN = 0.8
CLEAR_MIN_M = 0.07
PICK_OUT_X = d.FACE_X - 0.01
PICK_OUT_R = 0.03
HOUSING_TOL_M = 0.002
SUCCESS_HOLD_S = 2.0


def pin_set(driver_bottom: float, key_top: float) -> bool:
    return d.SET_DRIVER_MIN <= driver_bottom <= SET_DRIVER_MAX and key_top <= d.SET_KEY_MAX


def pick_out(tip) -> bool:
    """The pick's tip is out of the keyway: 1 cm in front of the face, or 3 cm or more from the plug axis."""
    x, y, z = tip
    return x <= PICK_OUT_X or math.hypot(y - d.AXIS[1], z - d.AXIS[2]) >= PICK_OUT_R


class LockScore:
    def __init__(self, control_hz: float):
        self.dt = 1.0 / float(control_hz)
        self.reset()

    def reset(self):
        self.failed, self.success = False, False
        self.best, self.count, self.count_s, self.open_s = 0, 0, 0.0, 0.0
        self.max_flex = 0.0
        self.cleared_binding = False
        self.reason = "start"

    def info(self) -> dict:
        return {"success": self.success, "score_reason": self.reason, "pins_set": self.count,
                "pins_set_best": self.best, "open_window_s": self.open_s, "max_flex_deg": self.max_flex,
                "failed": self.failed, "cleared_binding": self.cleared_binding}

    def update(self, theta, flex, driver_bottom, key_top, pick_tip, grip_open, tcp_clear, housing_moved) -> float:
        if (len(driver_bottom), len(key_top), len(pick_tip), len(grip_open), len(tcp_clear)) != (3, 3, 3, 2, 2):
            self.failed, self.success, self.reason = True, False, "invalid state dimensions"
            return 0.0
        values = [theta, flex, *driver_bottom, *key_top, *pick_tip, *grip_open, *tcp_clear, housing_moved]
        if self.failed or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            if not self.failed:
                self.failed, self.success, self.reason = True, False, "invalid state"
            return 0.0
        self.max_flex = max(self.max_flex, abs(flex))
        if abs(flex) > FLEX_CAP_DEG:
            self.failed, self.success, self.reason = True, False, "forced: turning torque above 0.15 N m"
            return 0.0
        if housing_moved > HOUSING_TOL_M:
            self.failed, self.success, self.reason = True, False, "trainer body moved"
            return 0.0
        n = sum(pin_set(b, k) for b, k in zip(driver_bottom, key_top))
        if theta < -BINDING_LIMIT_DEG or (theta > BINDING_LIMIT_DEG and not self.cleared_binding and n != 3):
            self.failed, self.success, self.reason = True, False, "plug bypassed unset pins"
            return 0.0
        # Evaluate only the exit from binding. In the open detents the drivers
        # legitimately sit below the shear line, so n is no longer three.
        self.cleared_binding = theta > BINDING_LIMIT_DEG
        if n == self.count:
            self.count_s += self.dt
        else:
            self.count, self.count_s = n, self.dt
        if self.count_s >= SET_HOLD_S - 1e-9:
            self.best = max(self.best, self.count)
        released = (all(g >= OPEN_MIN for g in grip_open) and all(c >= CLEAR_MIN_M for c in tcp_clear))
        open_ok = (theta >= OPEN_DEG and abs(flex) <= FLEX_RELEASED_DEG and pick_out(pick_tip) and released)
        self.open_s = self.open_s + self.dt if open_ok else 0.0
        if open_ok and self.open_s >= SUCCESS_HOLD_S - 1e-9:
            self.success = True
        if self.success:
            self.reason = "open, tools out, hands clear"
            return 1.0
        self.reason = f"{self.best} pin(s) set" if self.best else ("plug turned" if theta >= OPEN_DEG else "no pin set")
        return 0.25 * self.best
