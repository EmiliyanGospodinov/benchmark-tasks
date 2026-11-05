"""Scorer state for the modular retrieval tool. Pure Python + NumPy (CPU-testable); measured state only.

Credit (best tier reached, except the final tier):
    0.2  the head has been locked on the shaft (seated, aligned, turned and release returned)
    0.4  the cartridge has been pulled out of the hood by the locked tool: it travelled >= HOOKED_TRAVEL toward the
         mouth while the head was locked and the tooth was inside its pocket, and its rear edge is now outside
    0.6  after that, the cartridge rests inside the tray
    0.8  and the tool, still locked, lies at its rest (pin within 3 cm of its start, flat on the table)
    1.0  success: all of the above with both grippers open and at least 7 cm from tool and cartridge, held 2 s
Non-finite poses make the step invalid (score 0, the window restarts).
"""
from __future__ import annotations

import math

import numpy as np

from . import design as d

HOLD_S, OPEN_MIN, CLEAR_MIN = 2.0, 0.8, 0.07
HOOKED_TRAVEL = 0.10  # m of cartridge travel toward the mouth with the locked tooth in its pocket
REST_SPEED = 0.02


def tooth_in_pocket(head_pose, cart_pose, margin=0.002):
    """The head's tooth tip lies inside the cartridge's pocket (cartridge frame)."""
    rh = d.quat_to_matrix(head_pose[3:])
    tip = np.asarray(head_pose[:3]) + rh @ np.array([(d.TOOTH_X0 + d.ARM_X1) / 2, 0.0, -d.TOOTH_DROP])
    rc = d.quat_to_matrix(cart_pose[3:])
    p = rc.T @ (tip - np.asarray(cart_pose[:3]))
    return (d.POCKET_X0 - margin <= p[0] <= d.POCKET_X1 + margin and abs(p[1]) <= d.POCKET_HALF_W + margin
            and d.CART_BASE_T - margin <= p[2] <= d.CART_H + margin)


def cart_in_tray(cart_pose):
    rc = d.quat_to_matrix(cart_pose[3:])
    footprint = np.array([[x, y, 0.0] for x in (0.0, d.CART_L)
                          for y in (-d.CART_W / 2, d.CART_W / 2)])
    corners = np.asarray(cart_pose[:3]) + footprint @ rc.T
    ix, iy = d.TRAY_INNER[0] / 2, d.TRAY_INNER[1] / 2
    # The tray floor is 3 mm above the table. Check every bottom corner, including
    # height: a centre inside the footprint alone could credit a fallen object.
    return (bool(np.all(np.abs(corners[:, 0] - d.TRAY_CENTER[0]) <= ix + 0.001))
            and bool(np.all(np.abs(corners[:, 1] - d.TRAY_CENTER[1]) <= iy + 0.001))
            and bool(np.all((corners[:, 2] >= d.TOP + 0.001) & (corners[:, 2] <= d.TOP + 0.010)))
            and rc[2, 2] > math.cos(math.radians(15)))


def tool_at_rest(shaft_pose):
    rs = d.quat_to_matrix(shaft_pose[3:])
    return (math.hypot(shaft_pose[0] - d.PIN_XY[0], shaft_pose[1] - d.PIN_XY[1]) <= d.TOOL_REST_TOL
            and abs(shaft_pose[2] - d.TOP - d.MAT_T) < 0.006 and rs[2, 2] > math.cos(math.radians(10)))


def cart_out(cart_pose):
    rc = d.quat_to_matrix(cart_pose[3:])
    # The fixed hood's outward normal is -x. Check the whole bounding box,
    # including rotated side/top corners; a rear midpoint can clear too soon.
    corners = np.array([[x, y, z] for x in (0.0, d.CART_L)
                        for y in (-d.CART_W / 2, d.CART_W / 2)
                        for z in (0.0, d.CART_BODY_H)])
    world = np.asarray(cart_pose[:3]) + corners @ rc.T
    return bool(np.all(world[:, 0] < d.HOOD_MOUTH_X))


class RetrievalScore:
    def __init__(self, control_hz):
        self.dt = 1.0 / float(control_hz)
        self.reset()

    def reset(self):
        self.best, self.window_s, self.success = 0.0, 0.0, False
        self.was_locked = self.tool_pulled = self.extracted = False
        self.hooked_travel, self._last_cart_x = 0.0, None
        self._last_engaged = False
        self.reason = "start"

    def update(self, poses, locked, grip_open, tcps, speeds):
        vals = [float(v) for p in poses.values() for v in p] + list(map(float, grip_open)) + \
               [float(v) for t in tcps for v in t] + [float(v) for v in speeds.values()]
        if not all(math.isfinite(v) for v in vals):
            self.window_s, self.reason = 0.0, "invalid state"
            self._last_cart_x, self._last_engaged = None, False
            self.success = False
            return 0.0
        shaft, head, cart = poses["shaft"], poses["head"], poses["cartridge"]
        if locked:
            self.was_locked = True
            self.best = max(self.best, 0.2)
        engaged = locked and tooth_in_pocket(head, cart)
        if self._last_cart_x is not None and not self.extracted:
            advance = self._last_cart_x - float(cart[0])
            if advance < 0.0 or (engaged and self._last_engaged):
                # Pushing the cartridge back removes progress, including while
                # unhooked. Repeated short oscillations cannot bank 10 cm credit.
                self.hooked_travel = max(0.0, self.hooked_travel + advance)
        self._last_cart_x = float(cart[0])
        self._last_engaged = engaged
        self.tool_pulled = self.extracted or self.hooked_travel >= HOOKED_TRAVEL
        if self.tool_pulled and cart_out(cart):
            self.extracted = True
            self.best = max(self.best, 0.4)
        in_tray = self.extracted and cart_in_tray(cart) and speeds["cartridge"] < REST_SPEED
        if in_tray:
            self.best = max(self.best, 0.6)
        stowed = in_tray and locked and tool_at_rest(shaft) and speeds["shaft"] < REST_SPEED
        if stowed:
            self.best = max(self.best, 0.8)
        parts = [shaft[:3], head[:3], cart[:3]]
        released = all(g >= OPEN_MIN for g in grip_open) and all(
            min(np.linalg.norm(np.asarray(t) - p) for p in parts) >= CLEAR_MIN for t in tcps)
        if stowed and released:
            self.window_s += self.dt
            self.success = self.success or self.window_s >= HOLD_S - 1e-9
        else:
            self.window_s = 0.0
        if self.success:
            self.reason = "cartridge recovered, tool stowed and released"
            return 1.0
        self.reason = ("stowed, still held" if stowed else "cartridge in tray" if in_tray else
                       "cartridge extracted" if self.extracted else "tool assembled" if self.was_locked else
                       "not assembled")
        return self.best
