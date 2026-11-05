"""Scorer state for the ratchet lowering trainer. Pure Python (CPU-testable).

Fed once per control step (20 Hz) with measured state only: the lever angle THETA (degrees above
horizontal, from the rotor's pose; a protractor on a real trainer), its angular speed, the grippers' measured
openings and each TCP's distance to the nearest of crank-handle centre and pawl-tab centre.

    released      both grippers at least 80 % open and both TCPs at least 5 cm from the handle and tab
    still         lever speed under 1 deg/s
    intermediate  THETA inside HOLD_BAND (the pawl-supported middle tooth; nothing else supports it there)
    landed        THETA within REST_TOL of the rest angle (the weight on its receiving rest)

Credit (latched stages, in order):
    0.25  the lever has been lowered at least 10 degrees below its start
    0.50  released and still in the intermediate band for HOLD_S (the pawl holds the load, hands clear)
    0.75  after that hold, the lever has been lowered again below -10 degrees
    1.00  success: after the hold, landed, released and still for HOLD_S; the trial ends.
Zero, latched, if the lever ever moves faster than SPEED_LIMIT (a dropped or runaway load) or any input is
non-finite. Landing without the intermediate released hold earns at most 0.25.
"""

from __future__ import annotations

import math

START_DEG = 35.3
HOLD_BAND = (1.0, 10.0)
REST_DEG, REST_TOL = -28.0, 1.5
SPEED_LIMIT = 40.0
STILL = 1.0
HOLD_S = 2.0
LOWERED_DEG = 10.0
RESTART_DEG = -10.0
OPEN_MIN = 0.8
CLEAR_MIN_M = 0.05
MECHANISM_CLEAR_MIN_M = 0.005
PAWL_ENGAGED_BAND = (2.0, 5.0)  # middle tooth's seated pawl is about +3.2 degrees


class RatchetScore:
    def __init__(self, control_hz: float):
        self.dt = 1.0 / float(control_hz)
        self.reset()

    def reset(self):
        self.failed, self.lowered, self.held, self.restarted, self.success = False, False, False, False, False
        self.window_s, self.reason = 0.0, "start"

    def info(self) -> dict:
        return {"success": self.success, "score_reason": self.reason, "hold_window_s": self.window_s,
                "held_on_pawl": self.held, "runaway": self.failed}

    def update(self, theta, omega, pawl, grip_open, tcp_clear, mechanism_clear_m) -> float:
        values = [theta, omega, pawl, *grip_open, *tcp_clear, mechanism_clear_m]
        if self.failed or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            self.failed, self.reason, self.window_s = True, self.reason if self.failed else "invalid state", 0.0
            return 0.0
        if abs(omega) > SPEED_LIMIT:
            self.failed, self.reason = True, "runaway or dropped load"
            return 0.0
        released = (all(g >= OPEN_MIN for g in grip_open) and all(c >= CLEAR_MIN_M for c in tcp_clear)
                    and mechanism_clear_m >= MECHANISM_CLEAR_MIN_M)
        still = abs(omega) < STILL
        if theta <= START_DEG - LOWERED_DEG:
            self.lowered = True
        condition = False
        if not self.held:
            condition = (released and still and HOLD_BAND[0] <= theta <= HOLD_BAND[1]
                         and PAWL_ENGAGED_BAND[0] <= pawl <= PAWL_ENGAGED_BAND[1])
        elif not self.success:
            if theta < RESTART_DEG:
                self.restarted = True
            condition = released and still and abs(theta - REST_DEG) <= REST_TOL
        self.window_s = self.window_s + self.dt if condition else 0.0
        if condition and self.window_s >= HOLD_S - 1e-9:
            if not self.held:
                self.held, self.window_s = True, 0.0
            else:
                self.success = True
        if self.success:
            self.reason = "landed on the rest after a released intermediate hold"
            return 1.0
        if self.held:
            self.reason = "restarted after the hold" if self.restarted else "held on the pawl, hands clear"
            return 0.75 if self.restarted else 0.5
        self.reason = "lowered under control" if self.lowered else "not lowered"
        return 0.25 if self.lowered else 0.0
