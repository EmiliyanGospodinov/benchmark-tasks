"""Scorer state for the inspection mount. Pure Python + NumPy (CPU-testable).

Fed once per control step with the measured state (never the policy's commands or phase):

    elevation_deg   actual head elevation (degrees above horizontal, from the arm body's pose)
    locked          actual clamp state (rosette meshed, from the knob's measured rotation)
    grip_open       each gripper's measured opening, 0 closed .. 1 open
    tcp_clear_m     each TCP's distance to the nearest of head centre and knob centre

Credit (the score is the best tier reached, except tier 3 which needs the current state):
    0.25  the head has been raised within APPROACH_DEG of the target
    0.50  the head has been within AIM_TOL_DEG of the target
    0.75  the clamp is locked now with the head within AIM_TOL_DEG
    1.00  success: locked, within AIM_TOL_DEG, both grippers open and withdrawn, continuously for
          HOLD_S, with the elevation staying within DRIFT_DEG of its value when that window began.
Any non-finite input makes the step invalid: score 0 for that step and the hold window restarts.
"""

from __future__ import annotations

import math

TARGET_ELEV_DEG = 15.0
AIM_TOL_DEG = 3.0
APPROACH_DEG = 10.0
DRIFT_DEG = 2.0
HOLD_S = 2.0
OPEN_MIN = 0.8  # gripper opening fraction counted as released
CLEAR_MIN_M = 0.07  # TCP to head/knob centre: the fingers are off the parts


class MountScore:
    def __init__(self, control_hz: float):
        self.dt = 1.0 / float(control_hz)
        self.reset()

    def reset(self):
        self.best = 0.0
        self.window_s = 0.0
        self.window_start = None
        self.success = False
        self.reason = "start"

    def update(self, elevation_deg, locked, grip_open, tcp_clear_m) -> float:
        values = [elevation_deg, *grip_open, *tcp_clear_m]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            self.window_s, self.window_start, self.reason = 0.0, None, "invalid state"
            return 0.0
        err = abs(elevation_deg - TARGET_ELEV_DEG)
        if err <= APPROACH_DEG:
            self.best = max(self.best, 0.25)
        if err <= AIM_TOL_DEG:
            self.best = max(self.best, 0.5)
        at_aim_locked = bool(locked) and err <= AIM_TOL_DEG
        released = all(g >= OPEN_MIN for g in grip_open) and all(c >= CLEAR_MIN_M for c in tcp_clear_m)
        if at_aim_locked and released:
            if self.window_start is None:
                self.window_start, self.window_s = elevation_deg, 0.0
            elif abs(elevation_deg - self.window_start) > DRIFT_DEG:
                self.window_start, self.window_s = elevation_deg, 0.0  # drifted: restart the window
            else:
                self.window_s += self.dt
            if self.window_s >= HOLD_S - 1e-9:
                self.success = True
        else:
            self.window_s, self.window_start = 0.0, None
        if self.success:
            self.reason = "aim retained after release"
            return 1.0
        score = max(self.best, 0.75 if at_aim_locked else 0.0)
        self.reason = ("locked at the aim" if at_aim_locked else
                       "reached the aim" if self.best >= 0.5 else
                       "approached the aim" if self.best > 0 else "not raised")
        return score
