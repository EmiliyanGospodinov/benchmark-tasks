"""Scorer state for the hand pump. Pure Python (CPU-testable).

Fed once per control step with measured state only:
    gauge_kpa      the test chamber's pressure from the gas model (what the dial shows)
    grip_open      each gripper's measured opening, 0 closed .. 1 open
    tcp_clear_m    each TCP's distance to the nearest of barrel centre and handle centre
    pump_rest      the pump is put down on the mat: both feet's footprints on the mat, barrel axis within 15 mm
                   of its resting height and slower than 2 cm/s
Credit:
    0.25  the chamber has reached at least REACH_KPA
    0.50  the chamber has been inside the band (TARGET +- BAND)
    0.75  the chamber is inside the band now
    1.00  success: inside the band, pump put down on the mat and released (both grippers open and withdrawn),
          continuously for HOLD_S. Ever exceeding OVER_KPA (over-pressurising the chamber) caps the score at 0.25.
Any non-finite input makes the step invalid: score 0 and the hold window restarts.
"""
from __future__ import annotations

import math

TARGET_KPA, BAND_KPA, OVER_KPA, REACH_KPA = 60.0, 6.0, 80.0, 20.0
HOLD_S = 2.0
OPEN_MIN = 0.8
CLEAR_MIN_M = 0.07


class PumpScore:
    def __init__(self, control_hz: float):
        self.dt = 1.0 / float(control_hz)
        self.reset()

    def reset(self):
        self.best, self.window_s, self.success, self.over = 0.0, 0.0, False, False
        self.reason = "start"

    def update(self, gauge_kpa, grip_open, tcp_clear_m, pump_rest) -> float:
        values = [gauge_kpa, *grip_open, *tcp_clear_m]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            self.window_s, self.reason = 0.0, "invalid state"
            return 0.0
        self.over = self.over or gauge_kpa > OVER_KPA
        in_band = abs(gauge_kpa - TARGET_KPA) <= BAND_KPA
        if gauge_kpa >= REACH_KPA:
            self.best = max(self.best, 0.25)
        if in_band:
            self.best = max(self.best, 0.5)
        released = all(g >= OPEN_MIN for g in grip_open) and all(c >= CLEAR_MIN_M for c in tcp_clear_m)
        if in_band and released and bool(pump_rest) and not self.over:
            self.window_s += self.dt
            if self.window_s >= HOLD_S - 1e-9:
                self.success = True
        else:
            self.window_s = 0.0
        if self.over:
            self.reason = "over-pressurised"
            return min(self.best, 0.25)
        if self.success:
            self.reason = "pressure set and retained after release"
            return 1.0
        score = max(self.best, 0.75 if in_band else 0.0)
        self.reason = ("in band" if in_band else "has been in band" if self.best >= 0.5 else
                       "pressurising" if self.best > 0 else "below 30 kPa")
        return score
