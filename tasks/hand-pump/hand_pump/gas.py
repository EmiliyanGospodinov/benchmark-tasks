"""Quasistatic isothermal gas model with ideal inlet and outlet check valves.

Q=P*V is proportional to gas quantity at fixed temperature. Gas delivery follows
measured piston volume, not action count or policy phase. The receiver is fixed
volume; hose dead volume, leakage and temperature transients are not modeled.
"""
from dataclasses import dataclass
import math


@dataclass
class PumpGas:
    area: float
    dead_volume: float
    stroke: float
    receiver_volume: float
    atmospheric: float = 101325.0

    def __post_init__(self):
        if min(self.area, self.dead_volume, self.stroke, self.receiver_volume, self.atmospheric) <= 0:
            raise ValueError("Pump dimensions and atmospheric pressure must be positive")
        self.reset()

    def reset(self, extension=None):
        self.extension = self.stroke if extension is None else self._valid_extension(extension)
        self.volume = self.dead_volume + self.area * self.extension
        self.cylinder_q = self.atmospheric * self.volume
        self.receiver_q = self.atmospheric * self.receiver_volume
        self.delivered_q = 0.0

    def _valid_extension(self, x):
        x = float(x)
        if not math.isfinite(x) or x < -0.002 or x > self.stroke + 0.002:
            raise ValueError(f"Piston escaped physical travel: {x}")
        return min(self.stroke, max(0.0, x))

    @property
    def cylinder_pressure(self):
        return self.cylinder_q / self.volume

    @property
    def receiver_pressure(self):
        return self.receiver_q / self.receiver_volume

    @property
    def gauge_kpa(self):
        return (self.receiver_pressure - self.atmospheric) / 1000.0

    @property
    def outward_force(self):
        return (self.cylinder_pressure - self.atmospheric) * self.area

    def update(self, extension):
        self.extension = self._valid_extension(extension)
        self.volume = self.dead_volume + self.area * self.extension
        # Inlet draws atmosphere only when the cylinder pressure falls below it.
        self.cylinder_q = max(self.cylinder_q, self.atmospheric * self.volume)
        if self.cylinder_pressure > self.receiver_pressure:
            # Outlet equalizes the cylinder and receiver without losing gas.
            total_q = self.cylinder_q + self.receiver_q
            pressure = total_q / (self.volume + self.receiver_volume)
            receiver_q = pressure * self.receiver_volume
            self.delivered_q += max(0.0, receiver_q - self.receiver_q)
            self.receiver_q = receiver_q
            self.cylinder_q = pressure * self.volume
        return self.outward_force
