"""Nominal teaching-fixture geometry, SI units; no calibrated bearing claims."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Fit:
    initial_play: float = .012
    target_min: float = .002
    target_max: float = .004
    spacer_thicknesses: tuple = (.006, .009, .014)

    def choose(self, measured_play):
        if not math.isfinite(measured_play):
            raise ValueError("nonfinite measurement")
        options = [t for t in self.spacer_thicknesses
                   if self.target_min - 1e-9 <= measured_play-t <= self.target_max + 1e-9]
        if len(options) != 1:
            raise ValueError("measurement does not select exactly one spacer")
        return options[0]

    def residual(self, thickness):
        return self.initial_play - thickness

SPEC = Fit()
AXIS_Z = .10
SHAFT_RADIUS = .008
STOP_RADIUS = .028
STOP_THICKNESS = .008
# Sliding collar occupies x in [0, .008] at the near stop.
# Rear stop inner face is at .020; a .009 spacer reduces travel to .003.
REAR_FACE = STOP_THICKNESS + SPEC.initial_play

# Standard station measurement prototype (metres).
import numpy as np
TOP = .7475
ORIGIN = np.array([.24, 0., TOP + .075])
GAUGE_ORIGIN = ORIGIN + np.array([-.14, 0., 0.])
def start_poses():
    out = {n: np.r_[ORIGIN, 0.,0.,0.,1.] for n in ("housing","shaft","retainer")}
    # The long indicator bridge rests on the housing foot. Start in its supported
    # pose, rather than horizontally interpenetrating that foot and settling sideways.
    angle=math.radians(9.)
    quat=np.array([0.,-math.sin(angle/2),0.,math.cos(angle/2)])
    pos=np.array([.06665,0.,TOP+.0175*math.sin(angle)+.024*math.cos(angle)])
    extension=.020-.012*9.81*math.sin(angle)/20.
    out["gauge"]=np.r_[pos,quat]
    out["probe"]=np.r_[pos+(.05+extension)*np.array([math.cos(angle),0.,math.sin(angle)]),quat]
    from .fitting import STARTS
    out.update({name:pose.copy() for name,pose in STARTS.items()})
    out["readout"]=np.array([.41,-.25,TOP+.008,0.,0.,0.,1.])
    return out
