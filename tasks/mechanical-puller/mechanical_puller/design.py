"""Geometry and mechanism constants of the puller fixture. NumPy only. Metres, kg, radians; quaternions xyzw.

Screw model (declared constitutive coupling): a rotating-nut screw. The handle turns in the puller body on a
revolute joint; the ram slides in the body on a prismatic joint whose stiff drive (physics rate, internal, so
equal and opposite on ram and body) tracks LEAD x the handle's measured turns; the drive's resisting force is
fed back to the handle as torque F*LEAD/(2 pi). PhysX's native rack-and-pinion joint is refused in this
direct-GPU scene (probe-001).
"""
import math

import numpy as np

TOP = 0.7475
LEAD = 0.006  # m of ram travel per handle turn (coarse multi-start training screw), nominal
SCREW_K = 20000.0  # N/m stiffness of the screw coupling (ram drive)
SCREW_DAMPING = 50.0  # N s/m
SCREW_MAX_FORCE = 60.0  # N, cap of the coupling (thread strength of the toy fixture)
SCREW_RATIO = 2 * math.pi / LEAD  # rad of handle per m of ram (unit to be verified in the probe)

# isolated probe (A10 step 1)
PROBE_FRAME = np.array([0.35, 0.18, TOP + 0.08])  # fixed frame centre; screw axis along +x
RAM_TRAVEL = 0.020
ANVIL_GAP = 0.006  # ram tip to anvil face at the start
ANVIL_K = 2000.0  # N/m spring holding the anvil (force gauge: F = k * displacement)
PROBE_SPEED_DEG = 180.0  # stand-in motor on the handle, deg/s
PROBE_TORQUE = 0.05  # N m motor cap


# -- draft scene (A10 step 2): vertical shaft fixture and a two-jaw puller; heights above TOP ------------
LEAD = 0.012  # coarse training screw: 12 mm of ram per handle turn
FIX = np.array([0.28, 0.04])  # within both arms reach for every fitting and turning target (offline IK)
FLANGE = (0.10, 0.10, 0.010)
SHAFT_R, SHAFT_TOP = 0.006, 0.035
CUP_R, CUP_H = 0.0035, 0.003
SHAFT_CONTACT_TOP = SHAFT_TOP + CUP_H
COLLAR_R, COLLAR_BORE, HUB_R = 0.022, 0.0065, 0.011
COLLAR_Z0, HUB_H, RIM_H = FLANGE[2], 0.004, 0.010  # collar frame origin: hub bottom on the flange
COLLAR_CLEAR = SHAFT_CONTACT_TOP - COLLAR_Z0  # rise at which the bore leaves the shaft
# Integral custom-machined iglidur J collar, no metal housing or bonded sleeve.
# The physical free land is interference-fitted; the collision proxy remains relieved.
COLLAR_FREE_BORE_D = .011990
COLLAR_LAND_H = .002
COLLAR_COLLISION_BORE = .0065
COLLAR_DENSITY = 1490.0
# Offline axisymmetric elasticity includes hub/flange constraint; see tools/axisymmetric_fit.py.
FIT_OVERLAP = (0., .000125, .00025, .0005, .001, .0015, .002)
FIT_NORMAL = (0.,53.2158202245,60.8246475653,72.6763773263,92.6575170302,111.449766833,122.530140752)
FIT_MU = .12
COLLAR_HOLD = FIT_NORMAL[-1]*FIT_MU
FIT_FORCE_SCALE = 1.0  # capability sensitivity only; nominal public task is 1

def collar_overlap(rise):
    overlap=float(np.clip(SHAFT_CONTACT_TOP - COLLAR_Z0 - rise, 0., COLLAR_LAND_H))
    return 0.0 if overlap<1e-10 else overlap

def collar_resistance(rise):
    overlap=collar_overlap(rise)
    if overlap<=0:return 0.0
    normal=float(np.interp(overlap,FIT_OVERLAP,FIT_NORMAL))
    # Add steel radial compliance in series. The cup is an annulus, not a solid rod.
    # Uniform-pressure cylinder approximation; mixed contact is length-weighted.
    solid=float(np.clip(SHAFT_TOP-COLLAR_Z0-rise,0.,overlap))
    hoop=(SHAFT_R**2+CUP_R**2)/(SHAFT_R**2-CUP_R**2)-.30
    compliance_factor=((1-.30)*solid+hoop*(overlap-solid))/overlap
    steel_k=2*math.pi*overlap*210e9/compliance_factor
    polymer_k=normal/((2*SHAFT_R-COLLAR_FREE_BORE_D)/2)
    return normal/(1+polymer_k/steel_k)*FIT_MU*FIT_FORCE_SCALE
COLLAR_HOLD = collar_resistance(0.0)
JAW_HOLD = 3.0  # N, self-retaining friction of each jaw slide
STICK_WINDOW = 0.0005  # m of elastic travel before slipping (stick-slip model)
MASS = {"body": 0.12, "handle": 0.12,  # die-cast knob: a 20 g knob slipped in the pads (draft-006; A5 lesson)
         "ram": 0.02, "jaw": 0.02, "collar": 0.03}
# Analytic polymer volume and centre of mass, using physical bore, not proxy clearance.
_COLLAR_SECTIONS = ((0.,COLLAR_LAND_H,COLLAR_FREE_BORE_D/2,HUB_R),
                    (COLLAR_LAND_H,HUB_H,COLLAR_COLLISION_BORE,HUB_R),
                    (HUB_H,HUB_H+RIM_H,COLLAR_COLLISION_BORE,COLLAR_R))
_COLLAR_V = [math.pi*(ro*ro-ri*ri)*(z1-z0) for z0,z1,ri,ro in _COLLAR_SECTIONS]
MASS["collar"] = COLLAR_DENSITY*sum(_COLLAR_V)
COLLAR_COM_Z = sum(v*(section[0]+section[1])/2 for v,section in zip(_COLLAR_V,_COLLAR_SECTIONS))/sum(_COLLAR_V)

# puller body frame: on the screw axis at the crossbar centre
CROSSBAR = (0.100, 0.016, 0.012)
BOSS = (0.030, 0.030, 0.030)  # above the crossbar
KNOB_R, KNOB_Z = 0.018, 0.048  # handle knob centre height in the body frame
RAM_TOP, RAM_TIP = 0.030, -0.0375  # point clears shaft while jaws enter the rim gap
JAW_OPEN, JAW_CLOSED = 0.040, 0.025  # jaw axis |x| in the body frame
HOOK_Z, HOOK_IN = -0.067, 0.010  # hook lip bottom height and inward reach
TAB = (0.008, 0.020, 0.020)  # jaw tab above the crossbar, for closing the jaws
 # Centre hole dimensions are defined with SHAFT_CONTACT_TOP above.
POINT_R, POINT_L = 0.0025, 0.004  # ram point
FIT_Z = COLLAR_Z0 + (HUB_H - 0.003) / 2 - HOOK_Z  # body height with the hooks in the gap under the rim
REST_XY = (0.38, 0.10)  # reachable for the handle-end hold (offline IK scan)
HANDLE = (0.020, 0.110, 0.024)  # side handle bar toward -y, from the boss face (holding point 11 cm off the axis)
HANDLE_Z = 0.021
GRIP_Z = HANDLE_Z + 0.020  # upper gripping face clears the connecting handle bar
GRIP_BLOCK = (0.024, 0.030, 0.024)  # block at the handle end, pinched across y
REST_Z = -HOOK_Z + 0.005  # body height on the tool rest (hooks 5 mm above the table)
PILLAR_X = 0.009  # 6 mm pillars between the ram (4 mm) and the holder's lower finger (from 16 mm)


def draft_layout():
    b = np.r_[REST_XY, TOP + REST_Z, 0, 0, 0, 1.0]
    lay = {"body": b, "handle": b + np.r_[0, 0, KNOB_Z, 0, 0, 0, 0], "ram": b.copy(),
           "jaw_pos": b + np.r_[JAW_OPEN, 0, 0, 0, 0, 0, 0], "jaw_neg": b + np.r_[-JAW_OPEN, 0, 0, 0, 0, 0, 0],
           "collar": np.r_[FIX, TOP + COLLAR_Z0, 0, 0, 0, 1.0], "fixture": np.r_[FIX, TOP, 0, 0, 0, 1.0]}
    return lay


def prepared_layout():
    """Operator-prepared jaw engagement; all tool bodies remain free."""
    lay=draft_layout()
    b=np.r_[FIX,TOP+FIT_Z,0.,0.,0.,1.]
    lay.update(body=b,handle=b+np.r_[0,0,KNOB_Z,0,0,0,0],ram=b.copy(),
               jaw_pos=b+np.r_[JAW_CLOSED,0,0,0,0,0,0],
               jaw_neg=b+np.r_[-JAW_CLOSED,0,0,0,0,0,0])
    return lay

# Recovery tray and raised parking rest clear the extended ram.
TRAY_XY = (.28, .18)
TRAY_INNER = (.110, .090)
TRAY_FLOOR = .004
PARK_Z = .095
REST_XY = TRAY_XY
PILLAR_X = .045
