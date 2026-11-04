"""Physical stop geometry. Spacer diagnostic variants are scene builds, not runtime overrides."""
import math
from .builder import _Body, GREY, BLUE, LABEL, BLACK
from .design import *

def ring(b, x, thickness, inner=.010, outer=.032, open_top=False):
    # Convex box staves approximate an annulus. The central bore stays open.
    n=32
    for i in range(n):
        a=2*math.pi*i/n
        if open_top and math.sin(a)>.65:
            continue
        r=(inner+outer)/2
        b.box((x,r*math.cos(a),r*math.sin(a)),
              (thickness,2*outer*math.tan(math.pi/n),outer-inner),
              0,LABEL if open_top else GREY,roll=a-math.pi/2 or 1e-12)

def build(thickness=0., fixture=False):
    housing=_Body("housing",1.,(.3,.2),kinematic=True)
    ring(housing,-.004,.008)
    housing.box((.01,0,-.065),(.14,.12,.02),0,GREY)
    housing.box((-.004,0,-.045),(.008,.05,.04),0,GREY)
    if fixture:
        # Visible bearing sleeve corresponding to the ideal native shaft guide.
        ring(housing,-.035,.020,inner=.0082,outer=.018)
        housing.box((-.035,0,-.035),(.020,.030,.040),0,GREY)
        from .fitting import add_guides
        add_guides(housing)
    shaft=_Body("shaft",.12,(.25,.18))
    shaft.cylinder_x(-.07,.11,SHAFT_RADIUS,BLUE)
    shaft.cylinder_x(0,STOP_THICKNESS,STOP_RADIUS,BLUE)
    shaft.box((.095,0,0),(.025,.035,.035),0,BLUE)
    retainer=_Body("retainer",.05,(.3,.2),kinematic=True)
    ring(retainer,REAR_FACE+.004,.008)
    retainer.box((REAR_FACE+.004,0,-.045),(.008,.05,.04),0,GREY)
    out={"housing":housing.save(),"shaft":shaft.save(),"retainer":retainer.save()}
    if thickness:
        spacer=_Body("spacer",.03,(.25,.18),kinematic=True)
        ring(spacer,REAR_FACE-thickness/2,thickness,open_top=True)
        out["spacer"]=spacer.save()
    return out

def build_station():
    out = build(fixture=True)
    gauge=_Body("gauge",.18,(.8,.6))
    # Rectangular handle and four rails leave a real bore for the contact probe.
    gauge.box((0,0,0),(.035,.035,.030),0,GREY)
    gauge.box((0,0,-.0195),(.035,.035,.009),0,GREY)
    for y,z in ((.009,0),(-.009,0),(0,.009),(0,-.009)):
        gauge.box((.035,y,z),(.04,.006,.006),0,GREY)
    # Reference bridge: two pads meet the rear face of the fixed housing.
    # They reference axially but do not support the handheld instrument vertically.
    gauge.box((.03,0,-.02),(.01,.06,.008),0,GREY)
    for y in (-.026,.026):
        gauge.box((.078,y,-.02),(.10,.008,.008),0,GREY)
        gauge.box((.128,y,-.008),(.008,.008,.032),0,GREY)
    probe=_Body("probe",.012,(.15,.1))
    probe.cylinder_x(-.02,.022,.003,BLACK)
    probe.cylinder_x(.022,.026,.006,LABEL)
    out.update(gauge=gauge.save(),probe=probe.save())
    from .fitting import SPACERS,build_spacer
    from .readout import build_readout
    out.update({name:build_spacer(name,t) for name,t in SPACERS.items()})
    out["readout"]=build_readout()
    return out
