"""Physical split spacers and housing guide geometry for radial shim installation."""
import numpy as np
from . import design as d

SPACERS={'shim_6':.006,'shim_9':.009,'shim_14':.014}
STARTS={name:np.r_[x,.18,d.TOP+.024,0.,0.,0.,1.]
        for name,x in zip(SPACERS,(.14,.26,.38))}
FINAL_Z=-.012
GRIP_Z=.063

def add_guides(housing):
    from .builder import GREY
    for y in (-.034,.034):
        # Open-topped channel retains the shim radially without a hidden weld.
        housing.box((.014,y,-.0175),(.022,.006,.035),0,GREY)
        housing.box((.0075,y,-.0175),(.001,.010,.035),0,GREY)
        housing.box((.0205,y,-.0175),(.001,.010,.035),0,GREY)
    # Bottom supports only the spacer legs, outside the rotating shaft/collar.
    for y in (-.023,.023):
        housing.box((.014,y,-.035),(.012,.016,.004),0,GREY)

def build_spacer(name,thickness):
    from .builder import _Body,LABEL,BLUE,RED
    color={"shim_6":BLUE,"shim_9":LABEL,"shim_14":RED}[name]
    b=_Body(name,.035,(.3,.2))
    for y in (-.020,.020):
        b.box((0,y,0),(thickness,.020,.040),0,color)
    b.box((0,0,.025),(thickness,.060,.010),0,color)
    b.box((0,0,.041),(thickness,.020,.026),0,color)
    b.box((0,0,GRIP_Z),(.026,.028,.025),0,color)
    return b.save()

def seated_pose(thickness):
    return np.r_[d.ORIGIN+[(d.STOP_THICKNESS+d.REAR_FACE)/2,0.,FINAL_Z],0.,0.,0.,1.]
