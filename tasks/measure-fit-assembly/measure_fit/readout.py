"""Remote contact-indicator display. Pixels show measured probe travel, never a shim answer."""
import math

DIGITS = ("abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")

def display_digits(extension_mm):
    """Offset scale: physical probe -20..20 mm corresponds to visible 00.0..40.0."""
    if not math.isfinite(extension_mm): return None
    value=max(0,min(400,round((extension_mm+20)*10)))
    return (value//100, value//10%10, value%10)

def build_readout():
    from .builder import _Body, WHITE, BLACK, GREY
    b=_Body('readout',.2,(.5,.4),kinematic=True)
    b.box((0,0,0),(.085,.15,.016),0,WHITE)
    specs={'a':(0,.024,.023,.004),'b':(.012,.012,.004,.021),
           'c':(.012,-.012,.004,.021),'d':(0,-.024,.023,.004),
           'e':(-.012,-.012,.004,.021),'f':(-.012,.012,.004,.021),
           'g':(0,0,.023,.004)}
    for x in (-.047,-.012,.033):
        for key in 'abcdefg':
            sx,sy,w,h=specs[key]
            b.box((sy,-(x+sx),.009),(h,w,.001),0,BLACK,collide=False)
            from pxr import UsdGeom
            UsdGeom.Imageable(b.stage.GetPrimAtPath(f"/Part/box_{b._n}")).MakeInvisible()
    b.box((-.024,-.010,.009),(.004,.004,.001),0,BLACK,collide=False)
    return b.save()

class Readout:
    def __init__(self,stage,path):
        from pxr import UsdGeom
        self.UsdGeom=UsdGeom
        self.segments=[UsdGeom.Imageable(stage.GetPrimAtPath(f'{path}/box_{i}')) for i in range(2,23)]
        self.previous=None
    def update(self,extension_mm):
        digits=display_digits(extension_mm)
        if digits==self.previous:return
        for i,segment in enumerate(self.segments):
            on=digits is not None and 'abcdefg'[i%7] in DIGITS[digits[i//7]]
            segment.GetVisibilityAttr().Set('inherited' if on else 'invisible')
        self.previous=digits
