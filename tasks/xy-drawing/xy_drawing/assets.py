"""Original analytic plotter geometry; body helper reused from A10/A5."""
import math
from .body import _Body
from . import design as d
WHITE=(.96,.96,.91); DARK=(.17,.2,.23)
def frame():
 b=_Body('frame',2.,(.6,.5),kinematic=True)
 b.box((.06,0,.009),(.24,.20,.012),0,WHITE)
 for p in d.KNOBS:
  local=p-d.layout()['frame'][:3]
  b.box((local[0],local[1],local[2]/2-.005),(.025,.025,local[2]-.01),0,DARK)
 for x,y,col in zip(d.POINTS[:,0],d.POINTS[:,1],[(.1,.35,.95),(.95,.45,.02),(.6,.1,.8)]):
  local=d.START-d.layout()['frame'][:3]+[x,y,.0001]
  b.cylinder(local[2],local[2]+.0005,.005,col,False,at=local[:2])
 # Enclosed carriage: physical case walls/floor and a clear colliding lid.
 b.box((.06,0,.003),(.24,.36,.006),0,DARK)
 for x in (-.059,.179):b.box((x,0,.027),(.002,.36,.042),0,DARK)
 for y in (-.179,.179):b.box((.06,y,.027),(.24,.002,.042),0,DARK)
 b.box((.06,0,.047),(.24,.36,.002),0,(.65,.85,.9))
 from pxr import UsdShade,Sdf,Gf
 glass=UsdShade.Material.Define(b.stage,'/Part/clear_lid')
 shader=UsdShade.Shader.Define(b.stage,'/Part/clear_lid/shader');shader.CreateIdAttr('UsdPreviewSurface')
 shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.8,.9,.95))
 shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(0.)
 shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(.5)  # optically ideal clear pane; physical collider retained
 shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.08)
 glass.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
 UsdShade.MaterialBindingAPI.Apply(b.stage.GetPrimAtPath('/Part/box_'+str(b._n))).Bind(glass)
 return b.save()
def knob(name,col):
 b=_Body(name,.12,(1.,.9))
 for k in range(8):b.box((0,0,0),(.04,.04*math.tan(math.pi/16),.018),k*math.pi/8,col)
 b.box((.012,0,.0095),(.008,.003,.001),0,DARK,False)
 return b.save()
def stage_x():
 b=_Body('stage_x',.025,(.3,.2));b.box((0,0,0),(.006,.18,.004),0,(.55,.6,.65),False)
 return b.save()
def stylus():
 b=_Body('stylus',.015,(.3,.2));b.cylinder(-.010,.012,.002,(.02,.02,.02),False)
 b.box((0,0,.010),(.008,.008,.004),0,(.9,.25,.2),False)
 return b.save()
BUILDERS={'frame':frame,'knob_x':lambda:knob('knob_x',(.95,.55,.1)),
 'knob_y':lambda:knob('knob_y',(.1,.65,.75)),'stage_x':stage_x,'stylus':stylus}
