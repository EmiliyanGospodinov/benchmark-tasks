"""Offline physical spacer comparison; applied test loads, NOT a robot demo."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
import numpy as np
from pxr import UsdPhysics,UsdGeom,Gf,Sdf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from measure_fit.assets import build
from measure_fit.fitting import SPACERS,build_spacer,FINAL_Z
world=World(stage_units_in_meters=1.,physics_dt=1/240,rendering_dt=1/30)
world.scene.add_default_ground_plane()
paths=build(fixture=True)
rows=[];cases=[]
for i,(name,thickness) in enumerate(SPACERS.items()):
 root='/World/'+name;origin=np.array([0.,i*.25,.12])
 for part in ('housing','shaft','retainer','shim'):
  path=build_spacer(name,thickness) if part=='shim' else paths[part]
  prim=add_reference_to_stage(str(path),root+'/'+part)
  pos=origin+([.014,0,FINAL_Z+.04] if part=='shim' else [0,0,0])
  UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*map(float,pos)))
 shaft=world.scene.add(SingleRigidPrim(root+'/shaft',name=name+'_shaft'))
 shim=world.scene.add(SingleRigidPrim(root+'/shim',name=name+'_shim'))
 j=UsdPhysics.Joint.Define(world.stage,root+'/guide')
 j.CreateBody0Rel().SetTargets([Sdf.Path(root+'/housing')]);j.CreateBody1Rel().SetTargets([Sdf.Path(root+'/shaft')])
 j.CreateCollisionEnabledAttr(True)
 for axis in ('transY','transZ','rotY','rotZ'):
  limit=UsdPhysics.LimitAPI.Apply(j.GetPrim(),axis);limit.CreateLowAttr(0.);limit.CreateHighAttr(0.)
 cases.append((name,thickness,origin,shaft,shim))
world.reset()
try:
 for step in range(240*9):
  t=step/240
  force=-1. if 2<t<4 else (1. if 4<t<6 else 0.)
  torque=.002 if 6<t<8 else 0.
  for name,thickness,origin,shaft,shim in cases:
   shaft._rigid_prim_view.apply_forces_and_torques_at_pos(forces=np.array([[force,0,0]],dtype=np.float32),torques=np.array([[torque,0,0]],dtype=np.float32),is_global=True)
  world.step(render=step%8==0)
  if step%12==0:
   for name,thickness,origin,shaft,shim in cases:
    sp,sq=shaft.get_world_pose();pp,pq=shim.get_world_pose()
    rows.append(dict(t=t,name=name,shaft_position=(sp-origin).tolist(),shaft_quaternion_wxyz=sq.tolist(),shim_position=(pp-origin).tolist(),shim_quaternion_wxyz=pq.tolist()))
 (out/'telemetry.json').write_text(json.dumps(rows))
 results={}
 for name,thickness,origin,shaft,shim in cases:
  near=[r['shaft_position'][0] for r in rows if r['name']==name and 3.5<r['t']<4]
  far=[r['shaft_position'][0] for r in rows if r['name']==name and 5.5<r['t']<6]
  end=[r for r in rows if r['name']==name][-1]
  results[name]=dict(thickness_mm=thickness*1000,observed_travel_mm=1000*(np.mean(far)-np.mean(near)),final_shim_position=end['shim_position'])
 (out/'result.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
finally:app.close()
