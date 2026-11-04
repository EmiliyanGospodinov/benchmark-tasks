"""Force-driven mechanism diagnostic, NOT a robot demonstration."""
import argparse, json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--spacer',type=float,default=.009)
a=parser.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"width":960,"height":540})
import numpy as np
from pxr import UsdPhysics, Gf, Sdf, UsdLux
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from measure_fit.assets import build
world=World(stage_units_in_meters=1.,physics_dt=1/240,rendering_dt=1/30)
world.scene.add_default_ground_plane()
parts=build(a.spacer)
for name,path in parts.items():
    prim=add_reference_to_stage(str(path),'/World/'+name)
    from pxr import UsdGeom
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0,0,.12))
shaft=world.scene.add(SingleRigidPrim('/World/shaft',name='shaft'))
stage=world.stage
# Cylindrical guide: x translation and x rotation free; physical stop surfaces set travel.
j=UsdPhysics.Joint.Define(stage,'/World/guide')
j.CreateBody0Rel().SetTargets([Sdf.Path('/World/housing')]);j.CreateBody1Rel().SetTargets([Sdf.Path('/World/shaft')])
j.CreateCollisionEnabledAttr(True)
for axis in ['transY','transZ','rotY','rotZ']:
    lim=UsdPhysics.LimitAPI.Apply(j.GetPrim(),axis);lim.CreateLowAttr(0.);lim.CreateHighAttr(0.)
light=UsdLux.DomeLight.Define(stage,'/World/light');light.CreateIntensityAttr(1500.)
cam=Camera('/World/camera',position=np.array([.32,-.38,.35]),resolution=(960,540),frequency=30)
cam.initialize();set_camera_view(eye=np.array([.32,-.38,.35]),target=np.array([.01,0,.12]),camera_prim_path='/World/camera')
world.reset()
from scipy.spatial.transform import Rotation
pos=np.array([.32,-.38,.35]);target=np.array([.01,0,.12])
z=(pos-target);z/=np.linalg.norm(z)
x=np.cross(np.array([0.,0.,1.]),z);x/=np.linalg.norm(x)
y=np.cross(z,x)
q=Rotation.from_matrix(np.column_stack([x,y,z])).as_quat()
cam.set_world_pose(position=pos,orientation=np.r_[q[3],q[:3]],camera_axes="usd")
cam.set_clipping_range(.005,10.)
for _ in range(12):world.render()
import imageio.v2 as imageio
video=imageio.get_writer(str(out/'mechanism.mp4'),fps=30,codec='libx264',pixelformat='yuv420p')
rows=[]
try:
    for step in range(240*9):
        t=step/240
        force=0. if t<1 else (-1. if t<3 else (1. if t<5 else 0.))
        torque=.002 if 5<t<8 else 0.
        shaft._rigid_prim_view.apply_forces_and_torques_at_pos(forces=np.array([[force,0,0]],dtype=np.float32),torques=np.array([[torque,0,0]],dtype=np.float32),is_global=True)
        world.step(render=step%8==0)
        if step%8==0:
            pos,quat=shaft.get_world_pose();rows.append({'t':t,'position':pos.tolist(),'quaternion_wxyz':quat.tolist(),'force_x':force,'torque_x':torque})
            frame=cam.get_rgba()
            if frame is not None and frame.size:
                video.append_data(frame[:,:,:3].astype(np.uint8))
                if step==240:imageio.imwrite(out/'setup.png',frame[:,:,:3].astype(np.uint8))
    (out/'telemetry.json').write_text(json.dumps(rows,indent=2))
    xs=[r['position'][0] for r in rows if 2<r['t']<5]
    (out/'result.json').write_text(json.dumps({'spacer_m':a.spacer,'observed_travel_mm':1000*(max(xs)-min(xs)),'diagnostic':'externally applied forces, no robot or gauge'},indent=2))
finally:
    video.close();app.close()
