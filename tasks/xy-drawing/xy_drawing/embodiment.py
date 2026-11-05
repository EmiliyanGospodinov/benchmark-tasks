"""Original two-knob XY plotter. Standard YAM station/action/cameras, A14 scene wrapper."""
import atexit,json,math
from pathlib import Path
import numpy as np
from inspect_robots.types import StepResult
from yam_arena import inspect_embodiment as upstream
from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS,READY
from yam_arena import ik,station
from yam_arena.inspect_embodiment import ARMS,YamArenaEmbodiment
from . import design as d
TASK_FILE=Path(__file__).resolve().parents[1]/'task.py'
PARTS=d.PARTS
ACTIVE=None
class DrawingEmbodiment(YamArenaEmbodiment):
    def __init__(self,telemetry=None,**kwargs):
        global ACTIVE
        kwargs.setdefault('task',str(TASK_FILE));kwargs.setdefault('object_jitter',0)
        super().__init__(**kwargs)
        self.layout=d.layout();self._joints=None;self._trace=None;self.enabled=False
        self.telemetry=Path(telemetry) if telemetry else None
        ACTIVE=self
    def _ensure_sim(self, goal):
        if self._env is not None:
            return
        if upstream._APP is None:
            from isaaclab.app import AppLauncher
            upstream._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(upstream._close_app_at_exit)
        import torch
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import env_cfg_callback, task_object, task_room

        from .assets import BUILDERS

        robot = BimanualYamEmbodiment(enable_cameras=True, rig=self.rig, wrist_cam_mount=self.wrist_cam_mount,
                                      contact_sensors=True, robot_contact_offset=0.001)
        objects = []
        for name in PARTS:
            pose = self.layout[name]
            objects.append(Object(name=name, object_type=ObjectType.RIGID, usd_path=str(BUILDERS[name]()),
                                  initial_pose=Pose(position_xyz=tuple(map(float, pose[:3])),
                                                    rotation_xyzw=tuple(map(float, pose[3:])))))
        objects.extend(task_object(o) for o in self.task.objects)
        for obj in objects:
            obj.initial_pose = Pose.from_dict(obj.initial_pose.to_dict())
            obj.enable_reset_pose()
        environment = IsaacLabArenaEnvironment(
            name="yam_xy_drawing", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
            embodiment=robot, env_cfg_callback=env_cfg_callback(self.info.control_hz))
        self._env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(
            num_envs=1, solve_relations=False, device=self.device)).make_registered()
        self._base = self._env.unwrapped
        self._torch, self._goal = torch, goal
        self._free_objects = self.task.free_objects
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {n: (self._base.scene[n].find_joints(ARM_JOINTS, preserve_order=True)[0],
                         self._base.scene[n].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for n in ARMS}
        self._make_joints()
        original_update=self._base.scene.update
        def measured_update(dt):
            original_update(dt)
            if self.enabled:self._transmission(dt)
        self._base.scene.update=measured_update
    def _make_joints(self):
        from pxr import Gf,UsdPhysics,UsdGeom
        stage=self._base.sim.stage
        self._joints={};self.drives={}
        for name,b0,b1,axis,kind,lo,hi in [('xknob','frame','knob_x','Z',UsdPhysics.RevoluteJoint,-5,150),
          ('yknob','frame','knob_y','Z',UsdPhysics.RevoluteJoint,-5,150),
          ('x','frame','stage_x','X',UsdPhysics.PrismaticJoint,-.003,.17),
          ('y','stage_x','stylus','Y',UsdPhysics.PrismaticJoint,-.003,.17)]:
            j=kind.Define(stage,'/World/Plotter_'+name);j.CreateJointEnabledAttr(False)
            j.CreateAxisAttr(axis);j.CreateLocalPos0Attr(Gf.Vec3f(*map(float,self.layout[b1][:3]-self.layout[b0][:3])))
            j.CreateLocalPos1Attr(Gf.Vec3f(0));j.CreateLocalRot0Attr(Gf.Quatf(1.));j.CreateLocalRot1Attr(Gf.Quatf(1.))
            j.CreateLowerLimitAttr(lo);j.CreateUpperLimitAttr(hi);j.CreateCollisionEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)
            drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular' if 'knob' in name else 'linear')
            drive.CreateTypeAttr('force');drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
            drive.CreateStiffnessAttr(0. if 'knob' in name else d.K)
            drive.CreateDampingAttr(.0001 if 'knob' in name else d.DAMP)
            drive.CreateMaxForceAttr(.1 if 'knob' in name else d.MAX_FORCE)
            j.CreateBody0Rel().SetTargets(['/World/envs/env_0/'+b0]);j.CreateBody1Rel().SetTargets(['/World/envs/env_0/'+b1])
            self._joints[name]=j;self.drives[name]=drive
        self.ink=UsdGeom.Mesh.Define(stage,'/World/PlotterInk')
        self.ink.CreateDoubleSidedAttr(True);self.ink.CreateSubdivisionSchemeAttr('none')
        self.ink.CreateDisplayColorAttr([(.008,.008,.008)])
        guide=UsdGeom.Mesh.Define(stage,'/World/PlotterGuide')
        guide.CreateDoubleSidedAttr(True);guide.CreateSubdivisionSchemeAttr('none')
        guide.CreateDisplayColorAttr([(.03,.3,.35)])
        self._ribbon(guide,[d.START+[x,y,-.0003] for x,y in d.POINTS],.012)
    @staticmethod
    def _ribbon(mesh,points,width,breaks=()):
        from pxr import Gf
        vertices=[];indices=[]
        for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
            if i+1 in breaks:continue
            v=np.asarray(b)-a;n=np.linalg.norm(v[:2])
            if n<1e-8:continue
            offset=np.array([-v[1],v[0],0.])*width/(2*n)
            k=len(vertices);vertices.extend([np.asarray(a)+offset,np.asarray(a)-offset,np.asarray(b)-offset,np.asarray(b)+offset]);indices.extend([k,k+1,k+2,k+3])
        # Round caps preserve real isolated contact dots and do not bridge contact gaps.
        if points:
            ends=set([0,len(points)-1,*breaks,*[i-1 for i in breaks]])
            for i in ends:
                p=np.asarray(points[i]);k=len(vertices)
                for theta in np.arange(8)*math.pi/4:vertices.append(p+np.array([math.cos(theta),math.sin(theta),0])*width/2)
                for j in range(1,7):indices.extend([k,k+j,k+j+1,k+j+1])
        mesh.CreatePointsAttr().Set([Gf.Vec3f(*map(float,p)) for p in vertices])
        mesh.CreateFaceVertexCountsAttr().Set([4]*(len(indices)//4));mesh.CreateFaceVertexIndicesAttr().Set(indices)
        if vertices:
            v=np.array(vertices);mesh.CreateExtentAttr().Set([Gf.Vec3f(*map(float,v.min(0))),Gf.Vec3f(*map(float,v.max(0)))])
    def poses(self):
        return {n:self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float) for n in PARTS}
    @staticmethod
    def angle(p):
        x,y,z,w=p[3:];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    def _transmission(self,dt):
        p=self.poses();angles=np.array([self.angle(p[n]) for n in ('knob_x','knob_y')])
        xy=p['stylus'][:2]-d.START[:2]
        for i,key in enumerate(('x','y')):
            target=d.RATIO*angles[i];self.drives[key].GetTargetPositionAttr().Set(float(target))
            force=float(np.clip(d.K*(target-xy[i]),-d.MAX_FORCE,d.MAX_FORCE))
            tq=self._torch.tensor([[[0.,0.,-force*d.RATIO]]],device=self._base.device)
            self._base.scene['knob_'+key].permanent_wrench_composer.set_forces_and_torques_index(forces=self._torch.zeros_like(tq),torques=tq,is_global=True)
        # Ink is measured stylus output at physics rate, never the planned target.
        point=p['stylus'][:3].copy();point[2]=d.START[2]+.0007
        contact=abs(p['stylus'][2]-.010-(d.TOP+.027))<.001
        if contact:
            new_stroke=not self.in_contact
            if not self.points or new_stroke or np.linalg.norm(point[:2]-self.points[-1][:2])>=.0005:
                if self.points and new_stroke:self.breaks.append(len(self.points))
                self.points.append(point)
        self.in_contact=contact
        self.angles=angles;self.xy=xy
    def reset(self,scene,*,seed=None):
        self.enabled=False
        if self._joints:
            for j in self._joints.values():j.GetJointEnabledAttr().Set(False)
        if self._env is None:self._ensure_sim(None)
        super().reset(scene,seed=seed)
        t=self._torch;dev=self._base.device
        for key in ('x','y'):self.drives[key].GetTargetPositionAttr().Set(0.)
        for name in PARTS:
            pose=t.tensor(self.layout[name][None],dtype=t.float32,device=dev);pose[:,:3]+=self._base.scene.env_origins
            self._base.scene[name].write_root_pose_to_sim_index(root_pose=pose)
            self._base.scene[name].write_root_velocity_to_sim_index(root_velocity=t.zeros((1,6),device=dev))
        for j in self._joints.values():j.GetJointEnabledAttr().Set(True)
        self.points=[];self.breaks=[];self.in_contact=False;self.xy=np.zeros(2);self.angles=np.zeros(2);self.enabled=True
        for _ in range(4):self._env.step(self._ready)
        self._t=0.;self.points=[];self.breaks=[];self.in_contact=False
        from .scoring import DrawingScore
        self.scorer=DrawingScore()
        self._ribbon(self.ink,[],.004)
        for _ in range(4):self._base.sim.render()
        if self.telemetry and self._trace is None:
            self.telemetry.parent.mkdir(parents=True,exist_ok=True);self._trace=self.telemetry.open('a',buffering=1)
        if self._trace:self._trace.write(json.dumps({'reset':True})+'\n')
        return self._observation()
    def _task_step(self):
        from pxr import Gf
        if self.points:
            self._ribbon(self.ink,self.points,.004,self.breaks)
        p=self.poses();grips={}
        from yam_arena.runtime import contact_forces
        for name,side in zip(ARMS,('left','right')):
            ids,fids=self._ids[name];q=self._base.scene[name].data.joint_pos.torch[0]
            grips[side]={'tcp':ik.tcp_pose(q[ids].cpu().numpy().astype(float),station.BASES[side])[0].tolist(),
             'gap_mm':2000*abs(q[fids].mean().item()),'pads':{k:v for k,v in contact_forces(self._base.scene[side+'_contacts']).items() if k.startswith('tip_')}}
        info={'xy_mm':(self.xy*1000).tolist(),
          'knob_deg':np.degrees(self.angles).tolist(),'ink_points':len(self.points),'grips':grips}
        info.update(self.scorer.update(self.points,grips,self._t,self.breaks))
        info['stroke_breaks']=self.breaks.copy()
        if self._trace:self._trace.write(json.dumps({'t':round(self._t,3),**info})+'\n')
        if round(self._t*20)%20==0:print('[xy]',round(self._t,2),info['xy_mm'],info['knob_deg'],flush=True)
        return StepResult(observation=self._observation(),terminated=info['success'],termination_reason='success' if info['success'] else None,info=info)
    def close(self):
        if self.telemetry and getattr(self,'points',None):
            (self.telemetry.parent/'trace-final.json').write_text(json.dumps([p.tolist() for p in self.points]))
            (self.telemetry.parent/'trace-breaks.json').write_text(json.dumps(self.breaks))
        if self._trace:self._trace.close();self._trace=None
        super().close()
