from __future__ import annotations

import atexit
import json
import math
from pathlib import Path

import numpy as np
from inspect_robots.types import StepResult
from yam_arena import inspect_embodiment as upstream
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, YamArenaEmbodiment

from . import design as d

TASK_FILE = Path(__file__).resolve().parents[1] / "task.py"
PARTS = ("probe_l", "tip_l", "probe_r", "tip_r", "panel", "meter", "pointer", "display_dash", "display_ol", "display_one", "result_board")
FREE = ("probe_l", "tip_l", "probe_r", "tip_r", "pointer")
ACTIVE = None  # read only by the declared privileged demonstration policy


class HarnessEmbodiment(YamArenaEmbodiment):
    def __init__(self, telemetry=None, wiring="crossed", **kwargs):
        global ACTIVE
        kwargs.setdefault("task", str(TASK_FILE))
        kwargs.setdefault("object_jitter", 0)
        super().__init__(**kwargs)
        self.telemetry = Path(telemetry).expanduser() if telemetry else None
        self.wiring = str(wiring)
        self.layout = d.start_poses()
        self._trace = None
        self.scorer = None
        self._springs = []
        ACTIVE = self

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
                                      contact_sensors=False, robot_contact_offset=0.001)
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
            name="yam_harness_qa", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
            embodiment=robot, env_cfg_callback=env_cfg_callback(self.info.control_hz))
        self._env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(
            num_envs=1, solve_relations=False, device=self.device)).make_registered()
        self._base = self._env.unwrapped
        self._torch, self._goal = torch, goal
        self._free_objects = self.task.free_objects
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {n: (self._base.scene[n].find_joints(ARM_JOINTS, preserve_order=True)[0],
                         self._base.scene[n].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for n in ARMS}
        self._make_springs()

    def _make_springs(self):
        from pxr import UsdPhysics,Gf,Sdf
        for side in ['l','r']:
            j=UsdPhysics.PrismaticJoint.Define(self._base.sim.stage,'/World/TipGuide_'+side)
            j.CreateJointEnabledAttr(False);j.CreateAxisAttr('X');j.CreateLowerLimitAttr(-.008);j.CreateUpperLimitAttr(0.)
            for k in [0,1]:
                getattr(j,'CreateLocalPos'+str(k)+'Attr')(Gf.Vec3f(0.));getattr(j,'CreateLocalRot'+str(k)+'Attr')(Gf.Quatf(1.))
            j.CreateBody0Rel().SetTargets([Sdf.Path('/World/envs/env_0/probe_'+side)]);j.CreateBody1Rel().SetTargets([Sdf.Path('/World/envs/env_0/tip_'+side)])
            j.CreateCollisionEnabledAttr(False);j.CreateExcludeFromArticulationAttr(True)
            dr=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');dr.CreateTypeAttr('force');dr.CreateStiffnessAttr(200.);dr.CreateDampingAttr(1.);dr.CreateTargetPositionAttr(.001);dr.CreateMaxForceAttr(20.)
            self._springs.append(j)
    def poses(self):
        origin=np.r_[self._base.scene.env_origins[0].cpu().numpy(),0,0,0,0]
        return {n:self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float)-origin for n in PARTS}
    def reset(self,scene,*,seed=None):
        for j in self._springs:j.GetJointEnabledAttr().Set(False)
        if self.telemetry and self._trace is None:
            self.telemetry.parent.mkdir(parents=True,exist_ok=True);self._trace=self.telemetry.open('a',buffering=1)
        super().reset(scene,seed=seed)
        for _ in range(2):
            for j in self._springs:j.GetJointEnabledAttr().Set(False)
            for n in PARTS:
                pose=self._torch.tensor(self.layout[n][None],dtype=self._torch.float32,device=self._base.device);pose[:,:3]+=self._base.scene.env_origins
                obj=self._base.scene[n];obj.write_root_pose_to_sim_index(root_pose=pose);obj.write_root_velocity_to_sim_index(root_velocity=self._torch.zeros((1,6),device=self._base.device))
            for j in self._springs:j.GetJointEnabledAttr().Set(True)
            for _ in range(5):self._env.step(self._ready)
        from .scoring import HarnessScore
        self.scorer=HarnessScore(self._base.step_dt)
        self._t=0.;self.pair=None;self.settle=0.;self.readings={};self.live=None;self._display=None
        self._prev=self.poses()
        self._set_display('--')
        for _ in range(4):self._base.sim.render()
        if self._trace:self._trace.write(json.dumps({'reset':True,'wiring':self.wiring})+'\n')
        return self._observation()
    def _set_display(self,word):
        if word==self._display:return
        for name,label in [('dash','--'),('ol','OL'),('one','1')]:
            pose=self.layout['meter'].copy();pose[2]+=.008 if word==label else -.02
            pose=self._torch.tensor(pose[None],dtype=self._torch.float32,device=self._base.device);pose[:,:3]+=self._base.scene.env_origins
            self._base.scene['display_'+name].write_root_pose_to_sim_index(root_pose=pose)
        self._display=word
    def _gripper_state(self,poses=None):
        from yam_arena import ik,station
        from yam_arena.constants import FINGER_OPEN
        opening=[];tcps=[]
        for name,side in zip(ARMS,['left','right']):
            arm_ids,finger_ids=self._ids[name];q=self._base.scene[name].data.joint_pos.torch[0]
            opening.append(float(np.clip(q[finger_ids].mean().item()/FINGER_OPEN,0,1)));tcps.append(ik.tcp_pose(q[arm_ids].cpu().numpy().astype(float),station.BASES[side])[0])
        return opening,tcps
    def _task_step(self):
        poses=self.poses();contacts=[d.contact(poses['probe_'+s],poses['tip_'+s]) for s in ['l','r']]
        pair=d.canonical_pair([v[0] for v in contacts])
        valid=pair[0] in 'AB' and pair[1] in 'CD'
        self.settle=self.settle+self._base.step_dt if valid and pair==self.pair else 0.
        self.pair=pair;self.live=d.connected(self.wiring,*pair) if valid else None
        self._set_display('--' if self.live is None else '1' if self.live else 'OL')
        opening,tcps=self._gripper_state()
        speeds={n:float(np.linalg.norm(poses[n][:3]-self._prev[n][:3]))/self._base.step_dt for n in FREE}
        self._prev={k:v.copy() for k,v in poses.items()}
        score=self.scorer.update(pair,self.live,poses,opening,tcps,speeds)
        self.readings=self.scorer.readings
        info={'score':score,'success':self.scorer.success,'pair':pair,'settle':self.scorer.settle,'reading':self.live,'readings':self.readings.copy(),'compression_mm':[1000*c[1] for c in contacts],'verdict':self.scorer.result,'hold':self.scorer.hold}
        if self._trace:self._trace.write(json.dumps({'t':round(self._t,3),**info,'poses':{n:poses[n].round(6).tolist() for n in FREE},'tcps':[p.round(5).tolist() for p in self._gripper_state()[1]]})+'\n')
        if round(self._t*20)%10==0:print('[harness]',round(self._t,2),info,flush=True)
        return StepResult(observation=self._observation(),terminated=self.scorer.success,termination_reason="success" if self.scorer.success else None,info=info)
    def close(self):
        if self._trace:self._trace.close();self._trace=None
        super().close()
