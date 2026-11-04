"""Contact-indicator fitting fixture with the standard YAM station and cameras."""
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
PARTS = ("housing", "shaft", "retainer", "gauge", "probe", "shim_6", "shim_9", "shim_14", "readout")
FREE = ("shaft", "gauge", "probe")
ACTIVE = None  # read only by the declared privileged demonstration policy


class MeasureFitEmbodiment(YamArenaEmbodiment):
    def __init__(self, telemetry=None, **kwargs):
        global ACTIVE
        kwargs.setdefault("task", str(TASK_FILE))
        kwargs.setdefault("object_jitter", 0)
        super().__init__(**kwargs)
        self.telemetry = Path(telemetry).expanduser() if telemetry else None
        self.layout = d.start_poses()
        self._trace = None
        self._joint = None
        ACTIVE = self

    def _ensure_sim(self, goal):
        print("[a11-init] ensure sim",flush=True)
        if self._env is not None:
            return
        if upstream._APP is None:
            from isaaclab.app import AppLauncher
            print("[a11-init] constructing AppLauncher",flush=True)
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

        from .assets import build_station
        print("[a11-init] building task assets",flush=True)
        paths = build_station()
        print("[a11-init] assets built",flush=True)

        robot = BimanualYamEmbodiment(enable_cameras=True, rig=self.rig, wrist_cam_mount=self.wrist_cam_mount,
                                      contact_sensors=False, robot_contact_offset=0.001)
        objects = []
        for name in PARTS:
            pose = self.layout[name]
            objects.append(Object(name=name, object_type=ObjectType.RIGID, usd_path=str(paths[name]),
                                  initial_pose=Pose(position_xyz=tuple(map(float, pose[:3])),
                                                    rotation_xyzw=tuple(map(float, pose[3:])))))
        objects.extend(task_object(o) for o in self.task.objects)
        for obj in objects:
            obj.initial_pose = Pose.from_dict(obj.initial_pose.to_dict())
            obj.enable_reset_pose()
        environment = IsaacLabArenaEnvironment(
            name="yam_measure_fit", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
            embodiment=robot, env_cfg_callback=env_cfg_callback(self.info.control_hz))
        self._env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(
            num_envs=1, solve_relations=False, device=self.device)).make_registered()
        self._base = self._env.unwrapped
        self._torch, self._goal = torch, goal
        self._free_objects = self.task.free_objects
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {n: (self._base.scene[n].find_joints(ARM_JOINTS, preserve_order=True)[0],
                         self._base.scene[n].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for n in ARMS}
        self._make_guide()
        from .readout import Readout
        self.readout=Readout(self._base.sim.stage,"/World/envs/env_0/readout")

    def _make_guide(self):
        from pxr import Gf, Sdf, UsdPhysics
        stage = self._base.sim.stage
        j = UsdPhysics.Joint.Define(stage, "/World/ShaftGuide")
        j.CreateJointEnabledAttr(False)
        j.CreateCollisionEnabledAttr(True)
        j.CreateExcludeFromArticulationAttr(True)
        for axis in ("transY","transZ","rotY","rotZ"):
            limit=UsdPhysics.LimitAPI.Apply(j.GetPrim(),axis)
            limit.CreateLowAttr(0.); limit.CreateHighAttr(0.)
        j.CreateBody0Rel().SetTargets([Sdf.Path("/World/envs/env_0/housing")])
        j.CreateBody1Rel().SetTargets([Sdf.Path("/World/envs/env_0/shaft")])
        g=UsdPhysics.PrismaticJoint.Define(stage,"/World/ContactProbe")
        g.CreateJointEnabledAttr(False)
        g.CreateAxisAttr("X")
        g.CreateLocalPos0Attr(Gf.Vec3f(.05,0,0))
        g.CreateLocalPos1Attr(Gf.Vec3f(0.))
        g.CreateLowerLimitAttr(-.02); g.CreateUpperLimitAttr(.02)
        g.CreateCollisionEnabledAttr(False)
        g.CreateExcludeFromArticulationAttr(True)
        g.CreateBody0Rel().SetTargets([Sdf.Path("/World/envs/env_0/gauge")])
        g.CreateBody1Rel().SetTargets([Sdf.Path("/World/envs/env_0/probe")])
        drive=UsdPhysics.DriveAPI.Apply(g.GetPrim(),"linear")
        drive.CreateTypeAttr("force"); drive.CreateStiffnessAttr(20.)
        drive.CreateDampingAttr(1.); drive.CreateTargetPositionAttr(.02)
        drive.CreateMaxForceAttr(1.)
        self._joint, self._gauge_joint=j,g

    def poses(self):
        return {n: self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float)
                for n in PARTS}

    def reset(self, scene, *, seed=None):
        print("[a11-init] embodiment reset",flush=True)
        for joint in (self._joint, getattr(self,"_gauge_joint",None)):
            if joint is not None: joint.GetJointEnabledAttr().Set(False)
        if self.telemetry and self._trace is None:
            self.telemetry.parent.mkdir(parents=True,exist_ok=True)
            self._trace=self.telemetry.open("a",buffering=1)
        super().reset(scene,seed=seed)
        torch,base=self._torch,self._base
        for name in PARTS:
            obj=base.scene[name]
            pose=torch.tensor(self.layout[name][None],dtype=torch.float32,device=base.device)
            pose[:,:3]+=base.scene.env_origins
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6),device=base.device))
        self._joint.GetJointEnabledAttr().Set(True)
        self._gauge_joint.GetJointEnabledAttr().Set(True)
        for _ in range(10): self._env.step(self._ready)
        from .measurement import read_indicator
        poses=self.poses()
        self.readout.previous=None
        self.readout.update(read_indicator(poses["gauge"],poses["probe"],poses["shaft"])["extension_mm"])
        # Advance Fabric once so reset-time visibility reaches the policy cameras.
        self._env.step(self._ready)
        for _ in range(4): base.sim.render()
        self._t=0.
        from .scoring import FitScore
        self.scorer=FitScore(self.info.control_hz)
        self._previous_poses=self.poses()
        if self._trace: self._trace.write(json.dumps({"reset":True})+"\n")
        return self._observation()

    def _task_step(self):
        from .measurement import read_indicator,rotation
        from .fitting import SPACERS,FINAL_Z,GRIP_Z
        from yam_arena import ik,station
        from yam_arena.constants import FINGER_OPEN
        poses=self.poses()
        if any(not np.all(np.isfinite(p)) or np.linalg.norm(p[3:])<1e-8 for p in poses.values()):
            self.scorer.score=0.;self.scorer.success=False
            return StepResult(observation=self._observation(),terminated=True,
                termination_reason='invalid_state',info={'score':0.,'success':False})
        measured=read_indicator(poses["gauge"],poses["probe"],poses["shaft"])
        self.readout.update(measured['extension_mm'])
        g=poses['gauge']; h=poses['housing']; R=rotation(g[3:])
        pads=[g[:3]+R@np.array([.132,y,0.]) for y in (-.026,.026)]
        gaps=[h[0]-.008-pad[0] for pad in pads]
        radial=[np.linalg.norm(pad[1:]-h[1:3]) for pad in pads]
        reference=all(abs(gap)<.0005 and .012<radius<.033 for gap,radius in zip(gaps,radial))
        installed=None;clearance=12.
        for name,thickness in SPACERS.items():
            pose=poses[name];r=rotation(pose[3:]);rel=pose[:3]-h[:3]
            if (rel[0]-thickness/2>=.0075 and rel[0]+thickness/2<=.0205
                and abs(rel[1])<.003 and abs(rel[2]-FINAL_Z)<.004
                and r[2,2]>.98 and r[0,0]>.98):
                installed=name;clearance=1000*(d.REAR_FACE-d.STOP_THICKNESS-thickness)
        points=[poses['gauge'][:3],poses['shaft'][:3]+rotation(poses['shaft'][3:])@np.array([.095,0,0])]
        points += [poses[n][:3]+rotation(poses[n][3:])@np.array([0,0,GRIP_Z]) for n in SPACERS]
        openings=[];distances=[]
        for arm,side in zip(ARMS,('left','right')):
            ids,fingers=self._ids[arm];q=self._base.scene[arm].data.joint_pos.torch[0]
            openings.append(float(q[fingers].mean().item()/FINGER_OPEN))
            tcp=ik.tcp_pose(q[ids].cpu().numpy(),station.BASES[side])[0]+self._base.scene.env_origins[0].cpu().numpy()
            distances.append(min(float(np.linalg.norm(tcp-v)) for v in points))
        released=min(openings)>.8 and min(distances)>.06
        watch=['shaft','gauge']+([installed] if installed else [])
        speed=max(np.linalg.norm(poses[n][:3]-self._previous_poses[n][:3])/self._base.step_dt for n in watch)
        angular_speed=max(2*math.acos(float(np.clip(abs(np.dot(poses[n][3:],self._previous_poses[n][3:]))/(np.linalg.norm(poses[n][3:])*np.linalg.norm(self._previous_poses[n][3:])),0,1)))/self._base.step_dt for n in watch)
        gauge_rest=abs(poses['gauge'][2]-(d.TOP+.024))<.006
        shaft_mm=1000*(poses['shaft'][0]-h[0])
        shaft_r=rotation(poses['shaft'][3:]);angle=math.atan2(shaft_r[2,1],shaft_r[1,1])
        score=self.scorer.update(shaft_mm=shaft_mm,probe_mm=measured['extension_mm'],
            contact_valid=measured['contact_geometry_valid'],reference_valid=reference,
            installed=installed is not None,clearance_mm=clearance,angle_rad=angle,
            released=released,stable=speed<.003 and angular_speed<.1 and gauge_rest)
        self._previous_poses=poses
        info={"score":score,"success":self.scorer.success,"shaft_x_mm":shaft_mm,
              "probe_mm":measured['extension_mm'],**measured,"reference_valid":reference,
              "reference_gap_mm":[1000*x for x in gaps],"installed":installed,
              "clearance_mm":clearance,"shaft_angle_rad":angle,"released":released,
              "gripper_open":openings,"gripper_clearance_m":distances,"speed_mps":float(speed),"angular_speed_radps":angular_speed,
              "initial_measurement_mm":self.scorer.initial_travel_mm,"post_measured":self.scorer.post_measured}
        if self._trace:self._trace.write(json.dumps({"t":round(self._t,4),**info,"poses":{k:v.tolist() for k,v in poses.items()}})+"\n")
        if round(self._t*20)%20==0:
            print(f"[measure] t={self._t:.2f} shaft={shaft_mm:.2f} probe={measured['extension_mm']:.2f} ref={reference} contact={measured['contact_geometry_valid']} installed={installed} score={score}",flush=True)
        return StepResult(observation=self._observation(),terminated=self.scorer.success,
            termination_reason="success" if self.scorer.success else None,info=info)

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()
