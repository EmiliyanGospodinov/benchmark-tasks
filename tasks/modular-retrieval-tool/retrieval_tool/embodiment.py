"""Embodiment ``yam_retrieval_tool``: the supplied bimanual YAM station with a loose two-piece retrieval tool,
a covered service hood holding a cartridge, a recovery tray and a head stand.

Robot, cameras, 14-D joint action and observations are the guide adapter's (``yam_arena``), unchanged. The
bayonet retention, hook engagement and cartridge motion use rigid-body contact.
A spring-return manual gate blocks insertion and accidental bayonet reversal; no joint or
weld attaches the shaft to the head. Station/reset pattern copied from our A6 hand pump.
"""
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
PARTS = ("shaft", "head", "cartridge", "hood", "tray", "stand", "plunger")
FREE = ("shaft", "head", "cartridge", "plunger")
ACTIVE = None  # read only by the declared privileged demonstration policy


class RetrievalEmbodiment(YamArenaEmbodiment):
    def __init__(self, telemetry=None, layout="loose", **kwargs):
        global ACTIVE
        kwargs.setdefault("task", str(TASK_FILE))
        kwargs.setdefault("object_jitter", 0)
        super().__init__(**kwargs)
        self.telemetry = Path(telemetry).expanduser() if telemetry else None
        self.layout_name = str(layout)
        self.layout = d.start_poses(self.layout_name)
        self._trace = None
        self.scorer = None
        self._spring = None
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
            name="yam_retrieval_tool", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
            embodiment=robot, env_cfg_callback=env_cfg_callback(self.info.control_hz))
        self._env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(
            num_envs=1, solve_relations=False, device=self.device)).make_registered()
        self._base = self._env.unwrapped
        self._torch, self._goal = torch, goal
        self._free_objects = self.task.free_objects
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {n: (self._base.scene[n].find_joints(ARM_JOINTS, preserve_order=True)[0],
                         self._base.scene[n].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for n in ARMS}
        self._make_spring()

    def _make_spring(self):
        """Shaft-fixed lateral release guide with a passive return spring.
        Shaft/control contacts omitted for unmodelled guide rails; head/control contacts remain enabled."""
        from pxr import Gf, Sdf, UsdPhysics

        stage = self._base.sim.stage
        j = UsdPhysics.PrismaticJoint.Define(stage, "/World/DetentGuide")
        j.CreateJointEnabledAttr(False)
        j.CreateAxisAttr("Y")
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0))
        j.CreateLowerLimitAttr(0.0)
        j.CreateUpperLimitAttr(d.RELEASE_TRAVEL)
        j.CreateCollisionEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(True)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(d.RELEASE_K))
        drive.CreateDampingAttr(float(d.RELEASE_DAMP))
        drive.CreateTargetPositionAttr(float(-d.RELEASE_PRELOAD))
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(200.0)
        j.CreateBody0Rel().SetTargets([Sdf.Path("/World/envs/env_0/shaft")])
        j.CreateBody1Rel().SetTargets([Sdf.Path("/World/envs/env_0/plunger")])
        self._spring = j

    def poses(self):
        origin = np.r_[self._base.scene.env_origins[0].cpu().numpy(), 0, 0, 0, 0]
        return {n: self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float) - origin for n in PARTS}

    def reset(self, scene, *, seed=None):
        if self._spring is not None:
            self._spring.GetJointEnabledAttr().Set(False)
        if self.telemetry and self._trace is None:
            self.telemetry.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.telemetry.open("a", buffering=1)
        super().reset(scene, seed=seed)
        torch, base = self._torch, self._base
        for k in range(2):
            self._spring.GetJointEnabledAttr().Set(False)
            for name in PARTS:
                pose = torch.tensor(self.layout[name][None], dtype=torch.float32, device=base.device)
                pose[:, :3] += base.scene.env_origins
                base.scene[name].write_root_pose_to_sim_index(root_pose=pose)
                base.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))
            self._spring.GetJointEnabledAttr().Set(True)  # control written at its rest pose on the shaft
            for _ in range(max(1, round(.25 / base.step_dt))):
                self._env.step(self._ready)
        for _ in range(4):
            base.sim.render()
        from .scoring import RetrievalScore
        self.scorer = RetrievalScore(self.info.control_hz)
        poses = self.poses()
        self._start = {k: v.copy() for k, v in poses.items()}
        self._prev = {k: v.copy() for k, v in poses.items()}
        self._t = 0.0
        if self._trace:
            self._trace.write(json.dumps({"reset": True, "poses": {k: v.round(6).tolist() for k, v in poses.items()}}) + "\n")
        return self._observation()

    def _gripper_state(self, poses):
        from yam_arena import ik, station
        from yam_arena.constants import FINGER_OPEN

        opening, tcps = [], []
        origin = self._base.scene.env_origins[0].cpu().numpy()
        for name, side in zip(ARMS, ("left", "right")):
            arm_ids, finger_ids = self._ids[name]
            q = self._base.scene[name].data.joint_pos.torch[0]
            opening.append(float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0)))
            tcps.append(ik.tcp_pose(q[arm_ids].cpu().numpy().astype(float), station.BASES[side])[0])
        return opening, tcps

    def _task_step(self):
        poses = self.poses()
        dz, dr, yaw = d.coupling_state(poses["shaft"], poses["head"])
        is_locked = (d.locked(poses["shaft"], poses["head"])
                     and d.release_closed(poses["shaft"], poses["plunger"]))
        opening, tcps = self._gripper_state(poses)
        speeds = {k: float(np.linalg.norm(poses[k][:3] - self._prev[k][:3])) / self._base.step_dt for k in FREE}
        self._prev = {k: v.copy() for k, v in poses.items()}
        result = self.scorer.update(poses, is_locked, opening, tcps, speeds)
        cart = poses["cartridge"]
        rh = d.quat_to_matrix(poses["shaft"][3:])
        guide_offset = rh.T @ (poses["plunger"][:3] - poses["shaft"][:3])
        lift = float(guide_offset[1])
        guide_rotation = rh.T @ d.quat_to_matrix(poses["plunger"][3:])
        guide_angle = math.degrees(math.acos(np.clip((np.trace(guide_rotation)-1)/2,-1,1)))
        info = {
            "score": result, "release_travel_mm": 1000 * lift, "detent_lift_mm": 1000 * lift, "success": self.scorer.success, "score_reason": self.scorer.reason,
            "guide_lateral_error_mm": 1000 * float(np.linalg.norm(guide_offset[[0,2]])),
            "guide_angle_error_deg": guide_angle,
            "locked": is_locked, "seat_mm": 1000 * dz, "pin_offset_mm": 1000 * dr, "head_yaw_deg": math.degrees(yaw),
            "cart_x_mm": 1000 * float(cart[0]), "cart_out": bool(self.scorer.extracted),
            "hooked_travel_mm": 1000 * self.scorer.hooked_travel,
            "grip_open": opening, "speeds": speeds, "hold_window_s": self.scorer.window_s,
        }
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 4), **info,
                                          "poses": {k: poses[k].round(6).tolist() for k in FREE},
                                          "tcps": [t.round(5).tolist() for t in tcps]}) + "\n")
        if round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
            print(f"[tool] t={self._t:.2f} locked={is_locked} seat={1000*dz:+.1f}mm off={1000*dr:.1f}mm "
                  f"yaw={math.degrees(yaw):+.1f} lift={1000*lift:.2f}mm cart_x={1000*cart[0]:.0f}mm score={result:.2f} ({self.scorer.reason})",
                  flush=True)
        done = self.scorer.success
        return StepResult(observation=self._observation(), terminated=done,
                          termination_reason="success" if done else None, info=info)

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()
