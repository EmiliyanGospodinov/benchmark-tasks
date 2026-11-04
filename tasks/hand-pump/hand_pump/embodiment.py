"""Free-body handheld pump probe. Same station, robot, observation and action contract."""
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
from .scoring import PumpScore

TASK_FILE = Path(__file__).resolve().parents[1] / "task.py"
PARTS = ("barrel", "piston", "chamber", "needle")
FREE = ("barrel", "piston")
ACTIVE = None  # read only by the declared privileged demonstration policy


class PumpEmbodiment(YamArenaEmbodiment):
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
            name="yam_hand_pump", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
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

    def _make_guide(self):
        from pxr import Gf, Sdf, UsdPhysics
        stage = self._base.sim.stage
        j = UsdPhysics.PrismaticJoint.Define(stage, "/World/PumpGuide")
        j.CreateJointEnabledAttr(False)
        j.CreateAxisAttr("X")
        j.CreateLocalPos0Attr(Gf.Vec3f(d.MIN_HANDLE_OFFSET, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0))
        j.CreateLowerLimitAttr(0.0)
        j.CreateUpperLimitAttr(d.STROKE)
        # The guide replaces the pump's internal bearing and piston bore. These
        # linked bodies can collide with the robot/table, but not with each other.
        j.CreateCollisionEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(True)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")  # seal drag, at the physics rate
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(float(d.SEAL_DAMPING))
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(5.0)
        j.CreateBody0Rel().SetTargets([Sdf.Path("/World/envs/env_0/barrel")])
        j.CreateBody1Rel().SetTargets([Sdf.Path("/World/envs/env_0/piston")])
        self._joint = j

    def poses(self):
        return {n: self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float)
                for n in PARTS}

    def _extension(self, poses):
        axis = d.quat_to_matrix(poses["barrel"][3:])[:, 0]
        x = float(np.dot(axis, poses["piston"][:3] - poses["barrel"][:3]) - d.MIN_HANDLE_OFFSET)
        return x, axis

    def reset(self, scene, *, seed=None):
        from .gas import PumpGas
        if self._joint is not None:
            self._joint.GetJointEnabledAttr().Set(False)
        if self.telemetry and self._trace is None:
            self.telemetry.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.telemetry.open("a", buffering=1)
        super().reset(scene, seed=seed)
        torch, base = self._torch, self._base
        # Reset only. Bodies and guide have mutually consistent frames.
        for name in PARTS:
            obj = base.scene[name]
            if name in FREE:
                obj.permanent_wrench_composer.reset()
            pose = torch.tensor(self.layout[name][None], dtype=torch.float32, device=base.device)
            pose[:, :3] += base.scene.env_origins
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))
        self._joint.GetJointEnabledAttr().Set(True)
        for _ in range(max(1, round(.5 / base.step_dt))):
            self._env.step(self._ready)
        for _ in range(4):
            base.sim.render()
        self.gas = PumpGas(math.pi*d.BORE_RADIUS**2, d.DEAD_VOLUME, d.STROKE, d.RECEIVER_VOLUME)
        poses = self.poses()
        self._x_previous, _ = self._extension(poses)
        self.gas.reset(self._x_previous)
        self._barrel_start = poses["barrel"][:3].copy()
        self._barrel_prev = poses["barrel"][:3].copy()
        self.scorer = PumpScore(self.info.control_hz)
        self._t = 0.0
        if self._trace:
            self._trace.write(json.dumps({"reset": True, "poses": {k: v.tolist() for k, v in poses.items()}})+"\n")
        return self._observation()

    def _task_step(self):
        poses = self.poses()
        x, axis = self._extension(poses)
        force = self.gas.update(x)
        self._x_previous = x
        # Gas reaction only: equal and opposite axial wrenches on piston and barrel (seal drag is the guide's
        # damping). Neither body is attached to the table or robot.
        torch = self._torch
        for name, sign in (("piston", 1.0), ("barrel", -1.0)):
            f = torch.tensor((sign*force*axis)[None, None], dtype=torch.float32, device=self._base.device)
            self._base.scene[name].permanent_wrench_composer.set_forces_and_torques_index(
                forces=f, torques=torch.zeros_like(f), is_global=True)
        self._show_gauge()
        grip_open, clear = self._gripper_state(poses)
        speed = float(np.linalg.norm(poses["barrel"][:3] - self._barrel_prev)) / self._base.step_dt
        self._barrel_prev = poses["barrel"][:3].copy()
        height = float(poses["barrel"][2] - d.BARREL_POS[2])
        on_mat = d.feet_on_mat(poses["barrel"] - np.r_[self._base.scene.env_origins[0].cpu().numpy(), 0, 0, 0, 0])
        rest = abs(height) < 0.015 and speed < 0.02 and on_mat  # put down on the mat and still
        score = self.scorer.update(self.gas.gauge_kpa, grip_open, clear, rest)
        info = {
            "score": score, "success": self.scorer.success, "score_reason": self.scorer.reason,
            "extension_mm": 1000*x, "pressure_kpa": self.gas.gauge_kpa, "gas_force_n": force,
            "barrel_shift_mm": 1000*float(np.linalg.norm(poses["barrel"][:3] - self._barrel_start)),
            "barrel_height_mm": 1000*height, "barrel_speed_mps": speed, "on_mat": on_mat, "pump_rest": rest,
            "grip_open": grip_open, "tcp_clear_m": clear, "hold_window_s": self.scorer.window_s,
        }
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 4), **info,
                "poses": {k: v.round(6).tolist() for k, v in poses.items()}})+"\n")
        if round(self._t*self.info.control_hz) % round(self.info.control_hz/2) == 0:
            print(f"[pump] t={self._t:.2f} x={1000*x:.1f}mm pressure={self.gas.gauge_kpa:.2f}kPa "
                  f"gas_force={force:.2f}N barrel_shift={info['barrel_shift_mm']:.1f}mm "
                  f"h={1000*height:.1f}mm score={score:.2f} ({self.scorer.reason})", flush=True)
        done = self.scorer.success
        return StepResult(observation=self._observation(), terminated=done,
                          termination_reason="success" if done else None, info=info)

    def _show_gauge(self):
        """Display only: turn the kinematic needle to the chamber pressure (no physical effect)."""
        torch, base = self._torch, self._base
        pose = np.r_[self.layout["needle"][:3], d.yaw_quat(d.needle_angle(self.gas.gauge_kpa))]
        t = torch.tensor(pose[None], dtype=torch.float32, device=base.device)
        t[:, :3] += base.scene.env_origins
        base.scene["needle"].write_root_pose_to_sim_index(root_pose=t)

    def _gripper_state(self, poses):
        from yam_arena import ik, station
        from yam_arena.constants import FINGER_OPEN

        handle = poses["piston"][:3]
        barrel = poses["barrel"][:3]
        origin = self._base.scene.env_origins[0].cpu().numpy()
        opening, clear = [], []
        for name, side in zip(ARMS, ("left", "right")):
            arm_ids, finger_ids = self._ids[name]
            q = self._base.scene[name].data.joint_pos.torch[0]
            opening.append(float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0)))
            tcp = ik.tcp_pose(q[arm_ids].cpu().numpy().astype(float), station.BASES[side])[0] + origin
            clear.append(float(min(np.linalg.norm(tcp - handle), np.linalg.norm(tcp - barrel))))
        return opening, clear

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()
