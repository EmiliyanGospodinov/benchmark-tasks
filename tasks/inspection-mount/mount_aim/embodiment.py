"""Embodiment ``yam_inspection_mount``: the supplied bimanual YAM station with an adjustable
inspection arm on a bench post.

Robot, cameras, 14-D joint action and observations are the guide adapter's (``yam_arena``),
unchanged. This module adds the post (fixed), the arm with its head and the clamp knob, their
hinge joints and the clamp model of design.py: the rosette lock engages from the knob's measured
rotation only. Nothing is teleported during an episode.
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
from .scoring import MountScore

TASK_FILE = Path(__file__).resolve().parents[1] / "task.py"
PARTS = ("post", "arm", "knob")
ACTIVE = None  # read only by the declared privileged demonstration policy


def _quatf(r):
    from pxr import Gf

    q = d.matrix_to_quat(r)
    return Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3])))


class MountEmbodiment(YamArenaEmbodiment):
    def __init__(self, telemetry=None, **kwargs):
        global ACTIVE
        kwargs.setdefault("task", str(TASK_FILE))
        kwargs.setdefault("object_jitter", 0)
        super().__init__(**kwargs)
        self.telemetry = Path(telemetry).expanduser() if telemetry else None
        self.layout = d.start_poses()
        self._trace = None
        self._joints = None
        self.scorer = None
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
            name="yam_inspection_mount", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
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

    def _revolute(self, path, axis, anchor, body, lower=None, upper=None):
        from pxr import Gf, Sdf, UsdPhysics

        origin = self._base.scene.env_origins[0].cpu().numpy()
        j = UsdPhysics.RevoluteJoint.Define(self._base.sim.stage, path)
        j.CreateJointEnabledAttr(False)  # enabled after the start layout is written
        j.CreateAxisAttr(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, np.asarray(anchor) + origin)))  # body0 = world
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0))
        if lower is not None:
            j.CreateLowerLimitAttr(float(lower))
            j.CreateUpperLimitAttr(float(upper))
        j.CreateCollisionEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(True)
        return j, (lambda: j.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/envs/env_0/{body}")]))

    @staticmethod
    def _drive(j, stiffness, damping, max_force):
        from pxr import UsdPhysics

        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(stiffness))
        drive.CreateDampingAttr(float(damping))
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(float(max_force))

    def _make_joints(self):
        pitch, b1 = self._revolute("/World/MountPitch", "X", d.HINGE, "arm", d.PHI_UP_DEG, d.PHI_REST_DEG + 1.0)
        lock, b2 = self._revolute("/World/MountLock", "X", d.HINGE, "arm")
        self._drive(lock, d.LOCK_STIFFNESS, d.LOCK_DAMPING, d.LOCK_TORQUE)
        knob, b3 = self._revolute("/World/KnobJoint", "Z", d.KNOB_CENTER, "knob", -d.KNOB_STOP_DEG, 2.0)
        self._drive(knob, 0.0, d.KNOB_DAMPING, d.KNOB_MAX_TORQUE)
        for bind in (b1, b2, b3):
            bind()  # bodies last
        self._joints = {"pitch": pitch, "lock": lock, "knob": knob}

    def poses(self) -> dict[str, np.ndarray]:
        return {n: self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float) for n in PARTS}

    def reset(self, scene, *, seed=None):
        if self._joints is not None:
            for j in self._joints.values():
                j.GetJointEnabledAttr().Set(False)
        if self.telemetry and self._trace is None:
            self.telemetry.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.telemetry.open("a", buffering=1)
        if self._env is None:
            self._ensure_sim(None)
        super().reset(scene, seed=seed)
        self._joints["pitch"].GetJointEnabledAttr().Set(True)
        self._joints["knob"].GetJointEnabledAttr().Set(True)
        self._restore_layout()
        self.locked, self.tooth_deg, self.tightened_deg = False, None, 0.0
        self.scorer = MountScore(self.info.control_hz)
        self._last_knob = d.yaw_of(d.quat_to_matrix(self.poses()["knob"][3:]))
        if self._trace:
            self._trace.write(json.dumps({"reset": True}) + "\n")
        return self._observation()

    def _restore_layout(self, settle_s=0.5):
        torch, base = self._torch, self._base
        for _ in range(2):
            for name in PARTS:
                pose = torch.tensor(self.layout[name][None], dtype=torch.float32, device=base.device)
                pose[:, :3] += base.scene.env_origins
                base.scene[name].write_root_pose_to_sim_index(root_pose=pose)
                base.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))
            for _ in range(max(1, round(settle_s / base.step_dt))):
                self._env.step(self._ready)
        for _ in range(4):
            base.sim.render()

    def _task_step(self):
        p = self.poses()
        ra, rk = d.quat_to_matrix(p["arm"][3:]), d.quat_to_matrix(p["knob"][3:])
        yk = d.yaw_of(rk)
        self.tightened_deg -= math.degrees((yk - self._last_knob + math.pi) % (2 * math.pi) - math.pi)
        self._last_knob = yk
        phi = math.degrees(d.phi_of(ra))
        if not self.locked and self.tightened_deg >= d.ENGAGE_DEG:
            self.tooth_deg = d.TOOTH_DEG * round(phi / d.TOOTH_DEG)  # teeth mesh at the nearest tooth
            self._joints["lock"].GetLocalRot0Attr().Set(_quatf(d.rot_x(math.radians(self.tooth_deg))))
            self._joints["lock"].GetJointEnabledAttr().Set(True)
            self.locked = True
            print(f"[mount] t={self._t:5.2f} rosette meshed at elevation {-self.tooth_deg:+.1f} deg "
                  f"(arm at {-phi:+.2f}, knob {self.tightened_deg:.0f} deg)", flush=True)
        elif self.locked and self.tightened_deg < d.RELEASE_DEG:
            self._joints["lock"].GetJointEnabledAttr().Set(False)
            self.locked = False
            print(f"[mount] t={self._t:5.2f} rosette released (knob {self.tightened_deg:.0f} deg)", flush=True)
        grip_open, clear = self._gripper_state(p)
        score = self.scorer.update(-phi if math.isfinite(phi) else float("nan"), self.locked, grip_open, clear)
        info = {"elevation_deg": -phi, "knob_deg": self.tightened_deg, "locked": self.locked,
                "tooth_elevation_deg": None if self.tooth_deg is None else -self.tooth_deg,
                "grip_open": grip_open, "tcp_clear_m": clear, "hold_window_s": self.scorer.window_s,
                "score": score, "score_reason": self.scorer.reason, "success": self.scorer.success}
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 3), **info,
                                          "poses": {k: v.round(5).tolist() for k, v in p.items()}}) + "\n")
        if round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
            print(f"[mount] t={self._t:5.2f} elev={-phi:+6.2f}deg knob={self.tightened_deg:6.1f}deg "
                  f"locked={self.locked} score={score:.2f} ({self.scorer.reason})", flush=True)
        done = self.scorer.success
        return StepResult(observation=self._observation(), terminated=done,
                          termination_reason="success" if done else None, info=info)

    def _gripper_state(self, p):
        """Measured gripper openings (0..1) and TCP distances to the nearest of head and knob centre."""
        from yam_arena import ik, station
        from yam_arena.constants import FINGER_OPEN

        head = p["arm"][:3] + d.quat_to_matrix(p["arm"][3:]) @ np.array([0.0, -d.ARM_L, 0.0])  # world
        knob = p["knob"][:3]
        opening, clear = [], []
        for name, side in zip(ARMS, ("left", "right")):
            arm_ids, finger_ids = self._ids[name]
            q = self._base.scene[name].data.joint_pos.torch[0]
            opening.append(float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0)))
            origin = self._base.scene.env_origins[0].cpu().numpy()
            tcp = ik.tcp_pose(q[arm_ids].cpu().numpy().astype(float), station.BASES[side])[0] + origin
            clear.append(float(min(np.linalg.norm(tcp - head), np.linalg.norm(tcp - knob))))
        return opening, clear

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()
