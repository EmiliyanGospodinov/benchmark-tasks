"""Embodiment ``yam_blade_change``: the supplied bimanual YAM station with an unpowered quick-release
blade holder, its installed blade, a replacement blade, a rack and a rest tray.

Robot, cameras, 14-D joint action and observations are the guide adapter's (``yam_arena``),
unchanged. This module adds the holder (on its spring carriage), the spring-return collar, the
spring-loaded locking pin and the two blades as rigid bodies. The collar lifts the pin only through
cam contact; the pin holds the tang only through contact. Nothing is teleported or driven.
Station wrapper, joint and reset handling adapted from our A5 / A7 tasks.
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
PARTS = ("holder", "collar", "pin", "old_blade", "new_blade")
ACTIVE = None  # read only by the declared privileged demonstration policy


class BladeEmbodiment(YamArenaEmbodiment):
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
            name="yam_blade_change", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
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

    def _joint(self, kind, path, axis, body0, pos0, body1, lower=None, upper=None):
        from pxr import Gf, Sdf, UsdPhysics

        cls = UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint
        j = cls.Define(self._base.sim.stage, path)
        j.CreateJointEnabledAttr(False)  # enabled after the start layout is written
        j.CreateAxisAttr(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0))
        if lower is not None:
            j.CreateLowerLimitAttr(float(lower))
            j.CreateUpperLimitAttr(float(upper))
        j.CreateCollisionEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(True)

        def bind():
            if body0:
                j.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/envs/env_0/{body0}")])
            j.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/envs/env_0/{body1}")])
        return j, bind

    @staticmethod
    def _spring(j, kind, stiffness, damping, target, max_force=1.0e4):
        from pxr import UsdPhysics

        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), kind)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(stiffness))
        drive.CreateDampingAttr(float(damping))
        drive.CreateTargetPositionAttr(float(target))
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(float(max_force))

    def _make_joints(self):
        origin = self._base.scene.env_origins[0].cpu().numpy()
        rail = np.array([d.AXIS_X, d.NOSE_Y, d.AXIS_Z]) + origin
        # carriage: holder slides along y on a bench rail; spring onto the +y stop (the tug gauge)
        car, b1 = self._joint("prismatic", "/World/HolderCarriage", "Y", None, rail, "holder", -d.CARRIAGE_TRAVEL, 0.0)
        self._spring(car, "linear", d.CARRIAGE_K, 2.0, 0.002)  # 2 mm past the stop: 0.3 N preload
        # collar: spring-return about the axis, stops at 0 and -travel
        col, b2 = self._joint("revolute", "/World/CollarBearing", "Y", "holder", (0, d.COLLAR_Y - d.NOSE_Y, 0), "collar",
                              -d.COLLAR_TRAVEL_DEG, 0.0)
        self._spring(col, "angular", d.COLLAR_K_DEG, 2.0e-5, d.COLLAR_PRELOAD_DEG)
        # locking pin: vertical guide in the holder, spring down onto its engaged stop
        pin, b3 = self._joint("prismatic", "/World/PinGuide", "Z", "holder", (0, d.PIN_Y - d.NOSE_Y, 0), "pin",
                              0.0, d.PIN_TRAVEL)
        self._spring(pin, "linear", d.PIN_K, 1.0, -d.PIN_PRELOAD / d.PIN_K)
        for bind in (b1, b2, b3):
            bind()  # bodies last
        self._joints = {"carriage": car, "collar": col, "pin": pin}

    def poses(self) -> dict[str, np.ndarray]:
        return {n: self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float) for n in PARTS}

    def reset(self, scene, *, seed=None):
        from .scoring import BladeScore

        if self._joints is not None:
            for j in self._joints.values():
                j.GetJointEnabledAttr().Set(False)
        if self.telemetry and self._trace is None:
            self.telemetry.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.telemetry.open("a", buffering=1)
        if self._env is None:
            self._ensure_sim(None)
        super().reset(scene, seed=seed)
        for j in self._joints.values():
            j.GetJointEnabledAttr().Set(True)
        self._restore_layout()
        self.scorer = BladeScore(self.info.control_hz)
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

    def state(self, p=None) -> dict:
        """Measured mechanism and blade state (holder-relative where it matters)."""
        p = self.poses() if p is None else p
        origin = self._base.scene.env_origins[0].cpu().numpy()
        rh = d.quat_to_matrix(p["holder"][3:])
        h = p["holder"][:3]
        out = {"carriage_ext_mm": 1000 * float(d.NOSE_Y + origin[1] - h[1]),
               "collar_deg": math.degrees(d.roll_y(rh.T @ d.quat_to_matrix(p["collar"][3:]))),
               "pin_lift_mm": 1000 * float((rh.T @ (p["pin"][:3] - h))[2])}
        for n in ("old_blade", "new_blade"):
            rb = d.quat_to_matrix(p[n][3:])
            rel = rh.T @ (p[n][:3] - h)  # shoulder centre in the holder frame (0 = seated)
            rr = rh.T @ rb
            out[n] = {"rel_mm": (1000 * rel).tolist(), "roll_deg": math.degrees(d.roll_y(rr)),
                      "yaw_deg": math.degrees(math.atan2(rr[1, 0], rr[0, 0])),
                      "pitch_deg": math.degrees(math.asin(float(np.clip(rr[2, 1], -1, 1)))),
                      "world": (p[n][:3] + rb @ np.array([0.0, -(d.BODY_L - d.TANG_L) / 2, 0.0])
                                - origin - [0, 0, d.TOP]).tolist()}  # blade middle
        out["speeds"] = {n: float(np.linalg.norm(self._base.scene[n].data.root_lin_vel_w.torch[0].cpu().numpy()))
                         for n in PARTS}
        return out

    def _task_step(self):
        p = self.poses()
        st = self.state(p)
        grip_open, clear, new_grip_dist, mechanism_clear = self._gripper_state(p)
        st["new_grip_dist_m"], st["mechanism_clear_m"] = new_grip_dist, mechanism_clear
        score = self.scorer.update(st, grip_open, clear)
        info = {**{k: v for k, v in st.items()}, "grip_open": grip_open, "tcp_clear_m": clear,
                "hold_window_s": self.scorer.window_s, "score": score, "score_reason": self.scorer.reason,
                "success": self.scorer.success}
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 3), **info,
                                          "poses": {k: v.round(5).tolist() for k, v in p.items()}}) + "\n")
        if round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
            ob, nb = st["old_blade"]["rel_mm"], st["new_blade"]["rel_mm"]
            print(f"[blade] t={self._t:5.2f} collar={st['collar_deg']:6.1f} pin={st['pin_lift_mm']:5.1f}mm "
                  f"carriage={st['carriage_ext_mm']:5.1f}mm old_rel=({ob[0]:.0f},{ob[1]:.0f},{ob[2]:.0f}) "
                  f"new_rel=({nb[0]:.0f},{nb[1]:.0f},{nb[2]:.0f}) score={score:.2f} ({self.scorer.reason})", flush=True)
        done = self.scorer.success
        return StepResult(observation=self._observation(), terminated=done,
                          termination_reason="success" if done else None, info=info)

    def tcp(self, side: str) -> np.ndarray:
        """World TCP position from the arm's measured joints (for the privileged demonstration)."""
        from yam_arena import ik, station

        name = dict(zip(("left", "right"), ARMS))[side]
        arm_ids, _ = self._ids[name]
        q = self._base.scene[name].data.joint_pos.torch[0][arm_ids].cpu().numpy().astype(float)
        return ik.tcp_pose(q, station.BASES[side])[0] + self._base.scene.env_origins[0].cpu().numpy()

    def _gripper_state(self, p):
        """Measured openings and TCP distances for release and the blade-tug grasp proxy."""
        from yam_arena import ik, station
        from yam_arena.constants import FINGER_OPEN

        origin = self._base.scene.env_origins[0].cpu().numpy()
        rc = d.quat_to_matrix(p["collar"][3:])
        sites = [p["collar"][:3] + rc @ np.array([0.0, 0.0, d.LEVER_R - 0.008])]
        rh = d.quat_to_matrix(p["holder"][3:])
        rp = d.quat_to_matrix(p["pin"][3:])
        mechanism_sites = [p["holder"][:3] + rh @ np.array([0.0, d.HOLDER_LEN / 2, 0.0]),
                           sites[0], p["pin"][:3] + rp @ np.array([d.FOLLOWER_X / 2, 0.0,
                                                                  d.follower_bottom_rest() + 0.005])]
        for n in ("old_blade", "new_blade"):
            rb = d.quat_to_matrix(p[n][3:])
            for s in (d.GRIP_FROM_SHOULDER, d.BODY_L / 2):  # grip region and body centre
                sites.append(p[n][:3] + rb @ np.array([0.0, -s, 0.0]))
        new_grip = p["new_blade"][:3] + d.quat_to_matrix(p["new_blade"][3:]) @ np.array(
            [0.0, -d.GRIP_FROM_SHOULDER, 0.0])
        opening, clear, new_grip_dist, mechanism_clear = [], [], [], []
        for name, side in zip(ARMS, ("left", "right")):
            arm_ids, finger_ids = self._ids[name]
            q = self._base.scene[name].data.joint_pos.torch[0]
            opening.append(float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0)))
            tcp = ik.tcp_pose(q[arm_ids].cpu().numpy().astype(float), station.BASES[side])[0] + origin
            clear.append(float(min(np.linalg.norm(tcp - s) for s in sites)))
            new_grip_dist.append(float(np.linalg.norm(tcp - new_grip)))
            mechanism_clear.append(float(min(np.linalg.norm(tcp - s) for s in mechanism_sites)))
        return opening, clear, new_grip_dist, mechanism_clear

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()
