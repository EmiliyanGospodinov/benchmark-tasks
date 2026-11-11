"""Embodiment ``yam_lock_picking``: the supplied bimanual YAM station with an enlarged clear pin-tumbler trainer.

Robot, cameras, 14-D joint action and observations are the guide adapter's (``yam_arena``), unchanged. This
module adds the trainer bodies and their native PhysX joints:

- plug on a revolute joint in the fixed housing, 0..92 degrees, with a declared return spring (tailpiece);
- tension wrench on a revolute joint to the plug with a declared torsion spring (its flex = applied torque) and
  +-25 degree stops; a free-spinning sleeve on its end;
- three key pins on prismatic joints in the plug holes and three driver pins on prismatic joints in the housing
  chambers with declared compression springs.

Pins, plug surface, keyway and pick interact only by contact. Nothing is teleported after the reset.
Station integration adapted from our A26 ratchet-lowering task (MIT).

``diag=1`` (feasibility probes only): the pick is kinematic and posed by the probe script, and the wrench is
held by a stiff angular drive standing in for a hand. Never set in demos or evaluations.
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
PARTS = ("housing", "plug", "lever", "sleeve", "pick", "key0", "key1", "key2", "driver0", "driver1", "driver2")
ACTIVE = None  # read only by the declared privileged demonstration policy and the probe


class LockEmbodiment(YamArenaEmbodiment):
    def __init__(self, telemetry=None, diag=0, **kwargs):
        global ACTIVE
        kwargs.setdefault("task", str(TASK_FILE))
        kwargs.setdefault("object_jitter", 0)
        super().__init__(**kwargs)
        self.telemetry = Path(telemetry).expanduser() if telemetry else None
        self.diag = bool(int(diag))
        self.layout = d.start_poses()
        self._trace = None
        self._joints = None
        self.scorer = None
        self.last = {}
        ACTIVE = self

    # ------------------------------------------------------------------ build
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

        from .assets import BUILDERS, build_pick

        robot = BimanualYamEmbodiment(enable_cameras=True, rig=self.rig, wrist_cam_mount=self.wrist_cam_mount,
                                      contact_sensors=False, robot_contact_offset=0.001)
        objects = []
        for name in PARTS:
            pose = self.layout[name]
            usd = build_pick(kinematic=True) if (name == "pick" and self.diag) else BUILDERS[name]()
            objects.append(Object(name=name, object_type=ObjectType.RIGID, usd_path=str(usd),
                                  initial_pose=Pose(position_xyz=tuple(map(float, pose[:3])),
                                                    rotation_xyzw=tuple(map(float, pose[3:])))))
        objects.extend(task_object(o) for o in self.task.objects)
        for obj in objects:
            obj.initial_pose = Pose.from_dict(obj.initial_pose.to_dict())
            obj.enable_reset_pose()
        environment = IsaacLabArenaEnvironment(
            name="yam_lock_picking", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
            embodiment=robot, env_cfg_callback=env_cfg_callback(self.info.control_hz))
        self._env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(
            num_envs=1, solve_relations=False, device=self.device)).make_registered()
        self._base = self._env.unwrapped
        self._torch, self._goal = torch, goal
        self._free_objects = self.task.free_objects
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {n: (self._base.scene[n].find_joints(ARM_JOINTS, preserve_order=True)[0],
                         self._base.scene[n].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for n in ARMS}
        self._origin = self._base.scene.env_origins[0].cpu().numpy().astype(float)
        self._make_joints()

    def _joint(self, kind, path, axis, anchor0, body0, body1, limits=None):
        from pxr import Gf, Sdf, UsdPhysics

        cls = UsdPhysics.RevoluteJoint if kind == "rev" else UsdPhysics.PrismaticJoint
        j = cls.Define(self._base.sim.stage, path)
        j.CreateJointEnabledAttr(False)  # enabled after the start layout is written
        j.CreateAxisAttr(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, anchor0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0))
        if limits is not None:
            j.CreateLowerLimitAttr(float(limits[0]))
            j.CreateUpperLimitAttr(float(limits[1]))
        j.CreateCollisionEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(True)
        if body0:
            j.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/envs/env_0/{body0}")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/envs/env_0/{body1}")])
        return j

    @staticmethod
    def _drive(j, kind, stiffness, damping, target, cap):
        from pxr import UsdPhysics

        dr = UsdPhysics.DriveAPI.Apply(j.GetPrim(), kind)
        dr.CreateTypeAttr("force")
        dr.CreateStiffnessAttr(float(stiffness))
        dr.CreateDampingAttr(float(damping))
        dr.CreateTargetPositionAttr(float(target))
        dr.CreateMaxForceAttr(float(cap))
        return dr

    def _make_joints(self):
        J = {}
        J["plug"] = self._joint("rev", "/World/LockPlug", "X", d.AXIS + self._origin, None, "plug", d.THETA_LIMITS_DEG)
        self._drive(J["plug"], "angular", d.RETURN_STIFF_DEG, 2e-5, d.RETURN_TARGET_DEG, 1.0)
        J["lever"] = self._joint("rev", "/World/LockWrench", "X", (d.LEVER_X - d.FACE_X, 0, 0), "plug", "lever",
                                 (-d.FLEX_LIMIT_DEG, d.FLEX_LIMIT_DEG))
        self._drive(J["lever"], "angular", d.FLEX_STIFF_DEG, 2e-5, 0.0, 2.0)
        tip = d.sleeve_center(0.0) - np.array([d.LEVER_X, 0.0, d.AXIS_Z])
        J["sleeve"] = self._joint("rev", "/World/LockSleeve", "X", tip, "lever", "sleeve")
        for i, px in enumerate(d.PIN_X):
            J[f"key{i}"] = self._joint("pri", f"/World/LockKey{i}", "Z", (px, 0, d.KEY_BOTTOM0 + d.KEY_L / 2), "plug",
                                       f"key{i}", d.KEY_LIMITS)
            J[f"driver{i}"] = self._joint("pri", f"/World/LockDriver{i}", "Z",
                                          (px, 0, d.DRIVER_BOTTOM0 + d.DRIVER_L / 2), "housing", f"driver{i}",
                                          d.DRIVER_LIMITS)
            self._drive(J[f"driver{i}"], "linear", d.SPRING_K, 0.05, d.SPRING_TARGET, 5.0)
        if self.diag:  # probe only: a stiff "hand" on the wrench
            J["diag_hand"] = self._joint("rev", "/World/DiagHand", "X", np.array([d.LEVER_X, 0, d.AXIS_Z]) + self._origin,
                                         None, "lever")
            self._hand = self._drive(J["diag_hand"], "angular", 0.0, 0.0, 0.0, 5.0)
        self._joints = J

    # ------------------------------------------------------------------ reset
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
        for j in self._joints.values():
            j.GetJointEnabledAttr().Set(True)
        self._restore_layout()
        self._make_scorer()
        if self._trace:
            self._trace.write(json.dumps({"reset": True, "state": _jsonable(self.state())}) + "\n")
        return self._observation()

    def _make_scorer(self):
        from .scoring import LockScore
        self.scorer = LockScore(self.info.control_hz)

    def _restore_layout(self, settle_s=0.6):
        torch, base = self._torch, self._base
        for _ in range(2):
            for name in PARTS:
                self.set_pose(name, self.layout[name])
            for _ in range(max(1, round(settle_s / base.step_dt))):
                self._env.step(self._ready)
        for _ in range(4):
            base.sim.render()

    def set_pose(self, name, pose):
        torch, base = self._torch, self._base
        p = torch.tensor(np.asarray(pose, float)[None], dtype=torch.float32, device=base.device)
        p[:, :3] += base.scene.env_origins
        base.scene[name].write_root_pose_to_sim_index(root_pose=p)
        base.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))

    def set_hand(self, stiffness, target_deg):
        """Probe only: the stand-in hand on the wrench (N m/deg, lever angle in degrees)."""
        self._hand.GetStiffnessAttr().Set(float(stiffness))
        self._hand.GetDampingAttr().Set(float(stiffness) * 0.05)
        self._hand.GetTargetPositionAttr().Set(float(target_deg))

    # ------------------------------------------------------------------ state
    def poses(self) -> dict[str, np.ndarray]:
        out = {}
        for n in PARTS:
            p = self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float)
            p[:3] -= self._origin
            out[n] = p
        return out

    def state(self, p=None) -> dict:
        """Measured trainer state (telemetry, scorer, privileged demo)."""
        p = self.poses() if p is None else p
        rp = d.quat_to_matrix(p["plug"][3:])
        theta = math.degrees(d.angle_about_x(rp))
        phi = math.degrees(d.angle_about_x(d.quat_to_matrix(p["lever"][3:])))
        drivers, keys = [], []
        for i in range(3):
            drivers.append(float(p[f"driver{i}"][2] - d.AXIS_Z - d.DRIVER_L / 2))  # driver bottom above the axis
            rel = rp.T @ (p[f"key{i}"][:3] - d.AXIS)
            keys.append(float(rel[2] + d.KEY_L / 2))  # key pin top, radial (plug frame)
        pick_tip = p["pick"][:3]
        return {"theta_deg": theta, "flex_deg": phi - theta, "driver_bottom": drivers, "key_top": keys,
                "pick_tip": pick_tip, "pick_quat": p["pick"][3:], "sleeve": p["sleeve"][:3],
                "housing": p["housing"][:3]}

    def tcp(self):
        from yam_arena import ik, station
        from yam_arena.constants import FINGER_OPEN

        opening, tcps = [], []
        for name, side in zip(ARMS, ("left", "right")):
            arm_ids, finger_ids = self._ids[name]
            q = self._base.scene[name].data.joint_pos.torch[0]
            opening.append(float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0)))
            tcps.append(ik.tcp_pose(q[arm_ids].cpu().numpy().astype(float), station.BASES[side])[0])
        return opening, tcps

    def _task_step(self):
        s = self.state()
        grip_open, tcps = self.tcp()
        clear = [float(min(np.linalg.norm(t - s["sleeve"]), np.linalg.norm(t - d.AXIS))) for t in tcps]
        housing_moved = float(np.linalg.norm(s["housing"] - d.AXIS))
        info = {"theta_deg": s["theta_deg"], "flex_deg": s["flex_deg"], "driver_bottom_mm":
                [round(1000 * v, 2) for v in s["driver_bottom"]], "key_top_mm": [round(1000 * v, 2) for v in s["key_top"]],
                "pick_tip": [round(float(v), 4) for v in s["pick_tip"]], "grip_open": grip_open, "tcp_clear_m": clear,
                "score": 0.0, "success": False}
        done = False
        if self.scorer is not None:
            info["score"] = self.scorer.update(s["theta_deg"], s["flex_deg"], s["driver_bottom"], s["key_top"],
                                               [float(v) for v in s["pick_tip"]], grip_open, clear, housing_moved)
            info.update(self.scorer.info())
            done = self.scorer.success
        self.last = info
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 3), **_jsonable(info)}) + "\n")
        if round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
            print(f"[lock] t={self._t:6.2f} theta={s['theta_deg']:+6.2f} flex={s['flex_deg']:+6.2f} "
                  f"drv={info['driver_bottom_mm']} key={info['key_top_mm']} score={info['score']:.2f} "
                  f"{info.get('score_reason', '')}", flush=True)
        return StepResult(observation=self._observation(), terminated=done,
                          termination_reason="success" if done else None, info=info)

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v
