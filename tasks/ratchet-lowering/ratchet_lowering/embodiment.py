"""Embodiment ``yam_ratchet_lowering``: the supplied bimanual YAM station with a ratchet lowering trainer.

Robot, cameras, 14-D joint action and observations are the guide adapter's (``yam_arena``), unchanged. This
module adds the trainer bodies and three native revolute joints: the rotor on its shaft (no drive, no
damping), the pawl on its pivot (a declared torsion spring) and the crank handle sleeve on its pin. The
load is held only by contact between the pawl nose and the ratchet teeth. Nothing is teleported.

Station integration adapted from our A5 inspection-mount task (yambition-a5-inspection-mount @ b48bac2,
mount_aim/embodiment.py, MIT).

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
PARTS = ("frame", "rotor", "handle", "pawl")
ACTIVE = None  # read only by the declared privileged demonstration policy


class RatchetEmbodiment(YamArenaEmbodiment):
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
            name="yam_ratchet_lowering", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
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
        from .clearance import Clearance
        self._clearance = Clearance()

    def _revolute(self, path, anchor0, body0, body1, lower=None, upper=None):
        from pxr import Gf, Sdf, UsdPhysics

        j = UsdPhysics.RevoluteJoint.Define(self._base.sim.stage, path)
        j.CreateJointEnabledAttr(False)  # enabled after the start layout is written
        j.CreateAxisAttr("Y")
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, anchor0)))
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

    def _make_joints(self):
        from pxr import UsdPhysics

        origin = self._base.scene.env_origins[0].cpu().numpy()
        lo, hi = d.THETA_LIMIT_DEG
        shaft, b1 = self._revolute("/World/RatchetShaft", d.SHAFT + origin, None, "rotor", -hi, -lo)
        pawl, b2 = self._revolute("/World/RatchetPawl", d.world(d.PAWL_PIVOT, d.PAWL_Y) + origin, None, "pawl",
                                  *d.PAWL_LIMITS_DEG)
        a = math.radians(d.CRANK_LOCAL_DEG)
        pin = (d.CRANK_R * math.cos(a), d.HANDLE_Y, d.CRANK_R * math.sin(a))
        sleeve, b3 = self._revolute("/World/CrankHandle", pin, "rotor", "handle")
        spring = UsdPhysics.DriveAPI.Apply(pawl.GetPrim(), "angular")  # declared pawl torsion spring
        spring.CreateTypeAttr("force")
        spring.CreateStiffnessAttr(float(d.PAWL_SPRING))
        spring.CreateDampingAttr(1.0e-4)
        spring.CreateTargetPositionAttr(float(d.PAWL_SPRING_TARGET_DEG))
        spring.CreateMaxForceAttr(1.0)
        self._spring = spring
        for bind in (b1, b2, b3):
            bind()
        self._joints = {"shaft": shaft, "pawl": pawl, "handle": sleeve}

    def poses(self) -> dict[str, np.ndarray]:
        return {n: self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float) for n in PARTS}

    def reset(self, scene, *, seed=None):
        if self._joints is not None:
            for j in self._joints.values():
                j.GetJointEnabledAttr().Set(False)
            self._spring.GetTargetPositionAttr().Set(float(d.PAWL_SPRING_TARGET_DEG))
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
            self._trace.write(json.dumps({"reset": True, "poses": {k: v.round(6).tolist()
                                                                   for k, v in self.poses().items()}}) + "\n")
        return self._observation()

    def _make_scorer(self):
        try:
            from .scoring import RatchetScore
        except ImportError:  # scorer not written yet (feasibility probe)
            self.scorer = None
            return
        self.scorer = RatchetScore(self.info.control_hz)

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
        """Measured mechanism state (used by telemetry, the scorer and the privileged demo)."""
        p = self.poses() if p is None else p
        rr, rp = d.quat_to_matrix(p["rotor"][3:]), d.quat_to_matrix(p["pawl"][3:])
        w = self._base.scene["rotor"].data.root_ang_vel_w.torch[0].cpu().numpy()
        return {"theta_deg": math.degrees(d.theta_of(rr)), "omega_dps": -math.degrees(float(w[1])),
                "pawl_deg": math.degrees(d.pawl_angle_of(rp)),
                "handle": p["handle"][:3] - self._base.scene.env_origins[0].cpu().numpy(),
                "tab": d.tab_center(d.pawl_angle_of(rp))}

    def _gripper_state(self, s):
        from yam_arena import ik, station
        from yam_arena.constants import FINGER_OPEN

        opening, clear = [], []
        for name, side in zip(ARMS, ("left", "right")):
            arm_ids, finger_ids = self._ids[name]
            q = self._base.scene[name].data.joint_pos.torch[0]
            opening.append(float(np.clip(q[finger_ids].mean().item() / FINGER_OPEN, 0.0, 1.0)))
            tcp = ik.tcp_pose(q[arm_ids].cpu().numpy().astype(float), station.BASES[side])[0]
            clear.append(float(min(np.linalg.norm(tcp - s["handle"]), np.linalg.norm(tcp - s["tab"]))))
        return opening, clear

    def _task_step(self):
        poses = self.poses()
        s = self.state(poses)
        clearance, tab_sdf, sleeve_sdf = self._clearance.measure(self._base.scene, poses, ARMS)
        grip_open, clear = self._gripper_state(s)
        info = {"theta_deg": s["theta_deg"], "omega_dps": s["omega_dps"], "pawl_deg": s["pawl_deg"],
                "grip_open": grip_open, "tcp_clear_m": clear, "mechanism_clear_m": clearance,
                "closest_collision_pair": self._clearance.closest_pair,
                "tab_vertex_sdf_m": tab_sdf, "sleeve_vertex_sdf_m": sleeve_sdf,
                "score": 0.0, "success": False}
        done = False
        if self.scorer is not None:
            info["score"] = self.scorer.update(s["theta_deg"], s["omega_dps"], s["pawl_deg"], grip_open, clear, clearance)
            info.update(self.scorer.info())
            done = self.scorer.success
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 3), **info}) + "\n")
        if round(self._t * self.info.control_hz) % round(self.info.control_hz / 4) == 0:
            print(f"[ratchet] t={self._t:5.2f} theta={s['theta_deg']:+7.2f}deg omega={s['omega_dps']:+7.1f}dps "
                  f"pawl={s['pawl_deg']:+6.2f}deg score={info['score']:.2f}", flush=True)
        return StepResult(observation=self._observation(), terminated=done,
                          termination_reason="success" if done else None, info=info)

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()
