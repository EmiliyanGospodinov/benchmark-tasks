"""Embodiment ``yam_mechanical_puller``: the guide's bimanual YAM station plus the puller fixture.

Robot, cameras, 14-D joint action and observations are the guide adapter's (``yam_arena``), unchanged.
Station wrapper adapted from our A5 task (f7e6030). Probe stage (A10 step 1): an isolated rotating-nut
screw — handle revolute and ram prismatic in a fixed frame, coupled by PhysX's native rack-and-pinion joint —
pushing a spring-held anvil (its displacement is the force gauge). Nothing is teleported during an episode.
"""

from __future__ import annotations

import atexit
import json
import math
from pathlib import Path

import numpy as np
from inspect_robots.types import StepResult
from yam_arena import inspect_embodiment as upstream
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, READY
from yam_arena.inspect_embodiment import ARMS, YamArenaEmbodiment

from . import design as d

TASK_FILE = Path(__file__).resolve().parents[1] / "task.py"
PROBE_PARTS = ("frame", "handle", "ram", "anvil")
DRAFT_PARTS = ("fixture", "collar", "body", "handle", "ram", "jaw_pos", "jaw_neg")
ACTIVE = None


def _quat_to_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def probe_layout():
    f = d.PROBE_FRAME
    return {
        "frame": np.r_[f, 0, 0, 0, 1.0],
        "handle": np.r_[f + [-0.025, 0, 0], 0, 0, 0, 1.0],
        "ram": np.r_[f, 0, 0, 0, 1.0],
        "anvil": np.r_[f + [0.044 + d.ANVIL_GAP + 0.005, 0, 0], 0, 0, 0, 1.0],
    }


class PullerEmbodiment(YamArenaEmbodiment):
    def __init__(self, telemetry=None, ratio=None, scene_kind="prepared", **kwargs):
        global ACTIVE
        kwargs.setdefault("task", str(TASK_FILE))
        kwargs.setdefault("object_jitter", 0)
        super().__init__(**kwargs)
        self.telemetry = Path(telemetry).expanduser() if telemetry else None
        self.ratio = float(ratio) if ratio is not None else d.SCREW_RATIO
        self.kind = str(scene_kind)
        self.parts = PROBE_PARTS if self.kind == "probe" else DRAFT_PARTS
        self.layout = probe_layout() if self.kind == "probe" else (d.prepared_layout() if self.kind=="prepared" else d.draft_layout())
        self._trace = None
        self._joints = None
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
                                      contact_sensors=True, robot_contact_offset=0.001)
        objects = [Object(name=n, object_type=ObjectType.RIGID, usd_path=str(BUILDERS[n]()),
                          initial_pose=Pose(position_xyz=tuple(map(float, self.layout[n][:3])),
                                            rotation_xyzw=tuple(map(float, self.layout[n][3:]))))
                   for n in self.parts]
        objects.extend(task_object(o) for o in self.task.objects)
        for obj in objects:
            obj.initial_pose = Pose.from_dict(obj.initial_pose.to_dict())
            obj.enable_reset_pose()
        environment = IsaacLabArenaEnvironment(
            name="yam_mechanical_puller", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
            embodiment=robot, env_cfg_callback=env_cfg_callback(self.info.control_hz))
        self._env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(
            num_envs=1, solve_relations=False, device=self.device)).make_registered()
        self._base = self._env.unwrapped
        self._torch, self._goal = torch, goal
        self._free_objects = self.task.free_objects
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {n: (self._base.scene[n].find_joints(ARM_JOINTS, preserve_order=True)[0],
                         self._base.scene[n].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for n in ARMS}
        self._make_joints() if self.kind == "probe" else self._make_draft_joints()

    def _make_draft_joints(self):
        from pxr import Gf, Sdf, UsdPhysics

        stage = self._base.sim.stage
        path = lambda n: Sdf.Path(f"/World/envs/env_0/{n}")  # noqa: E731
        origin = self._base.scene.env_origins[0].cpu().numpy()
        lay = self.layout

        def joint(kind, name, b0, b1, axis, lower=None, upper=None):
            j = kind.Define(stage, f"/World/{name}")
            j.CreateJointEnabledAttr(False)
            j.CreateAxisAttr(axis)
            p0 = lay[b1][:3] - lay[b0][:3] if b0 else lay[b1][:3] + origin
            if b1 in ("jaw_pos","jaw_neg"):
                p0=np.array([d.JAW_OPEN if b1=="jaw_pos" else -d.JAW_OPEN,0.,0.])
            j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, p0)))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0))
            if lower is not None:
                j.CreateLowerLimitAttr(float(lower))
                j.CreateUpperLimitAttr(float(upper))
            j.CreateCollisionEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)
            return j, (b0, b1)

        def drive(j, kind, k, c, cap):
            dr = UsdPhysics.DriveAPI.Apply(j[0].GetPrim(), kind)
            dr.CreateTypeAttr("force")
            dr.CreateStiffnessAttr(float(k))
            dr.CreateDampingAttr(float(c))
            dr.CreateTargetPositionAttr(0.0)
            dr.CreateTargetVelocityAttr(0.0)
            dr.CreateMaxForceAttr(float(cap))
            return dr

        R, P = UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint
        J = {
            "nut": joint(R, "PullerNut", "body", "handle", "Z"),
            "ram": joint(P, "PullerRam", "body", "ram", "Z", -0.060, 0.002),
            "jaw_pos": joint(P, "JawPos", "body", "jaw_pos", "X", -(d.JAW_OPEN - d.JAW_CLOSED) - 0.006, 0.002),
            "jaw_neg": joint(P, "JawNeg", "body", "jaw_neg", "X", -0.002, (d.JAW_OPEN - d.JAW_CLOSED) + 0.006),
            "collar": joint(P, "CollarFit", None, "collar", "Z", -0.001, d.COLLAR_CLEAR + 0.01),
        }
        self._screw = drive(J["ram"], "linear", d.SCREW_K, d.SCREW_DAMPING, d.SCREW_MAX_FORCE)
        # stick-slip (elasto-plastic Coulomb) friction, from our A1 bearing: anchor slides past the window
        self._stick = {
            "nut": [drive(J["nut"], "angular", 0.05 / 2.0, 1e-4, 0.05), "ang", 2.0, 0.0],  # 0.05 N m, 2 deg window
            "jaw_pos": [drive(J["jaw_pos"], "linear", d.JAW_HOLD / d.STICK_WINDOW, 2.0, d.JAW_HOLD), "lin",
                        d.STICK_WINDOW, 0.0],
            "jaw_neg": [drive(J["jaw_neg"], "linear", d.JAW_HOLD / d.STICK_WINDOW, 2.0, d.JAW_HOLD), "lin",
                        d.STICK_WINDOW, 0.0],
            "collar": [drive(J["collar"], "linear", d.COLLAR_HOLD / d.STICK_WINDOW, 5.0, d.COLLAR_HOLD), "lin",
                       d.STICK_WINDOW, 0.0],
        }
        for j, (b0, b1) in J.values():
            if b0:
                j.CreateBody0Rel().SetTargets([path(b0)])
            j.CreateBody1Rel().SetTargets([path(b1)])  # bodies last
        self._joints = {k: v[0] for k, v in J.items()}

    def _coords(self, p):
        rb = _quat_to_matrix(p["body"][3:])
        rel = lambda n: rb.T @ (p[n][:3] - p["body"][:3])  # noqa: E731
        return {"jaw_pos": float(rel("jaw_pos")[0] - d.JAW_OPEN), "jaw_neg": float(rel("jaw_neg")[0] + d.JAW_OPEN),
                "ram": float(rel("ram")[2]), "collar": float(p["collar"][2] - (d.TOP + d.COLLAR_Z0)),
                "nut": math.degrees(self._angle_draft(p))}

    @staticmethod
    def _angle_draft(p):
        r = _quat_to_matrix(p["body"][3:]).T @ _quat_to_matrix(p["handle"][3:])
        return math.atan2(r[1, 0], r[0, 0])  # about the screw axis (body z)

    def _make_joints(self):
        from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

        stage = self._base.sim.stage
        path = lambda n: Sdf.Path(f"/World/envs/env_0/{n}")  # noqa: E731
        origin = self._base.scene.env_origins[0].cpu().numpy()
        lay = self.layout

        def joint(kind, name, b0, b1, axis="X"):
            j = kind.Define(stage, f"/World/{name}")
            j.CreateJointEnabledAttr(False)
            j.CreateAxisAttr(axis)
            p0 = (lay[b1][:3] - (lay[b0][:3] if b0 else -origin))  # child origin in the parent (world) frame
            j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, p0)))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0))
            j.CreateCollisionEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)
            return j

        nut = joint(UsdPhysics.RevoluteJoint, "PullerNut", "frame", "handle")
        slide = joint(UsdPhysics.PrismaticJoint, "PullerSlide", "frame", "ram")
        slide.CreateLowerLimitAttr(-0.002)
        slide.CreateUpperLimitAttr(float(d.RAM_TRAVEL))
        motor = UsdPhysics.DriveAPI.Apply(nut.GetPrim(), "angular")  # probe stand-in for a turning hand
        motor.CreateTypeAttr("force")
        motor.CreateStiffnessAttr(0.0)
        motor.CreateDampingAttr(0.01)
        motor.CreateTargetVelocityAttr(float(d.PROBE_SPEED_DEG))
        motor.CreateMaxForceAttr(float(d.PROBE_TORQUE))
        gauge = joint(UsdPhysics.PrismaticJoint, "AnvilSpring", None, "anvil")
        spring = UsdPhysics.DriveAPI.Apply(gauge.GetPrim(), "linear")
        spring.CreateTypeAttr("force")
        spring.CreateStiffnessAttr(float(d.ANVIL_K))
        spring.CreateDampingAttr(20.0)
        spring.CreateTargetPositionAttr(0.0)
        spring.CreateMaxForceAttr(1000.0)
        screw = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")  # constitutive screw coupling
        screw.CreateTypeAttr("force")
        screw.CreateStiffnessAttr(float(d.SCREW_K))
        screw.CreateDampingAttr(float(d.SCREW_DAMPING))
        screw.CreateTargetPositionAttr(0.0)
        screw.CreateMaxForceAttr(float(d.SCREW_MAX_FORCE))
        self._screw = screw
        for j, b0, b1 in ((nut, "frame", "handle"), (slide, "frame", "ram"), (gauge, None, "anvil")):
            if b0:
                j.CreateBody0Rel().SetTargets([path(b0)])
            j.CreateBody1Rel().SetTargets([path(b1)])  # bodies last
        self._joints = {"nut": nut, "slide": slide, "gauge": gauge}

    def poses(self):
        return {n: self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float) for n in self.parts}

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
        torch, base = self._torch, self._base
        for name in self.parts:
            pose = torch.tensor(self.layout[name][None], dtype=torch.float32, device=base.device)
            pose[:, :3] += base.scene.env_origins
            base.scene[name].write_root_pose_to_sim_index(root_pose=pose)
            base.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))
        for j in self._joints.values():
            j.GetJointEnabledAttr().Set(True)
        p = self.poses()
        self._t = 0.0
        from .scoring import PullerScore
        self.scorer = PullerScore(self.info.control_hz)
        if self.kind == "probe":
            self._turns, self._last_angle = 0.0, self._angle(p)
            self._x0 = self._ram_x(p)
            self._anvil0 = float(p["anvil"][0])
        else:
            self._turns, self._last_angle = 0.0, self._angle_draft(p)
            for k, s in self._stick.items():
                s[3] = ((d.JAW_CLOSED-d.JAW_OPEN)*(1 if k=="jaw_pos" else -1) if self.kind=="prepared" and k.startswith("jaw_") else 0.0)
                s[0].GetTargetPositionAttr().Set(s[3])
            self._screw.GetTargetPositionAttr().Set(0.0)
            self.collar_free = False
        if self._trace:
            self._trace.write(json.dumps({"reset": True}) + "\n")
        return self._observation()

    @staticmethod
    def _angle(p):
        r = _quat_to_matrix(p["frame"][3:]).T @ _quat_to_matrix(p["handle"][3:])
        return math.atan2(r[2, 1], r[1, 1])  # rotation about the screw axis (x)

    @staticmethod
    def _ram_x(p):
        return float((_quat_to_matrix(p["frame"][3:]).T @ (p["ram"][:3] - p["frame"][:3]))[0])

    def _task_step(self):
        if self.kind != "probe":
            return self._draft_step()
        p = self.poses()
        a = self._angle(p)
        self._turns += ((a - self._last_angle + math.pi) % (2 * math.pi) - math.pi) / (2 * math.pi)
        self._last_angle = a
        ram = self._ram_x(p) - self._x0
        target = d.LEAD * self._turns
        self._screw.GetTargetPositionAttr().Set(float(target))
        force = float(np.clip(d.SCREW_K * (target - ram), -d.SCREW_MAX_FORCE, d.SCREW_MAX_FORCE))
        torque = -force * d.LEAD / (2 * math.pi)  # load resists turning (equal/opposite on the frame)
        t = self._torch
        ax = _quat_to_matrix(p["frame"][3:])[:, 0]
        tq = t.tensor((torque * ax)[None, None], dtype=t.float32, device=self._base.device)
        self._base.scene["handle"].permanent_wrench_composer.set_forces_and_torques_index(
            forces=t.zeros_like(tq), torques=tq, is_global=True)
        anvil = float(p["anvil"][0]) - self._anvil0
        info = {"handle_turns": self._turns, "ram_mm": 1000 * ram, "screw_force_n": force, "anvil_mm": 1000 * anvil,
                "anvil_force_n": d.ANVIL_K * anvil, "score": 0.0}
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 3), **info}) + "\n")
        if round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
            print(f"[puller] t={self._t:5.2f} turns={self._turns:+.3f} ram={1000 * ram:+.3f}mm "
                  f"anvil={1000 * anvil:+.3f}mm gauge={d.ANVIL_K * anvil:+.2f}N screw={force:+.2f}N", flush=True)
        return StepResult(observation=self._observation(), terminated=False, termination_reason=None, info=info)

    def _draft_step(self):
        p = self.poses()
        a = self._angle_draft(p)
        self._turns += -((a - self._last_angle + math.pi) % (2 * math.pi) - math.pi) / (2 * math.pi)  # clockwise +
        self._last_angle = a
        c = self._coords(p)
        target = -d.LEAD * self._turns  # tightening drives the ram down (body frame -z)
        self._screw.GetTargetPositionAttr().Set(float(target))
        force = float(np.clip(d.SCREW_K * (c["ram"] - target), -d.SCREW_MAX_FORCE, d.SCREW_MAX_FORCE))
        torque = force * d.LEAD / (2 * math.pi)  # ram load resists tightening (clockwise = -z)
        t = self._torch
        az = _quat_to_matrix(p["body"][3:])[:, 2]
        for name, sgn in (("handle", 1.0), ("body", -1.0)):
            tq = t.tensor((sgn * torque * az)[None, None], dtype=t.float32, device=self._base.device)
            self._base.scene[name].permanent_wrench_composer.set_forces_and_torques_index(
                forces=t.zeros_like(tq), torques=tq, is_global=True)
        for k, s in self._stick.items():  # elasto-plastic Coulomb friction anchors
            if k == "collar" and self.collar_free:
                continue
            x = c[k]
            if abs(x - s[3]) > s[2]:
                s[3] = x - math.copysign(s[2], x - s[3])
                s[0].GetTargetPositionAttr().Set(float(s[3]))
        if not self.collar_free and c["collar"] >= d.COLLAR_CLEAR:
            self._joints["collar"].GetJointEnabledAttr().Set(False)
            self.collar_free = True
            print(f"[puller] t={self._t:5.2f} collar clear of the shaft", flush=True)
        from yam_arena import ik, station
        tcps = []
        for name, side in zip(ARMS, ("left", "right")):
            q = self._base.scene[name].data.joint_pos.torch[0][self._ids[name][0]].cpu().numpy().astype(float)
            tcps.append(ik.tcp_pose(q, station.BASES[side])[0])
        opening=[float(np.clip(self._base.scene[name].data.joint_pos.torch[0][self._ids[name][1]].mean().item()/FINGER_OPEN,0,1)) for name in ARMS]
        score=self.scorer.update(p,self._turns,self.collar_free,opening,tcps)
        hand_gap = float(np.linalg.norm(tcps[0] - tcps[1]))
        from yam_arena.runtime import contact_forces
        pads={side:{k:v for k,v in contact_forces(self._base.scene[side+"_contacts"]).items() if k.startswith("tip_")} for side in ("left","right")}
        info = {"pad_net_forces_n":pads,"tcp_positions": [v.tolist() for v in tcps], "hand_gap_mm": 1000 * hand_gap, "turns": self._turns, "ram_mm": 1000 * c["ram"], "screw_force_n": force,
                "collar_rise_mm": 1000 * c["collar"], "jaw_pos_mm": 1000 * c["jaw_pos"],
                "jaw_neg_mm": 1000 * c["jaw_neg"], "collar_free": self.collar_free, "grip_open":opening, "score":score, "score_reason":self.scorer.reason, "success":self.scorer.success}
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 3), **info,
                                          "poses": {k: v.round(5).tolist() for k, v in p.items()}}) + "\n")
        if round(self._t * self.info.control_hz) % round(self.info.control_hz / 2) == 0:
            bz = p["body"][2] - d.TOP
            print(f"[puller] t={self._t:5.2f} body_z={1000 * bz:.1f}mm turns={self._turns:+.2f} "
                  f"ram={1000 * c['ram']:+.1f}mm F={force:+.1f}N jaws={1000 * c['jaw_pos']:+.1f}/{1000 * c['jaw_neg']:+.1f}mm "
                  f"collar={1000 * c['collar']:+.1f}mm hands={1000 * hand_gap:.0f}mm", flush=True)
        return StepResult(observation=self._observation(), terminated=self.scorer.success, termination_reason="success" if self.scorer.success else None, info=info)

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()
