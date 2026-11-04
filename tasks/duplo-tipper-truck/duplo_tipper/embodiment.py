"""Embodiment ``yam_duplo_tipper``: the supplied bimanual YAM station with the tipper-truck parts.

The robot, cameras, 14-D joint action and observation contract are the guide adapter's
(``yam_arena``), unchanged. This module adds the truck bodies, the wheel bearings, the
connection surrogate and the scorer.

Connection surrogate (declared approximation): stud and hinge connections are not solved
by contact mechanics. A retention joint is enabled when a part is at its manual position
within the tolerances in ``design.py``, moving slowly, and pressed onto its parent with
measured contact force for 0.1 s. The joint is created at the measured relative pose, so
nothing is teleported or snapped to an ideal pose. The bed's joint is a revolute hinge with
the part's opening range. Joints have provisional break limits; a broken or drifted joint is
reported as a lost connection.
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
from .scoring import Observed, Scorer

TASK_FILE = Path(__file__).resolve().parents[1] / "task.py"
PARTS = ("chassis", "axle_front", "axle_rear", "brick", "cab", "hinge", "bed")
BODY: dict = {}  # prim of a part's rigid body under the env, where it is not the part's own prim
LINKS = {  # child: (parent, joint kind)
    "brick": ("chassis", "fixed"), "cab": ("chassis", "fixed"),
    "hinge": ("cab", "fixed"), "bed": ("hinge", "revolute"),
}
ACTIVE = None  # read only by the declared privileged demonstration policy


def start_layout(preassembled: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Fixed starting poses. Parts named in ``preassembled`` start joined (diagnostics only)."""
    built = d.assembled_poses()
    loose = {
        # loose parts lie in the truck's orientation (frame +y = world +y), clear of the tray at +y
        "brick": np.r_[0.20, -0.17, d.TOP + d.BRICK_H / 2, d.TRUCK_QUAT],
        "cab": np.r_[0.43, -0.16, d.TOP + 0.0005, d.TRUCK_QUAT],
        "hinge": np.r_[0.15, 0.10, d.TOP + d.PLATE_H + 0.0005, d.TRUCK_QUAT],
        "bed": np.r_[0.20, 0.22, d.TOP + 0.0005, d.TRUCK_QUAT],
    }
    poses = {n: built[n] for n in ("chassis", "axle_front", "axle_rear")}
    for name, pose in loose.items():
        poses[name] = built[name] if name in preassembled else pose
    poses["payload"] = np.array([0.12, 0.17, d.TOP + d.PAYLOAD_SIZE / 2 + 0.0005, 0, 0, 0, 1.0])
    return poses


class TipperEmbodiment(YamArenaEmbodiment):
    def __init__(self, telemetry=None, preassembled="", disable_connections=False, **kwargs):
        global ACTIVE
        debug = kwargs.pop("debug", False)
        kwargs.setdefault("task", str(TASK_FILE))
        kwargs.setdefault("object_jitter", 0)
        super().__init__(**kwargs)
        self.telemetry = Path(telemetry).expanduser() if telemetry else None
        self.preassembled = tuple(p for p in str(preassembled).replace("+", ",").split(",") if p)
        unknown = set(self.preassembled) - set(LINKS)
        if unknown:
            raise ValueError(f"preassembled names parts of {tuple(LINKS)}, got {sorted(unknown)}")
        self.disable_connections = bool(disable_connections)
        self.debug = str(debug).lower() in ("true", "1")
        self.layout = start_layout(self.preassembled)
        self.scorer = Scorer(preassembled=self.preassembled)
        self._trace = None
        self._joints = {}
        ACTIVE = self

    # -- scene ----------------------------------------------------------------------------
    def _ensure_sim(self, goal):
        if self._env is not None:
            return
        if upstream._APP is None:
            from isaaclab.app import AppLauncher
            upstream._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(upstream._close_app_at_exit)
        import torch
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import env_cfg_callback, task_object, task_room
        from yam_arena.task_file import Box

        from .assets import BUILDERS

        robot = BimanualYamEmbodiment(enable_cameras=True, rig=self.rig, wrist_cam_mount=self.wrist_cam_mount,
                                      contact_sensors=False, robot_contact_offset=0.001)
        objects = []
        for name in PARTS:
            pose = self.layout[name]
            kind = ObjectType.RIGID
            obj = Object(name=name, object_type=kind, usd_path=str(BUILDERS[name]()),
                         initial_pose=Pose(position_xyz=tuple(map(float, pose[:3])),
                                           rotation_xyzw=tuple(map(float, pose[3:]))),
                         spawn_cfg_addon={"activate_contact_sensors": True})
            objects.append(obj)
        objects.append(task_object(Box("payload", pos=tuple(map(float, self.layout["payload"][:3])),
                                       size=(d.PAYLOAD_SIZE,) * 3, mass=d.MASS["payload"], color=(0.80, 0.30, 0.12))))
        objects.extend(task_object(b) for b in tray_boxes())
        for obj in objects:
            obj.initial_pose = Pose.from_dict(obj.initial_pose.to_dict())  # float32-safe transforms
            obj.enable_reset_pose()

        def configure(cfg):
            cfg = env_cfg_callback(self.info.control_hz)(cfg)
            for child, (parent, _) in LINKS.items():
                setattr(cfg.scene, f"{child}_contact", ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/" + child, filter_prim_paths_expr=["{ENV_REGEX_NS}/" + BODY.get(parent, parent)],
                    history_length=2))
            return cfg

        environment = IsaacLabArenaEnvironment(
            name="yam_duplo_tipper", scene=Scene([*task_room("station", self.rig.top_cam), *objects]),
            embodiment=robot, env_cfg_callback=configure)
        self._env = ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(
            num_envs=1, solve_relations=False, device=self.device)).make_registered()
        self._base = self._env.unwrapped
        self._torch, self._goal = torch, goal
        self._free_objects = self.task.free_objects
        self._ready = torch.tensor([[*READY, 1.0, *READY, 1.0]], dtype=torch.float32, device=self._base.device)
        self._ids = {n: (self._base.scene[n].find_joints(ARM_JOINTS, preserve_order=True)[0],
                         self._base.scene[n].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for n in ARMS}
        self._make_joints()

    def _make_joints(self):
        from pxr import Gf, Sdf, UsdPhysics

        stage = self._base.sim.stage
        self._bearings, self._anchor = {}, {}
        path = lambda n: Sdf.Path(f"/World/envs/env_0/{BODY.get(n, n)}")  # noqa: E731
        for name, sign in (("axle_front", -1), ("axle_rear", 1)):
            j = UsdPhysics.RevoluteJoint.Define(stage, f"/World/Bearing_{name}")
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, sign * d.AXLE_Y, d.AXLE_Z))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0))
            j.CreateCollisionEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)
            # Bearing friction: a zero-velocity drive whose torque is capped, i.e. Coulomb friction
            # of d.BEARING_TORQUE per axle (uncalibrated). Free maximal-coordinate bearings crept.
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(d.BEARING_STIFFNESS))
            drive.CreateDampingAttr(d.BEARING_DAMPING)
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateMaxForceAttr(d.BEARING_TORQUE)
            self._bearings[name] = drive
            j.CreateBody0Rel().SetTargets([path("chassis")])  # bodies last: frames already consistent
            j.CreateBody1Rel().SetTargets([path(name)])
        for child, (parent, kind) in LINKS.items():
            # Author each retention joint disabled, with frames matching the bodies' current relative
            # pose, and attach its bodies last; otherwise PhysX briefly sees a "disjointed" joint and
            # snaps the bodies together, kicking the start layout.
            offset, r = d.relative(self.layout[parent], self.layout[child])
            q = d.matrix_to_quat(r)
            j = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.FixedJoint).Define(
                stage, f"/World/Link_{child}")
            j.CreateJointEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, offset)))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0))
            if kind == "revolute":
                j.CreateAxisAttr("X")
                j.CreateLowerLimitAttr(-math.degrees(d.BED_TIP_LIMIT))  # opening is rotation about -x
                j.CreateUpperLimitAttr(0.0)
            j.CreateCollisionEnabledAttr(False)
            j.CreateExcludeFromArticulationAttr(True)
            force, torque = d.BREAK[child]
            j.CreateBreakForceAttr(force)
            j.CreateBreakTorqueAttr(torque)
            j.CreateBody0Rel().SetTargets([path(parent)])
            j.CreateBody1Rel().SetTargets([path(child)])
            self._joints[child] = j

    # -- state ----------------------------------------------------------------------------
    def poses(self) -> dict[str, np.ndarray]:
        return {n: self._base.scene[n].data.root_pose_w.torch[0].cpu().numpy().astype(float)
                for n in (*PARTS, "payload")}

    def speeds(self, poses=None) -> dict[str, tuple[float, float]]:
        """Linear and angular speed of each part from its actual pose change over the last control step.

        PhysX's reported body velocities include constraint-correction velocity: the 8 g hinge plate, held
        between its fixed joint and the bed's hinge at its stop, reported 1.0 rad/s while its orientation
        changed by 0.0004 rad/s (bed-025). Rest checks therefore use measured motion.
        """
        poses = self.poses() if poses is None else poses
        prev, dt = getattr(self, "_speed_prev", None), self._base.step_dt
        out, self._vel = {}, {}
        for n in ("chassis", "brick", "cab", "hinge", "bed", "payload"):
            if prev is None:
                out[n], self._vel[n] = (0.0, 0.0), np.zeros(3)
                continue
            self._vel[n] = (poses[n][:3] - prev[n][:3]) / dt
            lin = float(np.linalg.norm(self._vel[n]))
            turn = d.angle_of(d.quat_to_matrix(prev[n][3:]).T @ d.quat_to_matrix(poses[n][3:])) / dt
            out[n] = (lin, float(turn))
        self._speed_prev = {k: v.copy() for k, v in poses.items()}
        return out

    def contact(self, child) -> float:
        m = self._base.scene[f"{child}_contact"].data.force_matrix_w
        m = m.torch if hasattr(m, "torch") else m
        return float(m[0].norm(dim=-1).sum().item())

    def _join(self, child, poses):
        from pxr import Gf

        parent, kind = LINKS[child]
        offset, r = d.relative(poses[parent], poses[child])
        j = self._joints[child]
        j.GetLocalPos0Attr().Set(Gf.Vec3f(*map(float, offset)))
        if kind == "revolute":
            tip = d.hinge_seat(poses[parent], poses[child])["tip"]
            residual = d.rot_x(tip) @ r  # r = rot_x(-tip) @ residual
            q0, q1 = np.array([0, 0, 0, 1.0]), d.matrix_to_quat(residual.T)
        else:
            q0, q1 = d.matrix_to_quat(r), np.array([0, 0, 0, 1.0])
        j.GetLocalRot0Attr().Set(Gf.Quatf(float(q0[3]), Gf.Vec3f(*map(float, q0[:3]))))
        j.GetLocalRot1Attr().Set(Gf.Quatf(float(q1[3]), Gf.Vec3f(*map(float, q1[:3]))))
        j.GetJointEnabledAttr().Set(True)
        self._joined[child] = True
        self._join_pose[child] = (offset, r)

    def _unjoin(self, child):
        self._joints[child].GetJointEnabledAttr().Set(False)
        self._joined[child] = False
        self._join_pose.pop(child, None)

    def reset(self, scene, *, seed=None):
        if self._joints:
            for child in LINKS:
                self._joints[child].GetJointEnabledAttr().Set(False)
        self._joined = {c: False for c in LINKS}
        self._speed_prev = None
        self._join_pose = {}
        self._gates = {c: d.Gate() for c in LINKS}
        self.scorer.reset()
        self.events = []
        if self.telemetry and self._trace is None:
            self.telemetry.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.telemetry.open("a", buffering=1)
        if self._env is None:
            self._ensure_sim(None)
        super().reset(scene, seed=seed)  # every joint disabled: teleports cannot strain one
        if self.debug:
            print("[tipper-debug] after base reset", {n: v.round(4).tolist() for n, v in self.poses().items() if n in ("chassis", "cab", "brick")}, flush=True)
        self._restore_layout()
        # Preassembled diagnostic parts rest on their parents; join them where they now are.
        poses = self.poses()
        for child in LINKS:
            if child in self.preassembled:
                self._join(child, poses)
        if self.debug:
            print("[tipper-debug] after restore", {n: v.round(4).tolist() for n, v in self.poses().items() if n in ("chassis", "cab", "brick")}, flush=True)
        self._anchor = {n: self._axle_angle(self.poses(), n) for n in self._bearings}
        for n, drive in self._bearings.items():
            drive.GetTargetPositionAttr().Set(float(self._anchor[n]))
        observation = self._observation()
        if self._trace:
            self._trace.write(json.dumps({"reset": True, "preassembled": list(self.preassembled)}) + "\n")
        return observation

    def _restore_layout(self, settle_s=0.5):
        """Write every part back to the documented start pose at rest, then settle with the arms at READY.

        The first physics steps after spawning give the free-rolling base a small impulse; without
        this the truck starts rolling and the start layout would not match the documentation.
        """
        torch, base = self._torch, self._base
        for _ in range(2):
            for name in (*PARTS, "payload"):
                obj = base.scene[name]
                pose = torch.tensor(self.layout[name][None], dtype=torch.float32, device=base.device)
                pose[:, :3] += base.scene.env_origins
                obj.write_root_pose_to_sim_index(root_pose=pose)
                obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))
            for _ in range(max(1, round(settle_s / base.step_dt))):
                self._env.step(self._ready)
        for _ in range(4):
            base.sim.render()

    def _task_step(self):
        from yam_arena.runtime import pad_tips_world, tcp_world

        poses = self.poses()
        speeds = self.speeds(poses)
        dt = self._base.step_dt
        events = []
        seats = {}
        for child, (parent, kind) in LINKS.items():
            seat = (d.hinge_seat(poses[parent], poses[child]) if kind == "revolute" else
                    d.fixed_seat(poses[parent], poses[child], {"brick": d.BRICK_IN_CHASSIS, "cab": d.CAB_IN_CHASSIS,
                                                               "hinge": d.HINGE_IN_CAB}[child]))
            seats[child] = seat
            if self._joined[child]:
                offset, r = d.relative(poses[parent], poses[child])
                j_offset, j_r = self._join_pose[child]
                drift = float(np.linalg.norm(offset - j_offset))
                turn = d.angle_of(j_r.T @ r) if kind == "fixed" else seat["axis_angle"]
                if drift > d.LOST_POS or turn > d.LOST_ANGLE or not self._joint_alive(child):
                    self._unjoin(child)
                    events.append(f"{child}_connection_lost")
                continue
            force = self.contact(child)
            rel_speed = float(np.linalg.norm(self._vel[child] - self._vel[parent]))  # measured, like the rest check
            seat["force"] = force
            if self._gates[child].update(seat["seated"], force, rel_speed, dt) and not self.disable_connections:
                self._join(child, poses)
                events.append(f"{child}_joined")
        obs = self._observation()
        hands = []
        for side in ARMS:
            robot = self._base.scene[side]
            hands += [tcp_world(robot), pad_tips_world(robot)]
        grippers_open = all(float(obs.state["joint_pos"][i]) > 0.9 for i in (6, 13))
        observed = Observed(poses=poses, speeds=speeds, joined=dict(self._joined), hands=hands,
                            grippers_open=grippers_open, dt=dt)
        n_before = len(self.scorer.events)
        score, info = self.scorer.update(observed)
        events += self.scorer.events[n_before:]
        self.events += [(round(self._t, 2), e) for e in events]
        info["bearings"] = self._update_bearings(poses)
        info.update({"events": events, "seats": _jsonable(seats), "contact_N": {c: seats[c].get("force") for c in LINKS}})
        if self._trace:
            self._trace.write(json.dumps({"t": round(self._t, 3), **_jsonable(info),
                                          "poses": {k: v.round(5).tolist() for k, v in poses.items()},
                                          "tcp": {s.split("_")[0]: hands[2 * i].round(4).tolist() for i, s in enumerate(ARMS)},
                                          "grip": [float(obs.state["joint_pos"][6]), float(obs.state["joint_pos"][13])]})
                              + "\n")
        if events or round(self._t * self.info.control_hz) % round(self.info.control_hz) == 0:
            c = poses["chassis"]
            print(f"[tipper] t={self._t:5.2f} score={score:.2f} links={''.join(k[0] if v else '-' for k, v in info['links'].items())} "
                  f"tip={info['bed_tip_deg'] if info['bed_tip_deg'] is None else round(info['bed_tip_deg'], 1)} "
                  f"truck=({c[0]:.4f},{c[1]:.4f}) loaded={info['loaded']} unloaded={info['unloaded']} "
                  f"tray={info['payload_in_tray']} events={events}", flush=True)
        return StepResult(observation=obs, terminated=info["success"],
                          termination_reason="success" if info["success"] else None, info=_jsonable(info))

    @staticmethod
    def _axle_angle(poses, name) -> float:
        """Axle rotation about the chassis x axis, degrees in (-180, 180]."""
        r = d.quat_to_matrix(poses["chassis"][3:]).T @ d.quat_to_matrix(poses[name][3:])
        return math.degrees(math.atan2(r[2, 1], r[1, 1]))

    def _update_bearings(self, poses) -> dict:
        """Elasto-plastic Coulomb bearing: slide each anchor so the spring never exceeds the breakaway torque."""
        window = d.BEARING_TORQUE / d.BEARING_STIFFNESS
        out = {}
        for n, drive in self._bearings.items():
            a = self._axle_angle(poses, n)
            err = (a - self._anchor[n] + 180.0) % 360.0 - 180.0
            if abs(err) > window:
                self._anchor[n] = (a - math.copysign(window, err) + 180.0) % 360.0 - 180.0
                drive.GetTargetPositionAttr().Set(float(self._anchor[n]))
            out[n] = {"angle_deg": a, "anchor_deg": self._anchor[n]}
        return out

    def _joint_alive(self, child) -> bool:
        """A broken PhysX joint stays enabled in USD; its bodies drifting apart detects it."""
        return True

    def close(self):
        if self._trace:
            self._trace.close()
            self._trace = None
        super().close()


def tray_boxes():
    """Fixed receiving tray (taped to the table): 4 mm floor, 12 mm walls."""
    from yam_arena.task_file import Box

    cx, cy = d.TRAY_CENTER
    (ix, iy), (t, h), f = d.TRAY_INNER, d.TRAY_WALL, d.TRAY_FLOOR
    grey = (0.55, 0.58, 0.62)
    return [
        Box("tray_floor", pos=(cx, cy, d.TOP + f / 2), size=(ix + 2 * t, iy + 2 * t, f), color=grey),
        Box("tray_wall_near", pos=(cx, cy - iy / 2 - t / 2, d.TOP + f + h / 2), size=(ix + 2 * t, t, h), color=grey),
        Box("tray_wall_far", pos=(cx, cy + iy / 2 + t / 2, d.TOP + f + h / 2), size=(ix + 2 * t, t, h), color=grey),
        Box("tray_wall_left", pos=(cx - ix / 2 - t / 2, cy, d.TOP + f + h / 2), size=(t, iy, h), color=grey),
        Box("tray_wall_right", pos=(cx + ix / 2 + t / 2, cy, d.TOP + f + h / 2), size=(t, iy, h), color=grey),
    ]


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x
