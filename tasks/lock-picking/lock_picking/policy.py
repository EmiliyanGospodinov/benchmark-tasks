"""Policy ``lock_picking_scripted``: privileged demonstration (reads simulator part poses).

Right gripper: the tension wrench's orange sleeve (fingers across it, kept vertical; the sleeve spins on its
pin). It keeps light turning pressure: the commanded wrench angle is the measured plug angle plus the wanted
flex (6 deg = 0.06 N m), with an integral correction on the measured flex (the arm is compliant), never more
than CAP degrees. Left gripper: the pick's upright post. It carries the pick to the keyway, slides the hook
under the binding pin, raises it slowly until the plug gives (the pin sets at the shear line), lowers, and
moves on in binding order (the declared order; a picker finds it by feel). Then it withdraws the pick and puts
it back on the table; the right gripper turns the plug to the open mark, lets the wrench relax and withdraws.
Outputs ordinary 14-D joint targets through the guide's IK; never sets object poses.

``-P mode=pick`` (default) is the demonstration; ``-P mode=idle`` keeps both arms at the ready pose;
``-P mode=notension`` runs the pick hand alone (no turning pressure) as a one-arm comparison.
"""

from __future__ import annotations

import math

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy

from . import design as d

SLEEVE_SQUEEZE = -0.008  # 16 mm target gap on the 18 mm sleeve
POST_SQUEEZE = 0.0  # fully closed target on the 10 mm handle plate
FLEX_WANT = 6.0  # deg (0.06 N m)
TURN_FLEX = 7.0
PICK_CAP = 18.0  # deg: the wrench is never commanded past this while the pick is in the keyway
SHAFT_LOCAL_Z = d.KEYWAY_FLOOR + d.PICK_SHAFT[2] / 2 + 0.0003  # shaft centre just above the keyway floor
LIFT_RATE = 0.0015  # m/s, tip lift
LIFT_MAX = 0.0100
TILT_MAX = math.radians(28.0)


def _rodrigues(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
GIVE_DEG = 1.5  # plug rotation that tells the pin has set


class Arm:
    def __init__(self, base, yaw):
        self.base = np.asarray(base, float)
        self.rot0 = grip_rotation(yaw)
        self.rot = self.rot0
        self.q = np.array(READY, float)
        self.pos = ik.tcp_pose(self.q, self.base)[0]
        self.finger = FINGER_OPEN
        self.worst_mm = 0.0

    def move(self, target, speed, dt):
        delta = np.asarray(target, float) - self.pos
        n = float(np.linalg.norm(delta))
        step = speed * dt
        self.pos = self.pos + (delta if n <= step else delta * step / n)
        return n <= max(step, 1e-5)

    def grip(self, finger, rate, dt):
        self.finger += float(np.clip(finger - self.finger, -rate * dt, rate * dt))
        return abs(finger - self.finger) < 1e-6

    def action(self):
        q, pe, _ = ik.solve(self.pos, ik.matrix_to_quat_xyzw(self.rot), self.q, base_pos=self.base, iters=60)
        self.q = q
        self.worst_mm = max(self.worst_mm, pe * 1000)
        return np.r_[q, float(np.clip(self.finger / FINGER_OPEN, 0.0, 1.0))]


class LockScriptedPolicy(YamScriptedPolicy):
    def __init__(self, mode="pick", **kwargs):
        super().__init__(**kwargs)
        self.mode = str(mode)
        assert self.mode in ("pick", "idle", "notension", "oneshots"), self.mode
        self._episode = -1

    def reset(self, scene):
        self._built = False
        self._printed = None
        self._episode += 1
        # one-arm shortcut tests (-P mode=oneshots): episode k runs route k % 3
        self.route = (["pickonly", "selfweight", "alternate"][self._episode % 3] if self.mode == "oneshots"
                      else self.mode)

    # ----------------------------------------------------------------- helpers
    def _build(self):
        self._built = True
        self.L = Arm(self._bases["left"], -math.pi / 2)
        self.R = Arm(self._bases["right"], 0.0)
        from yam_arena.urdf import axis_angle_matrix
        tx, ty = d.WRENCH_GRIP_TILT_DEG
        self.R.rot = self.R.rot0 = (axis_angle_matrix(np.array([0, 1.0, 0]), math.radians(ty))
                                    @ axis_angle_matrix(np.array([1.0, 0, 0]), math.radians(tx)) @ self.R.rot0)
        self.phase_l, self.phase_r = "start", "start"
        self.t_l = self.t_r = 0.0
        self.flex_bias = 0.0
        self.cap = PICK_CAP
        self.tension = False
        self.phi_cmd = 0.0
        self.tip_bias = np.zeros(3)
        self.pin_k, self.lift, self.theta0, self.tries = 0, 0.0, 0.0, 0
        self.order = d.binding_order()
        self.cur_x = d.FACE_X
        self.tilt_bias = 0.0
        self.roll, self.twisted, self.L_hold = 0.0, False, False
        self.grip_off = None
        self.pick_done = False
        self.done_t = None

    def _sleeve(self, phi_deg, off=(0.0, 0.0, 0.0)):
        return d.sleeve_center(math.radians(phi_deg)) + np.asarray(off)

    def _tip_world(self, s, x, local_z):
        th = math.radians(s["theta_deg"])
        p = d.AXIS + d.rot_x(th) @ np.array([0.0, 0.0, local_z])
        p[0] = x
        return p

    def _left_to_tip(self, s, tip_des, speed, dt, integrate=True):
        """Move the left TCP so that the pick tip goes to tip_des (grip offset + integral correction)."""
        if integrate:
            err = tip_des - np.asarray(s["pick_tip"])
            self.tip_bias += np.clip(1.5 * err, -0.006, 0.006) * dt
            self.tip_bias = np.clip(self.tip_bias, [-0.02, -0.008, -0.02], [0.02, 0.008, 0.02])
        return self.L.move(tip_des + self.grip_off + self.tip_bias, speed, dt)

    def _lever(self, s, x_pin, lift, dt):
        """Lift the hook by tilting the pick about the keyway floor at the face (the fulcrum)."""
        th = math.radians(s["theta_deg"])
        rx = d.rot_x(th)
        arm = max(x_pin - d.FACE_X, 0.01)
        tip_local = rx.T @ (np.asarray(s["pick_tip"]) - d.AXIS)
        if lift > 0:
            err = SHAFT_LOCAL_Z + lift - tip_local[2]
            self.tilt_bias = float(np.clip(self.tilt_bias + np.clip(1.5 * err / arm, -0.05, 0.05) * dt, -0.15, 0.15))
        alpha = float(np.clip(math.asin(min(lift / arm, 0.9)) + self.tilt_bias, 0.0, TILT_MAX))
        fulcrum = d.AXIS + rx @ np.array([0.0, 0.0, SHAFT_LOCAL_Z])
        fulcrum[0] = d.FACE_X
        self.tip_bias[0] = float(np.clip(self.tip_bias[0] + np.clip(1.5 * (x_pin - s["pick_tip"][0]), -0.006, 0.006) * dt,
                                         -0.02, 0.02))  # keep the hook under the pin while tilting
        tcp_flat = self._tip_world(s, x_pin, SHAFT_LOCAL_Z) + self.grip_off + self.tip_bias
        rot = _rodrigues(rx @ np.array([0.0, 1.0, 0.0]), -alpha)
        self.L.rot = rot @ self.L.rot0
        self.L.move(fulcrum + rot @ (tcp_flat - fulcrum), 0.05, dt)
        return alpha

    # ----------------------------------------------------------------- right arm (tension)
    def _right(self, s, dt):
        R, ph = self.R, self.phase_r
        self.t_r += dt
        start_phi = s["theta_deg"] + s["flex_deg"]
        if ph == "start":
            self.phi_cmd = start_phi
            if R.move(self._sleeve(self.phi_cmd, (-0.07, 0.0, 0.02)), 0.12, dt):
                self.phase_r, self.t_r = "slide on", 0.0
        elif ph == "slide on":
            if R.move(self._sleeve(self.phi_cmd, (0.0, 0.0, 0.0)), 0.05, dt) and self.t_r > 0.3:
                self.phase_r, self.t_r = "grip", 0.0
        elif ph == "grip":
            if R.grip(SLEEVE_SQUEEZE, 0.05, dt) and self.t_r > 1.2:
                self.phase_r, self.t_r = "tension", 0.0
                self.tension = True
        elif ph == "tension" and self.route == "alternate" and self.phase_l in ("raise", "set"):
            R.grip(FINGER_OPEN, 0.1, dt)  # let go of the wrench while lifting
            R.move(self._sleeve(self.phi_cmd, (-0.04, 0.0, 0.0)), 0.1, dt)
            self.phi_cmd = s["theta_deg"] + s["flex_deg"]
            self.regrip = True
        elif ph == "tension" and self.route == "alternate" and getattr(self, "regrip", False):
            if R.move(self._sleeve(self.phi_cmd), 0.05, dt) and R.grip(SLEEVE_SQUEEZE, 0.05, dt):
                self.regrip = False
        elif ph in ("tension", "turn"):
            want = TURN_FLEX if ph == "turn" else FLEX_WANT
            if ph == "turn":
                self.cap = min(self.cap + 15.0 * dt, d.THETA_OPEN_DEG + TURN_FLEX + 2.0)
                if s["theta_deg"] >= d.THETA_OPEN_DEG - 0.3 and self.t_r > 1.0:
                    self.phase_r, self.t_r = "relax", 0.0
            err = want - s["flex_deg"]
            self.flex_bias = float(np.clip(self.flex_bias + np.clip(1.5 * err, -8, 8) * dt, -20, 20))
            target = min(s["theta_deg"] + want + self.flex_bias, self.cap, s["theta_deg"] + 12.0)  # never command past 12 deg of flex (cap 15)
            self.phi_cmd += float(np.clip(target - self.phi_cmd, -20 * dt, 20 * dt))
            R.move(self._sleeve(self.phi_cmd), 0.2, dt)
            if ph == "tension" and self.pick_done:
                n_set = sum(b >= d.SET_DRIVER_MIN - 0.0003 for b in s["driver_bottom"])
                self.phase_r, self.t_r = ("turn", 0.0) if n_set == 3 else ("relax", 0.0)
        elif ph == "relax":
            self.phi_cmd += float(np.clip(s["theta_deg"] - 1.0 - self.phi_cmd, -6 * dt, 6 * dt))
            R.move(self._sleeve(self.phi_cmd), 0.1, dt)
            if self.t_r > 2.0:
                self.phase_r, self.t_r = "release", 0.0
        elif ph == "release":
            if R.grip(FINGER_OPEN, 0.06, dt) and self.t_r > 1.0:
                self.phase_r, self.t_r = "withdraw", 0.0
        elif ph == "withdraw":
            if R.move(self._sleeve(self.phi_cmd, (-0.09, 0.0, 0.0)), 0.06, dt):
                self.phase_r, self.t_r = "away", 0.0
        elif ph == "away":
            R.move(self._sleeve(self.phi_cmd, (-0.09, -0.04, 0.10)), 0.08, dt)

    # ----------------------------------------------------------------- left arm (pick)
    def _left(self, s, dt):
        L, ph = self.L, self.phase_l
        self.t_l += dt
        post = d.PICK_START + np.asarray(d.POST_GRIP_LOCAL)
        if ph == "start":
            if L.move(post + (0, 0, 0.07), 0.12, dt):
                self.phase_l, self.t_l = "descend", 0.0
        elif ph == "descend":
            if L.move(post, 0.05, dt) and self.t_l > 0.3:
                self.phase_l, self.t_l = "grip", 0.0
        elif ph == "grip":
            if L.grip(POST_SQUEEZE, 0.05, dt) and self.t_l > 1.0:
                self.grip_off = L.pos - np.asarray(s["pick_tip"])
                self.phase_l, self.t_l = "lift", 0.0
        elif ph == "lift":
            if L.move(d.PICK_START + self.grip_off + (0, 0, 0.06), 0.06, dt):
                self.phase_l, self.t_l = "wait tension", 0.0
        elif ph == "wait tension":
            pre = np.array([d.FACE_X - 0.03, 0.0, d.AXIS_Z + SHAFT_LOCAL_Z])
            arrived = L.move(pre + self.grip_off + (0, 0, 0.015), 0.08, dt)
            if arrived and ((self.tension and self.phase_r != "start") or self.route in ("notension", "pickonly", "selfweight")):
                self.phase_l, self.t_l = "align", 0.0
        elif ph == "align":
            tip = self._tip_world(s, d.FACE_X - 0.03, SHAFT_LOCAL_Z)
            self._left_to_tip(s, tip, 0.03, dt)
            err = float(np.linalg.norm(tip - np.asarray(s["pick_tip"])))
            if self.t_l > 1.5 and (err < 0.0015 or self.t_l > 6.0):
                self.phase_l, self.t_l = "to pin", 0.0
        elif ph == "to pin":
            low = (d.rot_x(math.radians(s["theta_deg"])).T @ (np.asarray(s["pick_tip"]) - d.AXIS))[2] < d.KEYWAY_FLOOR - 0.001
            if low and s["pick_tip"][0] < d.FACE_X + 0.012:
                self.phase_l, self.t_l = "align", 0.0  # missed the keyway mouth: back out and realign
                return
            i = self.order[self.pin_k]
            tip = self._tip_world(s, d.pin_world_x(i) + 0.0025, SHAFT_LOCAL_Z)
            arrived = self._left_to_tip(s, tip, 0.02, dt)
            if arrived and self.t_l > 1.0 and (abs(s["pick_tip"][0] - tip[0]) < 0.0025 or self.t_l > 5.0):
                self.phase_l, self.t_l = "raise", 0.0
                self.cur_x = tip[0]
                self.lift, self.theta0 = 0.0, s["theta_deg"]
        elif ph == "raise":
            self.lift = min(self.lift + LIFT_RATE * dt, LIFT_MAX)
            self._lever(s, self.cur_x, self.lift, dt)
            if s["theta_deg"] > self.theta0 + GIVE_DEG:
                self.phase_l, self.t_l = "set", 0.0
            elif self.lift >= LIFT_MAX and self.t_l > 10.0:
                self.phase_l, self.t_l = "lower", 0.0
                self.tries += 1
        elif ph == "set":
            self._lever(s, self.cur_x, self.lift, dt)
            if self.t_l > 0.5:
                self.phase_l, self.t_l = "lower", 0.0
                self.pin_k += 1
                self.tries = 0
        elif ph == "lower":
            self.lift = max(self.lift - 0.004 * dt, 0.0)
            self.tilt_bias *= max(0.0, 1.0 - 2.0 * dt)
            self._lever(s, self.cur_x, self.lift, dt)
            if self.lift <= 0.0 and self.t_l > 1.5:
                self.L.rot, self.tilt_bias = self.L.rot0, 0.0
                if self.pin_k >= 3 or self.tries >= 3:
                    self.phase_l, self.t_l = "withdraw", 0.0
                else:
                    self.phase_l, self.t_l = "to pin", 0.0
        elif ph == "withdraw":
            if self.route == "pickonly" and not self.twisted:
                return  # hold the pick in the keyway while twisting
            tip = self._tip_world(s, d.FACE_X - 0.04, SHAFT_LOCAL_Z)
            if self._left_to_tip(s, tip, 0.02, dt) and s["pick_tip"][0] < d.FACE_X - 0.03:
                self.phase_l, self.t_l = "carry back", 0.0
        elif ph == "carry back":
            if L.move(d.PICK_START + self.grip_off + self.tip_bias + (0, 0, 0.05), 0.06, dt):
                self.phase_l, self.t_l = "put down", 0.0
                self.pick_done = True
        elif ph == "put down":
            if L.move(d.PICK_START + self.grip_off + self.tip_bias + (0, 0, 0.004), 0.03, dt) and self.t_l > 1.0:
                self.phase_l, self.t_l = "let go", 0.0
        elif ph == "let go":
            if L.grip(FINGER_OPEN, 0.06, dt) and self.t_l > 0.8:
                self.phase_l, self.t_l = "away", 0.0
        elif ph == "away":
            L.move(post + (0, 0, 0.10), 0.06, dt)

    def _rolled(self, deg):
        """Left action with the commanded pick pose turned about the plug axis by deg (opening sense)."""
        from yam_arena.urdf import axis_angle_matrix
        r = axis_angle_matrix(np.array([1.0, 0, 0]), math.radians(deg))
        pos, rot = self.L.pos.copy(), self.L.rot.copy()
        p = d.AXIS + r @ (pos - d.AXIS)
        p[0] = pos[0]
        self.L.pos, self.L.rot = p, r @ rot
        a = self.L.action()
        self.L.pos, self.L.rot = pos, rot
        return a

    # ----------------------------------------------------------------- act
    def act(self, observation):
        from . import embodiment

        sim = embodiment.ACTIVE
        if not self._built:
            self._build()
        dt = self._dt
        s = sim.state()
        if self.route != "idle":
            self._left(s, dt)
            if self.route in ("pick", "alternate"):
                self._right(s, dt)
        if self.route == "pickonly":  # lever the plug with the pick: twist it about the plug axis while lifting
            if self.phase_l in ("raise", "set") or self.phase_l == "to pin" and self.pin_k > 0:
                self.roll = min(self.roll + 10.0 * dt, 15.0)
            if self.phase_l == "withdraw" and self.roll < 89.0 and not self.twisted:
                self.roll = min(self.roll + 10.0 * dt, 90.0)  # last try: turn the plug with the pick alone
                self.L_hold = True
            if self.roll >= 89.0:
                self.twisted = True
            out_l = self._rolled(self.roll)
        else:
            out_l = self.L.action()
        out = [out_l, self.R.action()]
        labels = (self.phase_l, self.phase_r)
        if labels != self._printed:
            print(f"[lock-demo] t={observation.state_time:6.2f} left={labels[0]} right={labels[1]} "
                  f"theta={s['theta_deg']:+.2f} flex={s['flex_deg']:+.2f} pin_k={self.pin_k} "
                  f"bias={np.round(self.tip_bias * 1000, 1).tolist()}mm ik_worst_mm={max(self.L.worst_mm, self.R.worst_mm):.1f}",
                  flush=True)
            self._printed = labels
        finished = self.phase_l == "away" and self.phase_r == "away" and self.t_r > 5.0  # leave time for the 2 s check
        meta = {"request_stop": True} if finished or observation.state_time > 170 else {}
        return ActionChunk(actions=[Action(data=pack(*out), meta=meta)], control_hz=1.0 / dt)
