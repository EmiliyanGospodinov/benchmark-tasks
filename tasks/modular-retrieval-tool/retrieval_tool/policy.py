"""Policy ``retrieval_tool_scripted``: privileged demonstration (reads simulator part poses).

full (default): the right gripper squeezes the release against the shaft handle; the left picks the hook head from its stand,
lowers it onto the pin with the slots on the lugs and turns it a quarter turn with the wrist (locking it);
the right releases the control, regrips the fixed rear handle, then lifts the tool, slides the hook into the hood over the cartridge, lowers the tooth into the
pocket and pulls the cartridge out; the left puts the cartridge in the tray; the right lays the tool back on
its mat; both release. Ordinary 14-D joint targets through the guide's IK; never sets object poses.

Diagnostic modes (``-P mode=...``): idle (no motion), assembly (stop after locking), onearm (the left
assembles alone, shaft not held), function (with ``-E layout=assembled``: retrieval only), bare (retrieval
moves with the shaft alone, head left on its stand), direct (the left gripper reaches into the mouth),
headonly (the head alone, held by its neck, pushed into the mouth).
"""
import math

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy

from . import design as d
from .motion import Script, Seg

HANDLE_SQUEEZE, NECK_SQUEEZE, CART_SQUEEZE = -0.006, -0.004, -0.008  # finger targets inside the parts
HANDLE_TCP_DOWN = 0.004  # TCP below the handle top (fingertips reach ~24 mm below the TCP)
BELL_TCP_Z = d.PLATE_T + d.BELL_CAVITY + d.BELL_ROOF + d.NECK[2] - 0.006  # TCP 6 mm under the neck top
INSERT_Z, ENGAGE_Z = 0.034, 0.008  # rod bottom above the table while inserting / with the tooth in the pocket
PULL = 0.260  # pull distance: the cartridge ends with its grip body ~8 cm clear of the hood (gripper housing)
BACK = -0.14  # tool held clear after the pull: hook ~4 cm in front of the recovered cartridge


GRIP_X = -(d.ROD_L + 0.012)  # handle grip point in the shaft frame: the front of the handle, near the
# assembled tool's balance point (the shaft's mass is in the handle), so the pinch carries little moment
_TOOL_Y_CORR = [0.0]
_TOOL_LEVEL = np.zeros(3)  # bounded integral orientation correction from measured tool pose
_TOOL_GRIP = [None]  # measured TCP position in the shaft frame and shaft yaw at pickup (privileged)


_TIP_MEASURED = [False]
_TIP_CORR = [np.zeros(2)]  # measured tooth-to-pocket error cancelled from "settle over pocket" on (privileged)


def _tooth_error(sim):
    poses = sim.poses()
    h, c = poses["head"], poses["cartridge"]
    tip = h[:3] + d.quat_to_matrix(h[3:]) @ np.array([(d.TOOTH_X0 + d.ARM_X1) / 2, 0.0, -d.TOOTH_DROP])
    pocket = c[:3] + d.quat_to_matrix(c[3:]) @ np.array([(d.POCKET_X0 + d.POCKET_X1) / 2, 0.0, 0.0])
    return (tip - pocket)[:2]


def _handle_tcp(dx=0.0, z_rod=None, up=0.0, measure=False, steer=False, aim=None, grip_x=GRIP_X, grip_y=-0.0065):
    """Right TCP on the handle for the tool's pin at PIN_XY + dx along x, rod bottom at z_rod above the table.
    steer: use the grip measured at lift-off so the shaft itself (not the TCP) follows the path along +x."""
    zb = (d.TOP + d.MAT_T) if z_rod is None else d.TOP + z_rod
    def f(sim):
        pos = np.array([d.PIN_XY[0] + grip_x + dx, d.PIN_XY[1] + grip_y, zb + d.HANDLE_H - HANDLE_TCP_DOWN + up])
        g = 0.0
        if measure and sim is not None:
            poses = sim.poses()
            s, tcp = poses["shaft"], sim._gripper_state(poses)[1][1]
            rs = d.quat_to_matrix(s[3:])
            _TOOL_GRIP[0] = (rs.T @ (tcp - s[:3]), d.yaw_of(rs))
            print(f"[tool-demo] tool in grip: tcp {np.round(1000*_TOOL_GRIP[0][0],1)} mm in the shaft frame, "
                  f"shaft yaw {np.degrees(_TOOL_GRIP[0][1]):+.1f} deg", flush=True)
        if steer and _TOOL_GRIP[0] is not None:
            rel, yaw0 = _TOOL_GRIP[0]
            pin = np.array([d.PIN_XY[0] + dx, d.PIN_XY[1], zb])
            pos = pin + rel  # shaft aligned with +x: shaft frame = world axes
            pos[2] = zb + rel[2]
            g = -yaw0  # turn the grip so the shaft points along +x
        if aim == "measure" and sim is not None and not _TIP_MEASURED[0]:
            _TIP_MEASURED[0] = True
            _TIP_CORR[0] = _TIP_CORR[0] - _tooth_error(sim)
            print(f"[tool-demo] tooth correction {np.round(1000 * _TIP_CORR[0], 1)} mm", flush=True)
        if aim in ("measure", "use"):
            pos[:2] += _TIP_CORR[0]
        rotation = grip_rotation(g)
        if steer and sim is not None:
            measured_shaft = sim.poses()["shaft"]
            desired_y = d.PIN_XY[1] + (_TIP_CORR[0][1] if aim in ("measure", "use") else 0.0)
            _TOOL_Y_CORR[0] = float(np.clip(_TOOL_Y_CORR[0] + 0.10*(desired_y-measured_shaft[1]), -0.012, 0.012))
            pos[1] += _TOOL_Y_CORR[0]
            rs = d.quat_to_matrix(measured_shaft[3:])
            error = np.array([math.atan2(rs[2,1],rs[2,2]),
                              math.asin(np.clip(-rs[2,0],-1,1)), d.yaw_of(rs)])
            _TOOL_LEVEL[:] = np.clip(_TOOL_LEVEL - 0.04*error,
                                     -math.radians(8), math.radians(8))
            x,y,z = _TOOL_LEVEL
            cx,sx,cy,sy,cz,sz = math.cos(x),math.sin(x),math.cos(y),math.sin(y),math.cos(z),math.sin(z)
            rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
            rotation = rz @ ry @ rx @ rotation
        return pos, rotation
    return f


def _head_tcp(xy, z_plate, yaw, up=0.0, align=None):
    """Left TCP on the bell. align='measure' records where the head actually sits in the grip (privileged:
    measured head pose and TCP, at segment entry, with the grip at its nominal yaw); align='use' applies that
    in-grip offset so the head's own axis and slots land on the pin and lugs; 'yaw' applies only the yaw."""
    def f(sim):
        pos = np.array([xy[0], xy[1], z_plate + BELL_TCP_Z + up])
        g = yaw
        if xy is PIN and sim is not None:  # aim at the shaft's measured pin axis and lug orientation
            s = sim.poses()["shaft"]
            pos[:2] = s[:2]
            pos[2] = s[2] + d.ROD_T + BELL_TCP_Z + up
            g = yaw + d.yaw_of(d.quat_to_matrix(s[3:]))
        if align == "measure" and sim is not None:
            poses = sim.poses()
            head, tcp = poses["head"], sim._gripper_state(poses)[1][0]
            hy = d.yaw_of(d.quat_to_matrix(head[3:]))
            _IN_GRIP[:] = [*(tcp - head[:3])[:2], (hy - d.HEAD_START_YAW + math.pi) % (2 * math.pi) - math.pi]
            print(f"[tool-demo] head in grip: offset {1000*_IN_GRIP[0]:+.1f},{1000*_IN_GRIP[1]:+.1f} mm, "
                  f"yaw {math.degrees(_IN_GRIP[2]):+.1f} deg", flush=True)
        if align in ("measure", "use", "servo"):
            pos[:2] += _IN_GRIP[:2]
        if align == "servo" and sim is not None:  # cancel the measured head-to-pin error while inserting
            poses = sim.poses()
            err = poses["shaft"][:2] - poses["head"][:2]
            _SERVO[:] = np.clip(np.asarray(_SERVO) + 0.25 * err, -0.006, 0.006)
            pos[:2] += _SERVO
        elif align == "measure":
            _SERVO[:] = [0.0, 0.0]
        if align in ("measure", "use", "servo", "yaw"):
            g = g - _IN_GRIP[2]
        return pos, grip_rotation(g)
    return f


_IN_GRIP = [0.0, 0.0, 0.0]  # measured head offset in the grip: dx, dy (m), dyaw (rad)
PIN = d.PIN_XY  # sentinel: target the measured pin
_SERVO = [0.0, 0.0]  # accumulated insertion correction (m)


def _cart_tcp(up=0.0, at=None):
    def f(sim):
        if at is not None:  # place the cartridge's centre (not the grip body) over this point
            x, y, z = at[0] + (d.POCKET_X1 + d.CART_L) / 2 - d.CART_L / 2, at[1], d.TOP + 0.003
        else:
            c = sim.poses()["cartridge"]
            r = d.quat_to_matrix(c[3:])
            g = c[:3] + r @ np.array([(d.POCKET_X1 + d.CART_L) / 2, 0.0, 0.0])  # centre of the grip body
            x, y, z = g[0], g[1], c[2]
        return np.array([x, y, z + d.CART_BODY_H - 0.008 + up]), grip_rotation(0.0)
    return f


def _direct_tcp(depth):
    return lambda sim: (np.array([d.HOOD_MOUTH_X + depth, 0.0, d.TOP + 0.03]), grip_rotation(math.pi / 2))


def assembly_left():
    seat = d.TOP + d.MAT_T + d.ROD_T
    stand = d.TOP + d.HEAD_STAND[2]
    y0 = d.HEAD_START_YAW
    # Carry the downward tooth clear of the pin/lugs before descending vertically.
    # The detent revision tilts the carried head enough to catch a lug at 30 mm.
    transit_up = 0.045
    return [Seg(2.0, _head_tcp(d.HEAD_XY, stand, y0, 0.06), FINGER_OPEN, "head approach"),
            Seg(1.2, _head_tcp(d.HEAD_XY, stand, y0), FINGER_OPEN, "head descend"),
            Seg(0.7, _head_tcp(d.HEAD_XY, stand, y0), NECK_SQUEEZE, "head grip"),
            Seg(1.0, _head_tcp(d.HEAD_XY, stand, y0, transit_up), NECK_SQUEEZE, "head lift"),
            Seg(2.0, _head_tcp(PIN, seat, y0, transit_up), NECK_SQUEEZE, "head over pin"),
            Seg(0.6, _head_tcp(PIN, seat, y0, transit_up, align="measure"), NECK_SQUEEZE, "head align"),
            Seg(2.0, _head_tcp(PIN, seat, y0, -0.001, align="servo"), NECK_SQUEEZE, "head insert", freeze=False),
            Seg(2.0, _head_tcp(PIN, seat, 0.0, -0.001, align="yaw"), NECK_SQUEEZE, "head quarter turn"),
            Seg(0.6, None, FINGER_OPEN, "head release"),
            Seg(1.0, _head_tcp(PIN, seat, 0.0, 0.07), FINGER_OPEN, "head retreat"),
            Seg(1.5, _park_left, FINGER_OPEN, "left park")]  # clear of the right wrist's path along y = 0


def _park_left(sim):
    return np.array([0.18, 0.24, d.TOP + 0.14]), grip_rotation(0.0)


def holder_right(hold_s):
    return [Seg(2.0, _handle_tcp(up=0.06), FINGER_OPEN, "handle approach"),
            Seg(1.2, _handle_tcp(), FINGER_OPEN, "handle descend"),
            Seg(0.7, _handle_tcp(), HANDLE_SQUEEZE, "handle grip"),
            Seg(max(0.1, hold_s - 3.9), None, HANDLE_SQUEEZE, "hold shaft")]


def _rear_tcp(up=0.0):
    def f(sim):
        shaft = sim.poses()["shaft"]
        r = d.quat_to_matrix(shaft[3:])
        return shaft[:3] + r @ np.array([-0.220, 0.0, d.HANDLE_H-HANDLE_TCP_DOWN+up]), grip_rotation(0.0)
    return f


def regrip_right():
    # Return the release before grasping the fixed rear handle for tool use.
    return [Seg(0.8, None, FINGER_OPEN, "return release"),
            Seg(1.0, _handle_tcp(up=0.06), FINGER_OPEN, "clear release"),
            Seg(1.0, _rear_tcp(up=0.06), FINGER_OPEN, "over rear handle"),
            Seg(1.0, _rear_tcp(), FINGER_OPEN, "rear handle descend"),
            Seg(0.7, _rear_tcp(), HANDLE_SQUEEZE, "rear handle grip")]


def retrieval_right(aim=True):
    reach = d.cart_start_x() + (d.POCKET_X0 + d.POCKET_X1) / 2 - (d.PIN_XY[0] + (d.TOOTH_X0 + d.ARM_X1) / 2)
    return [Seg(0.5, _handle_tcp(0.0, None, measure=True, grip_x=-0.220, grip_y=0.0), HANDLE_SQUEEZE, "tool check grip"),
            Seg(1.2, _handle_tcp(0.0, INSERT_Z, steer=True), HANDLE_SQUEEZE, "tool lift", freeze=False),
            Seg(3.5, _handle_tcp(reach, INSERT_Z, steer=True), HANDLE_SQUEEZE, "tool insert", freeze=False),
            Seg(1.0, _handle_tcp(reach, INSERT_Z, steer=True), HANDLE_SQUEEZE, "settle over pocket", freeze=False),
            Seg(1.2, _handle_tcp(reach, INSERT_Z, steer=True, aim="measure" if aim else None), HANDLE_SQUEEZE,
                "align tooth", freeze=False),
            Seg(1.2, _handle_tcp(reach, ENGAGE_Z, steer=True, aim="use" if aim else None), HANDLE_SQUEEZE,
                "tooth into pocket", freeze=False),
            Seg(1.0, _handle_tcp(reach, ENGAGE_Z, steer=True, aim="use" if aim else None), HANDLE_SQUEEZE,
                "settle tooth", freeze=False),
            Seg(3.5, _handle_tcp(reach - PULL, ENGAGE_Z, steer=True), HANDLE_SQUEEZE, "pull cartridge out", freeze=False),
            Seg(0.8, _handle_tcp(reach - PULL, INSERT_Z + 0.012, steer=True), HANDLE_SQUEEZE, "tooth out", freeze=False),
            Seg(1.5, _handle_tcp(BACK, INSERT_Z + 0.012, steer=True), HANDLE_SQUEEZE, "tool back", freeze=False)]


def stow_right():
    return [Seg(1.5, _handle_tcp(0.0, 0.006 + d.MAT_T, steer=True), HANDLE_SQUEEZE, "tool over rest", freeze=False),
            Seg(1.0, _handle_tcp(0.0, d.MAT_T, steer=True), HANDLE_SQUEEZE, "tool down", freeze=False),
            Seg(0.6, None, FINGER_OPEN, "tool release"),
            Seg(1.0, _handle_tcp(up=0.08), FINGER_OPEN, "tool retreat")]


def cart_left():
    tray = (d.TRAY_CENTER[0], d.TRAY_CENTER[1])
    return [Seg(1.8, _cart_tcp(0.10), FINGER_OPEN, "cartridge approach", freeze=True),
            Seg(1.0, _cart_tcp(), FINGER_OPEN, "cartridge descend"),
            Seg(0.6, _cart_tcp(), CART_SQUEEZE, "cartridge grip"),
            Seg(1.0, _cart_tcp(0.05), CART_SQUEEZE, "cartridge lift"),
            Seg(2.0, _cart_tcp(0.05, tray), CART_SQUEEZE, "to tray"),
            Seg(1.0, _cart_tcp(0.004, tray), CART_SQUEEZE, "lower into tray"),
            Seg(0.6, None, FINGER_OPEN, "cartridge release"),
            Seg(1.0, _cart_tcp(0.08, tray), FINGER_OPEN, "cartridge retreat")]


def _wait(t, label="wait"):
    return [Seg(max(0.1, t), None, FINGER_OPEN, label)]


def _dur(segs):
    return sum(s.duration for s in segs)


def plans(mode):
    if mode == "idle":
        return {}
    if mode == "direct":
        return {"left": [Seg(2.0, _direct_tcp(-0.10), FINGER_OPEN, "to mouth"),
                         Seg(2.5, _direct_tcp(0.12), FINGER_OPEN, "reach in"),
                         Seg(1.0, None, FINGER_OPEN, "hold"),
                         Seg(2.0, _direct_tcp(-0.10), FINGER_OPEN, "back out")]}
    if mode == "release":
        return {"right": holder_right(7.0) + [
            Seg(2.0, _handle_tcp(up=0.025), HANDLE_SQUEEZE, "support lift"),
            Seg(2.0, None, HANDLE_SQUEEZE, "support hold"),
            Seg(2.0, _handle_tcp(), HANDLE_SQUEEZE, "lower"),
            Seg(1.0, None, FINGER_OPEN, "release control"),
            Seg(1.0, _handle_tcp(up=0.08), FINGER_OPEN, "retreat")]}
    left_asm = assembly_left()
    if mode == "early":
        # Release control and withdraw before the head starts approaching.
        return {"right": holder_right(7.0) + [
            Seg(1.0, None, FINGER_OPEN, "release early"),
            Seg(1.0, _handle_tcp(up=0.08), FINGER_OPEN, "retreat")],
                "left": _wait(10.0) + left_asm}
    if mode == "onearm":
        return {"left": left_asm}
    if mode in ("overturn", "overturn_held"):  # window probe: keep turning 35 deg past the stop (torque into the shaft)
        seat = d.TOP + d.MAT_T + d.ROD_T
        asm = left_asm[:8]  # up to and including the quarter turn, head still gripped
        asm += [Seg(2.0, _head_tcp(PIN, seat, math.radians(35), -0.001, align="yaw"), NECK_SQUEEZE, "overturn"),
                Seg(1.0, None, NECK_SQUEEZE, "hold"),
                Seg(0.6, None, FINGER_OPEN, "head release"),
                Seg(1.0, _head_tcp(PIN, seat, 0.0, 0.07), FINGER_OPEN, "head retreat")]
        plans_ = {"left": asm}
        if mode == "overturn_held":
            plans_["right"] = holder_right(_dur(asm)) + [Seg(0.6, None, FINGER_OPEN, "release")]
        return plans_
    if mode in ("function", "bare"):
        right = holder_right(3.9) + regrip_right() + retrieval_right(aim=(mode == "function")) + stow_right()
        return {"right": right}
    if mode == "headonly":  # the hook head alone, held by its neck, pushed as far into the mouth as possible
        stand = d.TOP + d.HEAD_STAND[2]
        y0 = d.HEAD_START_YAW
        z_in = d.TOP + 0.030  # plate bottom above the table: tooth clears the cartridge lip
        deep = (d.HOOD_MOUTH_X + 0.10, 0.0)
        out = (d.HOOD_MOUTH_X - 0.12, 0.0)
        return {"left": [Seg(2.0, _head_tcp(d.HEAD_XY, stand, y0, 0.06), FINGER_OPEN, "head approach"),
                         Seg(1.2, _head_tcp(d.HEAD_XY, stand, y0), FINGER_OPEN, "head descend"),
                         Seg(0.7, _head_tcp(d.HEAD_XY, stand, y0), NECK_SQUEEZE, "head grip"),
                         Seg(1.0, _head_tcp(d.HEAD_XY, stand, y0, 0.05), NECK_SQUEEZE, "head lift"),
                         Seg(2.0, _head_tcp(out, z_in, 0.0), NECK_SQUEEZE, "head to mouth"),
                         Seg(3.0, _head_tcp(deep, z_in, 0.0), NECK_SQUEEZE, "push hook in"),
                         Seg(1.0, _head_tcp(deep, z_in - 0.012, 0.0), NECK_SQUEEZE, "lower hook"),
                         Seg(3.0, _head_tcp(out, z_in - 0.012, 0.0), NECK_SQUEEZE, "pull back")]}
    t_asm = _dur(left_asm)
    right = holder_right(t_asm)
    if mode == "assembly":
        return {"left": left_asm, "right": right + [Seg(0.6, None, FINGER_OPEN, "release"),
                                                    Seg(1.0, _handle_tcp(up=0.08), FINGER_OPEN, "retreat")]}
    right += regrip_right() + retrieval_right()
    left = left_asm + _wait(_dur(right) - t_asm)
    left += cart_left()
    right += _wait(_dur(left) - _dur(right), "hold tool") + stow_right()
    # the hold-tool wait keeps the handle grip (target None keeps the pose; finger stays closed)
    for s in right:
        if s.label == "hold tool":
            s.finger = HANDLE_SQUEEZE
    return {"left": left, "right": right}


class RetrievalScriptedPolicy(YamScriptedPolicy):
    def __init__(self, mode="full", **kwargs):
        super().__init__(**kwargs)
        self.mode = str(mode)
        assert self.mode in ("full", "idle", "release", "early", "assembly", "onearm", "function", "bare", "direct", "headonly", "overturn",
                             "overturn_held"), self.mode

    def reset(self, scene):
        _TOOL_Y_CORR[0] = 0.0
        _TIP_MEASURED[0] = False
        _TOOL_LEVEL[:] = 0.0
        _TIP_CORR[0] = np.zeros(2)
        _TOOL_GRIP[0] = None
        p = plans(self.mode)
        self._scripts = {arm: Script(segs, self._bases[arm]) for arm, segs in p.items()}
        self._end = (max(s.duration for s in self._scripts.values()) if p else 8.0) + 3.0
        self._printed = None

    def act(self, observation):
        from . import embodiment
        sim = embodiment.ACTIVE
        t = float(observation.state_time) + self._dt
        actions, labels = [], []
        for arm in ("left", "right"):
            if arm in self._scripts:
                a, label = self._scripts[arm].act(sim, t)
            else:
                a, label = np.r_[READY, 1.0], "idle"
            actions.append(a)
            labels.append(label)
        if tuple(labels) != self._printed:
            print(f"[tool-demo] t={t:.2f} phases={labels} "
                  f"ik_worst_mm={max([s.worst_mm for s in self._scripts.values()] or [0]):.1f}", flush=True)
            self._printed = tuple(labels)
        meta = {"request_stop": True} if t > self._end + self.hold_s else {}
        return ActionChunk(actions=[Action(data=pack(*actions), meta=meta)], control_hz=1 / self._dt)
