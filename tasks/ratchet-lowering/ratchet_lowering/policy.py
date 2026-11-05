"""Policy ``ratchet_lowering_scripted``: privileged demonstration (reads simulator part poses).

Left gripper: the crank handle sleeve (fingers across the sleeve, TCP on the sleeve axis, kept vertical; the
sleeve spins on its pin). Right gripper: the pawl's thumb tab (fingers across the tab), moved along the
tab's arc about the pawl pivot. Procedure (as in winch release instructions): take the load on the crank and
take in slightly to unload the pawl; open the pawl; lower slowly; let the pawl back onto the wheel; ease
down until it catches; release the crank, withdraw both hands and leave the load on the pawl; then retake,
unload, open and lower onto the receiving rest; release and withdraw.
Outputs ordinary 14-D joint targets through the guide's IK; never sets object poses.

The YAM arm is compliant (its PD gains are those of the real robot), so under the crank load the gripper
sags below its commanded pose. While a gripper works a part, the demo commands that part's angle (crank:
lever angle; tab: pawl angle) with a rate-limited integral correction on the measured angle (privileged
read), as a person adjusts to the felt load.

``-P mode=coop`` (default) is the demonstration; ``-P mode=idle`` keeps both arms at the ready pose;
``-P mode=noopen`` takes the crank load but never touches the pawl (one-arm comparison: try to lower);
``-P mode=pawlonly`` opens the pawl with the right gripper while nobody holds the crank (release without
takeover), holds it open 1 s, then lets it go.
"""

from __future__ import annotations

import math

import numpy as np
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy

from . import design as d
from .motion import Script, Seg

HANDLE_SQUEEZE = -0.009  # target 18 mm gap on the 20 mm sleeve
TAB_SQUEEZE = -0.008  # target 16 mm gap on the 18 mm tab; avoid excessive squeeze
TAKE_IN_DEG = 3.0  # crank turned this far up to unload the pawl
PAWL_OPEN_DEG = 12.0  # nose clear of the tooth tips
LOWER_DPS = 10.0  # controlled lowering speed
STEP_DEG = 3.0  # arc subdivision (chord error < 0.1 mm on the 70 mm crank)
ABOVE_CATCH_DEG = 2.0  # stop this far above the catch, re-engage the pawl, then ease down
EASE_BELOW_DEG = 0.6  # command this far below the catch; the pawl takes the load
# Approach/withdraw offsets at working height (straight up is beyond the arms' reach here): the crank
# gripper slides onto the sleeve from outboard (+y), fingers either side of it; the pawl gripper slides in
# from the robot side (-x), fingers either side of the tab.
HANDLE_OFF = np.array([0.0, 0.070, 0.010])
TAB_OFF = np.array([-0.070, 0.0, 0.012])


def _handle(theta_deg, off=None):
    pos = d.handle_center(math.radians(theta_deg)) + (0 if off is None else off)
    return lambda sim: (pos, grip_rotation(math.pi / 2))


def _tab(beta_deg, off=None):
    pos = d.tab_center(math.radians(beta_deg)) + (0 if off is None else off)
    return lambda sim: (pos, grip_rotation(0.0))


BIAS_GAIN = {"crank": 1.5, "pawl": 2.0}  # 1/s, integral correction of the commanded angle
BIAS_RATE = 8.0  # deg/s, limit on the correction's change (no stick-slip jumps)
BIAS_MAX = 30.0  # deg
NO_INTEGRATE = ("ease onto pawl", "load on pawl", "landed")  # the pawl or the rest carries the load here


class _Live:
    """Target for a worked part: desired angle plus the policy's integral correction (read each step)."""

    def __init__(self, policy, theta_deg, kind="crank"):
        self.policy, self.theta, self.kind = policy, theta_deg, kind

    def __call__(self, sim):
        angle = self.theta + self.policy.bias[self.kind]
        return (_handle if self.kind == "crank" else _tab)(angle)(sim)


def _pawl(policy, beta, t, finger, label):
    return Seg(t, _Live(policy, beta, "pawl"), finger, label, freeze=False)


def _arc(policy, theta0, theta1, finger, label, dps=LOWER_DPS):
    n = max(1, math.ceil(abs(theta1 - theta0) / STEP_DEG))
    dt = abs(theta1 - theta0) / dps / n
    return [Seg(max(dt, 0.15), _Live(policy, theta0 + (theta1 - theta0) * (k + 1) / n), finger, label, freeze=False)
            for k in range(n)]


def _hold(policy, theta, t, finger, label):
    return [Seg(t, _Live(policy, theta), finger, label, freeze=False)]


class RatchetScriptedPolicy(YamScriptedPolicy):
    def __init__(self, mode="coop", duration=8.0, **kwargs):
        super().__init__(**kwargs)
        self.mode, self.duration = str(mode), float(duration)
        assert self.mode in ("coop", "idle", "noopen", "pawlonly"), self.mode

    def _build_plans(self, theta0):
        catch = theta0 - 2 * d.TOOTH_DEG  # the second tooth below the start
        rest = d.THETA_REST_DEG
        L, R = [], []  # left (crank) and right (pawl) segments, kept in lock-step per phase

        def phase(left, right):
            tl, tr = sum(s.duration for s in left), sum(s.duration for s in right)
            t = max(tl, tr)
            def fill(segs, done, dt):
                last = segs[-1] if segs else (done[-1] if done else None)
                finger = last.finger if last else FINGER_OPEN
                if last is not None and isinstance(last.target, _Live):  # keep working the part
                    return segs + [Seg(dt, _Live(self, last.target.theta, last.target.kind), finger, "wait",
                                       freeze=False)]
                return segs + [Seg(dt, None, finger, "wait")]
            if tl < t:
                left = fill(left, L, t - tl)
            if tr < t:
                right = fill(right, R, t - tr)
            L.extend(left)
            R.extend(right)

        def take_and_open(theta, n):
            phase([Seg(2.5, _handle(theta, HANDLE_OFF), FINGER_OPEN, f"crank approach {n}"),
                   Seg(1.2, _handle(theta), FINGER_OPEN, "crank slide on"),
                   Seg(0.8, _handle(theta), HANDLE_SQUEEZE, "crank grip")],
                  [Seg(2.5, _tab(0.0, TAB_OFF), FINGER_OPEN, f"tab approach {n}"),
                   Seg(1.2, _tab(0.0), FINGER_OPEN, "tab slide in"),
                   Seg(0.8, _tab(0.0), TAB_SQUEEZE, "tab grip")])
            phase(_arc(self, theta, theta + TAKE_IN_DEG, HANDLE_SQUEEZE, "take in", dps=6.0)
                  + _hold(self, theta + TAKE_IN_DEG, 2.5, HANDLE_SQUEEZE, "take the load"), [])
            phase(_hold(self, theta + TAKE_IN_DEG, 1.8, HANDLE_SQUEEZE, "hold for pawl"),
                  [_pawl(self, PAWL_OPEN_DEG * 0.5, 1.0, TAB_SQUEEZE, "open pawl"),
                   _pawl(self, PAWL_OPEN_DEG, 0.8, TAB_SQUEEZE, "open pawl")])
            return theta + TAKE_IN_DEG

        def release_both(theta, n):
            phase([Seg(0.6, None, FINGER_OPEN, "crank release"),
                   Seg(1.2, _handle(theta, HANDLE_OFF), FINGER_OPEN, f"crank withdraw {n}")],
                  [Seg(0.6, None, FINGER_OPEN, "tab release"),
                   Seg(1.2, _tab(0.0, TAB_OFF), FINGER_OPEN, f"tab withdraw {n}")])

        # stage 1: lower to the intermediate tooth and leave it on the pawl
        th = take_and_open(theta0, 1)
        phase(_arc(self, th, catch + ABOVE_CATCH_DEG, HANDLE_SQUEEZE, "lower"), [])
        phase(_hold(self, catch + ABOVE_CATCH_DEG, 2.9, HANDLE_SQUEEZE, "hold above catch"),
                  [_pawl(self, PAWL_OPEN_DEG * 0.5, 0.8, TAB_SQUEEZE, "close pawl"),
                   _pawl(self, 2.0, 0.8, TAB_SQUEEZE, "close pawl"),
                   Seg(0.5, None, FINGER_OPEN, "tab let go"),
                   Seg(0.8, _tab(0.0, TAB_OFF * 0.5), FINGER_OPEN, "tab clear")])
        phase(_arc(self, catch + ABOVE_CATCH_DEG, catch - EASE_BELOW_DEG, HANDLE_SQUEEZE, "ease onto pawl", dps=3.0), [])
        phase(_hold(self, catch - EASE_BELOW_DEG, 1.0, HANDLE_SQUEEZE, "load on pawl"),
              [Seg(1.0, _tab(0.0, TAB_OFF), FINGER_OPEN, "tab withdraw")])
        release_both(catch - EASE_BELOW_DEG, 1)
        phase([Seg(3.0, None, FINGER_OPEN, "hands clear: hold check")],
              [Seg(3.0, None, FINGER_OPEN, "hands clear: hold check")])
        # stage 2: retake, unload, open and land on the rest
        th = take_and_open(catch, 2)
        phase(_arc(self, th, rest - 0.5, HANDLE_SQUEEZE, "lower to rest"), [])
        phase(_hold(self, rest - 0.5, 1.4, HANDLE_SQUEEZE, "landed"),
              [_pawl(self, PAWL_OPEN_DEG * 0.5, 0.8, TAB_SQUEEZE, "close pawl"),
               _pawl(self, 2.0, 0.6, TAB_SQUEEZE, "close pawl")])
        release_both(rest, 2)
        return {"left": L, "right": R}

    def reset(self, scene):
        # The plan is built on the first act(): reset() can run before the scene's own reset in later
        # episodes, when the lever still sits where the previous episode left it.
        self._scripts, self._built = {}, False
        self.bias = {"crank": 0.0, "pawl": 0.0}
        self._end = self.duration
        self._printed = None

    def _build(self, sim):
        self._built = True
        if self.mode != "idle":
            theta0 = sim.state()["theta_deg"]
            p = self._build_plans(theta0)
            if self.mode == "noopen":  # crank hand only: take the load and try to lower without the pawl hand
                th = theta0
                segs = [Seg(2.5, _handle(th, HANDLE_OFF), FINGER_OPEN, "crank approach"),
                        Seg(1.2, _handle(th), FINGER_OPEN, "crank slide on"),
                        Seg(0.8, _handle(th), HANDLE_SQUEEZE, "crank grip")]
                segs += _arc(self, th, th + TAKE_IN_DEG, HANDLE_SQUEEZE, "take in", dps=6.0)
                segs += _hold(self, th + TAKE_IN_DEG, 1.5, HANDLE_SQUEEZE, "take the load")
                segs += _arc(self, th + TAKE_IN_DEG, th - 2 * d.TOOTH_DEG, HANDLE_SQUEEZE, "try to lower")
                segs += [Seg(0.6, None, FINGER_OPEN, "crank release"),
                         Seg(1.2, _handle(th - 2 * d.TOOTH_DEG, HANDLE_OFF), FINGER_OPEN, "crank withdraw")]
                p = {"left": segs}
            elif self.mode == "pawlonly":
                p = {"right": [Seg(2.5, _tab(0.0, TAB_OFF), FINGER_OPEN, "tab approach"),
                               Seg(1.2, _tab(0.0), FINGER_OPEN, "tab slide in"),
                               Seg(0.8, _tab(0.0), TAB_SQUEEZE, "tab grip"),
                               _pawl(self, PAWL_OPEN_DEG * 0.5, 1.0, TAB_SQUEEZE, "open pawl"),
                               _pawl(self, PAWL_OPEN_DEG, 0.8, TAB_SQUEEZE, "open pawl"),
                               _pawl(self, PAWL_OPEN_DEG, 1.0, TAB_SQUEEZE, "hold open"),
                               Seg(0.6, None, FINGER_OPEN, "tab release"),
                               Seg(1.2, _tab(0.0, TAB_OFF), FINGER_OPEN, "tab withdraw")]}
            self._scripts = {a: Script(s, self._bases[a]) for a, s in p.items()}
            self._end = max(s.duration for s in self._scripts.values()) + 3.0

    def _integrate(self, sim, t):
        """Integral corrections while a gripper works a part: desired angle from its current live segment."""
        state = None
        for arm, kind in (("left", "crank"), ("right", "pawl")):
            script = self._scripts.get(arm)
            if script is None:
                continue
            acc, prev = 0.0, None
            for s in script.segs:
                if t < acc + s.duration:
                    break
                acc += s.duration
                prev = s
            else:
                continue
            if not isinstance(s.target, _Live) or s.label in NO_INTEGRATE:
                continue
            u = (t - acc) / s.duration
            start = prev.target.theta if prev is not None and isinstance(prev.target, _Live) else s.target.theta
            want = start + (s.target.theta - start) * u
            state = state or sim.state()
            err = want - state["theta_deg" if kind == "crank" else "pawl_deg"]
            step = float(np.clip(BIAS_GAIN[kind] * err, -BIAS_RATE, BIAS_RATE)) * self._dt
            self.bias[kind] = float(np.clip(self.bias[kind] + step, -BIAS_MAX, BIAS_MAX))

    def act(self, observation):
        from . import embodiment

        sim = embodiment.ACTIVE
        if not self._built:
            self._build(sim)
        t = float(observation.state_time) + self._dt
        self._integrate(sim, t)
        out, labels = [], []
        for a in ("left", "right"):
            if a in self._scripts:
                act, label = self._scripts[a].act(sim, t)
            else:
                act, label = np.r_[READY, 1.0], "-"
            out.append(act)
            labels.append(label)
        if tuple(labels) != self._printed and self._scripts:
            print(f"[ratchet-demo] t={t:5.2f} left={labels[0]} right={labels[1]} bias={self.bias['crank']:+.1f}/{self.bias['pawl']:+.1f}deg "
                  f"ik_worst_mm={max(s.worst_mm for s in self._scripts.values()):.1f}", flush=True)
            self._printed = tuple(labels)
        meta = {"request_stop": True} if t > self._end else {}
        return ActionChunk(actions=[Action(data=pack(*out), meta=meta)], control_hz=1.0 / self._dt)
