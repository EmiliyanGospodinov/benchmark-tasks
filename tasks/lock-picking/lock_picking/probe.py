"""L1 feasibility probe (preset mechanism diagnostic; robot idle at READY; NOT a demonstration).

The pick is kinematic and posed in the keyway in the plug frame (it follows the plug's turn); the wrench is
held by a stand-in angular "hand" drive. Everything else is the task mechanism unchanged. Checks:
  E  no turning pressure: the first-binding pin lifted to the shear line falls back (does not set)
  A  under pressure, a non-binding pin lifted and lowered falls back; the plug does not turn
  B  under pressure, the binding pin lifted until the plug gives sets and stays set with the pick lowered
  C  pressure released: set pins drop and the plug returns to 0
  D  all three set in binding order; the plug turns to the open mark and stays there with the hand removed

Run from tasks/lock-picking:  python -m lock_picking.probe OUT_DIR
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "probe-out")
HAND_K = 0.05  # N m/deg, stand-in hand stiffness
TENSION_FLEX = 6.0  # deg of wrench flex wanted (0.06 N m)
SHAFT_REST = -0.014  # shaft centre on the keyway floor, plug frame
LIFT_RATE = 0.002  # m/s


def main():
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena.constants import READY

    from lock_picking import design as d
    from lock_picking.embodiment import LockEmbodiment

    OUT.mkdir(parents=True, exist_ok=True)
    emb = LockEmbodiment(diag=1, control_hz=20, cam_width=320, cam_height=176, telemetry=str(OUT / "telemetry.jsonl"))
    obs = emb.reset(types.SimpleNamespace(instruction="probe", target=None), seed=0)
    # both arms parked in their picking poses, grippers open (camera check: arms present over the lock)
    from yam_arena import ik, station
    from yam_arena.demo import grip_rotation
    tip = np.array([d.pin_world_x(1) + 0.001, 0.0, d.AXIS_Z - 0.0134])
    lq, _, _ = ik.solve(tip + np.array(d.POST_GRIP_LOCAL) + (0, 0, 0.02), ik.matrix_to_quat_xyzw(grip_rotation(-math.pi / 2)),
                        base_pos=station.BASES["left"], iters=300)
    from yam_arena.urdf import axis_angle_matrix
    tx, ty = d.WRENCH_GRIP_TILT_DEG
    rrot = axis_angle_matrix(np.array([0, 1.0, 0]), math.radians(ty)) @ axis_angle_matrix(np.array([1.0, 0, 0]), math.radians(tx)) @ grip_rotation(0.0)
    rq, _, _ = ik.solve(d.sleeve_center(math.radians(8.0)), ik.matrix_to_quat_xyzw(rrot),
                        base_pos=station.BASES["right"], iters=300)
    act = Action(data=pack(np.r_[lq, 1.0], np.r_[rq, 1.0]))
    dt = 1.0 / 20
    log = (OUT / "probe.jsonl").open("w")
    frames = {k: [] for k in obs.images}
    st = {"pick_x": d.FACE_X + 0.010, "lift": 0.0, "hand": 0.0, "chase": False, "cap": 20.0, "t": 0.0}
    results = {}

    def pose_pick():
        s = emb.state()
        th = math.radians(s["theta_deg"])
        local = np.array([0.0, 0.0, SHAFT_REST + st["lift"]])
        world = d.AXIS + d.rot_x(th) @ local
        world[0] = st["pick_x"]
        emb.set_pose("pick", np.r_[world, d.quat_x(th)])

    def step(label=""):
        s = emb.state()
        if st["chase"] and st["hand"] > 0:
            target = min(s["theta_deg"] + TENSION_FLEX + TENSION_FLEX * d.FLEX_STIFF_DEG / HAND_K, st["cap"])
            emb.set_hand(st["hand"], target)
        else:
            emb.set_hand(0.0, 0.0)
        pose_pick()
        o = emb.step(act)
        st["t"] += dt
        info = emb.last
        row = {"t": round(st["t"], 2), "label": label, "theta": round(s["theta_deg"], 3), "flex": round(s["flex_deg"], 3),
               "drv": info["driver_bottom_mm"], "key": info["key_top_mm"], "lift_mm": round(st["lift"] * 1000, 2),
               "pick_x": round(st["pick_x"], 4), "score": info["score"], "reason": info.get("score_reason"),
               "set": info.get("pins_set")}
        log.write(json.dumps(row) + "\n")
        if round(st["t"] / dt) % 4 == 0:
            for k, v in o.observation.images.items():
                frames[k].append(v)
        return row

    def hold(sec, label):
        row = None
        for _ in range(round(sec / dt)):
            row = step(label)
        return row

    def move_x(x, label, speed=0.02):
        while abs(st["pick_x"] - x) > 1e-6:
            st["pick_x"] += float(np.clip(x - st["pick_x"], -speed * dt, speed * dt))
            step(label)

    def lift(i, label, stop_on_turn=True, max_lift=0.0095):
        th0 = emb.state()["theta_deg"]
        row = None
        while st["lift"] < max_lift:
            st["lift"] = min(st["lift"] + LIFT_RATE * dt, max_lift)
            row = step(label)
            if stop_on_turn and row["theta"] > th0 + 1.5:
                break
        row = hold(0.6, label + " hold up")
        top = dict(row)
        while st["lift"] > 0:
            st["lift"] = max(st["lift"] - 0.004 * dt, 0.0)
            step(label + " lower")
        row = hold(1.0, label + " lowered")
        return top, row

    def under(i):
        return d.pin_world_x(i) + 0.002

    def snap(name):
        for k, v in emb._observation().images.items():
            import imageio.v2 as iio
            iio.imwrite(OUT / f"{name}_{k}.png", v)

    first, second, third = d.binding_order()
    hold(1.0, "settle")
    snap("start")
    results["start"] = hold(0.05, "start")
    # E: no tension, lift the would-be first binder
    move_x(under(first), "E move")
    results["E"] = lift(first, "E lift binder, no tension", stop_on_turn=False, max_lift=0.0090)
    # A: tension on, lift a non-binding pin
    st["hand"], st["chase"] = HAND_K, True
    results["A_tension"] = hold(1.5, "A tension on")
    move_x(under(third), "A move")
    results["A"] = lift(third, "A lift non-binder", stop_on_turn=False, max_lift=0.0090)
    # B: lift the binder until the plug gives
    move_x(under(first), "B move")
    results["B"] = lift(first, "B lift binder")
    snap("one_set")
    move_x(under(second), "B2 move")
    results["B2"] = lift(second, "B2 lift second binder")
    # C: release tension
    st["hand"], st["chase"] = 0.0, False
    snap("two_set")
    results["C"] = hold(8.0, "C tension released")
    snap("released")
    # D: full sequence
    st["hand"], st["chase"] = HAND_K, True
    hold(1.0, "D tension on")
    for k, i in enumerate((first, second, third)):
        move_x(under(i), f"D move {k}")
        results[f"D{k}"] = lift(i, f"D lift {d.PIN_NAMES[i]}")
    snap("all_set")
    move_x(d.FACE_X - 0.03, "D withdraw pick")
    while st["cap"] < 95.0:  # turn at <= 20 deg/s
        st["cap"] = min(st["cap"] + 20.0 * dt, 95.0)
        step("D turn")
    results["D_turn"] = hold(2.0, "D turn hold")
    st["hand"], st["chase"] = 0.0, False
    results["D_release"] = hold(3.0, "D hand off")
    snap("open")
    log.close()
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    import imageio.v2 as iio
    for k, v in frames.items():
        iio.mimwrite(OUT / f"probe_{k}.mp4", v, fps=5, codec="libx264", pixelformat="yuv420p")
    print("[probe] results " + json.dumps(results), flush=True)
    print("[probe] done", flush=True)
    emb.close()


if __name__ == "__main__":
    main()
