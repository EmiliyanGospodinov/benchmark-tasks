# Lower a ratchet-held load in two stages with a released hold

This trainer practices controlled release of a ratchet-held load. One hand carries the load on the crank and
takes in slightly to unload the pawl. The other releases the pawl and holds it clear while the load is paid
out under control. To stop part-way, the pawl is let back onto the wheel and the crank eased down until it
catches. The handle is then released gradually to check that the pawl holds. Before restarting, the load is
taken on the crank again. The grip-before-release procedure is described in the [Fulton operating instructions](https://www.fultonperformance.com/product-assets/install/N142430.pdf) and [Dutton-Lainson pulling-winch instructions](https://www.dutton-lainson.com/assets/pdf/owner-manuals/206430.pdf). These references inform the control sequence.

This task is an original low-energy tabletop **trainer**, not a model of a commercial winch. A gravity-loaded
lever on a horizontal shaft is held by real ratchet teeth and a spring-returned pawl.

The task is a task-local plugin for the `yam_arena` adapter of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated
`station`; Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1). It registers:

- the embodiment `yam_ratchet_lowering`: the adapter's robot, cameras and 14-D joint action/observation,
  unchanged, plus the trainer and its joints;
- the privileged scripted demonstration policy `ratchet_lowering_scripted`.

## Instruction

The instruction is `instruction.txt`, the same as the `instruction` field of `task.json`:

1. Grip the orange crank handle and turn it slightly upward to take the load off the pawl.
2. With the other gripper, press the orange pawl tab down and keep it down.
3. Turn the crank slowly to lower the lever, never faster than 40 degrees per second, until the lever is just
   above horizontal.
4. Let the pawl tab back up, then ease the crank down until the pawl catches the next tooth, with the lever 1
   to 10 degrees above horizontal (the second tooth below the start).
5. Open both grippers and move them away: the lever must stay there on the pawl for at least 2 seconds.
6. Take the crank again, take in slightly, press the pawl tab down and lower the lever slowly until the weight
   rests on the wooden rest.
7. Let the pawl tab go, release the crank and move both grippers away.

## Setup

**Coordinates:** x forward from the line between the arm bases, y left, from the midpoint between them; heights
above the table top. **Lever angle** is the lever's angle above horizontal.

| Part | Description | Mass | Position |
|---|---|---|---|
| base plate and mat | dark steel plate 22 x 30 x 0.8 cm, clamped to the table, on a 30 x 38 cm mat | fixed | centred (27, -1) cm |
| uprights | two steel 3 x 1 cm uprights carrying the shaft | fixed | y = ±5 cm |
| shaft | 12 mm, along y | (rotor) | x 24 cm, 10 cm high |
| lever and weight | 13 cm steel lever pointing forward; blue aluminium weight 5 x 4 x 4 cm at its end | 50 g + 200 g | on the shaft at y 0 |
| ratchet wheel | 24 sawtooth teeth, tip radius 48 mm, root radius 40 mm, 12 mm thick | 80 g | on the shaft at y -8 cm |
| crank | 7 cm arm with an orange handle sleeve (20 mm diameter, 40 mm long) spinning freely on its pin | 30 g + 20 g | arm at y +9 cm, sleeve centred at y +11.6 cm |
| pawl | orange steel pawl with spring return; nose in the teeth; thumb tab 22 x 18 mm beyond the pivot | 30 g | pivot (29, -8) cm, 16.4 cm high; tab (33, -8) cm, 17.2 cm high |
| receiving rest | wooden pad under the weight; it stops the lever at -28 degrees | fixed | under the weight |

- **Rotor:** the shaft, lever, weight, wheel and crank form one 410 g rigid rotor.
- **Pawl spring:** a torsion spring (0.004 N m per degree) presses the pawl nose into the teeth. Pressing the
  tab down lifts the nose out.
- **Start:** the lever is 35.3 degrees above horizontal, with the pawl holding the load on a tooth. The pawl
  catches every 15 degrees (35.3, 20.3, 5.3, -9.7 ...). The middle tooth (5.3 degrees) is the only catch inside the 1-10 degree band.
- **Layout:** one fixed layout, with nothing randomized.

## Scoring

Lever angle comes from the rotor's actual pose (on a real trainer: a protractor on the frame, or the video). Speed is the
lever's angular speed. Both are measured at 20 Hz; the speed rule applies to those samples, not unobserved impact peaks between samples.

| Score | Condition |
|---|---|
| 0, latched | a sampled lever speed exceeds 40 degrees/s (a dropped or runaway load), or a non-finite state |
| 0.25 | lowered at least 10 degrees below the start |
| 0.5 | held at 1-10 degrees above horizontal, pawl at 2-5 degrees (engaged), still (under 1 degree/s), for 2 s with both grippers released and the whole robot clear as defined below |
| 0.75 | after that hold, lowered again below -10 degrees |
| 1.0, success, the trial ends | after the hold, resting within 1.5 degrees of -28 on the rest, still and released, for 2 s |

- Landing without the intermediate released hold scores at most 0.25.
- A lever still held at the finish is not success. Released means both grippers at least 80% open, both tool points between the fingertips at least 5 cm from the handle and tab centres, and at least 5 mm separation between every robot collision-shape world-axis bounding box and every moving trainer collision-shape box. Bounds come from measured poses and the actual collision geometry; they conservatively include empty space within each individual shape. The frame is excluded. The pawl-engagement band applies to the intermediate hold; at the final rest the unloaded pawl can ride on a tooth tip.
- The scorer is `ratchet_lowering/scoring.py`, in pure Python. `python -m pytest tests` checks it on CPU:
  untouched start, full sequence, unreleased hold, restart, landing without hold, runaway, landed but held,
  wrong tooth, non-finite input, an open pawl and support applied to another part while the tool points are clear.

## Running

Install the pinned adapter, which installs Isaac Lab-Arena and Inspect Robots, then this package into the
same environment. From the repository root:

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
uv pip install --no-deps -e tasks/ratchet-lowering
```

Scripted demonstration, two episodes with a reset between them. Run it from `tasks/ratchet-lowering`; the
generated USD parts are written to `assets/generated/`:

```bash
cd tasks/ratchet-lowering
inspect-robots run --embodiment yam_ratchet_lowering --policy ratchet_lowering_scripted \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(cat instruction.txt)" --scorer task_score --max-steps 1600 --epochs 2 \
  --store-frames --log-dir runs/ratchet-demo
```

- **Other policies:** a policy under test runs the same command with its own `--policy`.
- **Telemetry:** `-E telemetry=<file.jsonl>` writes a per-step trace of the lever angle, sampled speed, pawl angle, grippers, collision-shape clearance, sampled grip penetration and score. The diagnostic-only powered pawl opening used during development is absent from the submitted runtime.
- **Run limit:** 1,600 steps (80 s at 20 Hz).
- **Reset:** nothing is consumed. The reset puts the lever back on the start tooth.
- **Demonstration result:** 1.0 in both episodes, including a reset repeat, at 794 steps (39.7 s). Peak sampled speeds were 22.48 and 22.60 degrees/s. Both episodes took 235.5 s wall time including simulator startup on one GPU.
- **Dream-1 result:** 0.0 in one 1,600-step episode (80 s simulated, 178.4 s wall time), with no evaluation error. The load remained on the starting tooth; no lowering stage was achieved. Use `--policy dream --epochs 1` with the same instruction, scorer and 1,600-step limit.

The demonstration is privileged: it reads the simulator's part poses. Evaluation logs may use the adapter's inherited `yam_arena`/`yam_scripted` labels; the commands above select the task-specific plugin.

- **Compliance correction:** the YAM arm is compliant under the supplied adapter's unchanged PD gains. So
  while a gripper works the crank or the pawl tab, the script corrects its commanded angle by integrating
  the measured lever or pawl angle.
- **Procedure:** it follows the instruction. Lowering runs at about 10-16 degrees/s.
- **Outputs:** it outputs only ordinary joint targets through the guide's IK.

Development comparisons used the same physical mechanism with the earlier, stronger scripted grip targets:

- **`-P mode=pawlonly`** (pawl released with nobody on the crank): the load free-falls, peaking at 638
  degrees/s, and slams onto the rest.
- **`-P mode=noopen`** (crank only): the engaged pawl blocks lowering, and the lever stays at 35.3 degrees.

Letting the lever drop tooth by tooth with one hand on the pawl also exceeds the speed limit: a single
15-degree free drop reaches about 60 degrees/s. These runs support the two-hand procedure for this
mechanism; they do not prove every one-arm strategy impossible.

## Simulation notes and limitations

- **Joints:** the shaft, pawl and crank sleeve are native PhysX revolute joints. The shaft has no drive or
  damping, and the load is held only by contact between the pawl nose and the sawtooth teeth (convex-hull
  wedges).
- **Pawl spring:** a declared joint torsion spring. There is no motor, angular snapping, target-conditioned
  lock or software unlock.
- **Masses and friction:** nominal, with no wear or deformation. The demonstration commands an 18 mm gap on the 20 mm sleeve and a 16 mm gap on the 18 mm tab, avoiding excessive squeeze. Geometry diagnostics record sampled collider-vertex distances to the grip solids; they are not an exhaustive collision-penetration bound.
- **Scaling:** the 200 g load (about 0.29 N m on the shaft, 4 N at the crank handle) is scaled to the compliant
  arm's controllable range.
- **Rates:** physics runs at 240 Hz; control and scoring run at 20 Hz.

## Media

The submission includes the starting setup image, the corrected full-score scripted demonstration and the Dream-1 episode. The scripted reset repeat is also recorded. All evaluated runtime and instruction bytes match; only result text and media metadata are added afterward.
