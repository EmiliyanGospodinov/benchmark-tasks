# Pick a clear training lock: hold light turning pressure with one hand while the other sets each pin

Locksmiths open locked-out doors by single-pin picking (O*NET 49-9091, locksmiths and safe repairers). One hand
keeps light turning pressure on the plug with a tension wrench; because the pin holes are never perfectly in
line, only one pin stack binds at a time. The other hand lifts that pin with a hook pick until the split between
its key pin and driver reaches the shear line: the plug gives slightly, the driver rests on the plug's edge and
the next pin binds. With every pin set the plug turns to open. Releasing the pressure drops the set pins;
forcing the plug harder does not open it. This task uses an enlarged (~3x) clear trainer with three pin stacks
so that the pins are visible to cameras.

The task is a task-local plugin for the `yam_arena` adapter of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station`;
Isaac Lab 3.0 runtime / Isaac Sim 6.1; local Arena package metadata reports 0.3.0). The verified runner is
Inspect Robots 0.60.0 with inspect-robots-yam 0.37.0. It registers the embodiment `yam_lock_picking` (the
adapter's robot, cameras and 14-D joint action/observation, unchanged, plus the trainer, the pick and their
joints) and the privileged scripted demonstration policy `lock_picking_scripted`.

## Instruction

`task.json` field `instruction`.

## Setup

See `task.json` (`setup`). Geometry, masses and springs are in `lock_picking/design.py`.

| Part | Description | Mass |
|---|---|---|
| housing on pedestal | cut-away housing (9.5 cm long, only thin chamber fins drawn) on a dark 9.5 x 6.4 x 6.8 cm pedestal, clamped; keyway face at x 25 cm facing the robot | fixed |
| plug | clear, 40 mm diameter, 90 mm long, axis along x 10 cm above the table; 9 x 24 mm keyway, 75 mm deep | 110 g |
| pin stacks | front / middle / back at 35 / 50 / 65 mm: 8 x 8 x 16 mm brass key pins, red / green / blue 18 mm drivers, springs 0.3 N preload (authored trainer spring), 30 N/m | 8.7 g / 9.8 g |
| tension wrench | lever on the plug face, 45 degrees up toward -y, orange free-spinning sleeve (18 x 30 mm) 10 cm out | 30 g + 10 g |
| hook pick | 4 x 4 x 130 mm steel shaft, 7 mm tooth, 75 mm dog-leg, 10 x 30 x 40 mm handle plate; lies on the table, tip at (28, 12) cm | 50 g |

## Scoring

`lock_picking/scoring.py` (pure Python, `python -m pytest tests` on CPU) and `task.json` (`scoring`):

| Score | Condition |
|---|---|
| 0, latched | wrench flexed > 15 degrees from the plug's pointer (> 0.15 N m, forcing), lock body moved > 2 mm, invalid state, or the plug crosses beyond its 15-degree binding region without all three pins set |
| 0.25 per pin | most pins set at the same time for >= 0.5 s (driver bottom within 0.8 mm of the shear line, its key pin top >= 1 mm below) |
| 1.0, success | plug >= 85 degrees (green open mark), wrench flex <= 3 degrees, pick tip out of the keyway, both grippers >= 80 % open with each centre >= 7 cm from the sleeve centre and keyway mouth, for 2 s |

The crossing check applies when leaving the binding region. Drivers may subsequently drop into the open detents; this does not invalidate a legitimate opening. Returning to <= 15 degrees requires a fresh valid crossing. Reverse rotation below -15 degrees is invalid.

## Verified evidence

The frozen scripted run `final-001` scored 1.0 in both episodes, succeeding in 1,837 and 1,848 actions (91.85 and 92.4 seconds), including the reset repeat. Replaying its recorded 20 Hz measurements through the hardened scorer preserves every per-step score. The scene, controller and 166-character instruction are unchanged. Three recorded shortcut diagnostics also remain 0.0. These comparisons support the demonstrated cooperative procedure; wrench propping was not tested.

## Running

Install the pinned adapter (it installs Isaac Lab-Arena and Inspect Robots), then this package into the same
environment. From the repository root:

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
uv pip install 'inspect-robots==0.60.0' 'inspect-robots-yam==0.37.0'
uv pip install --no-deps -e tasks/lock-picking
```

Scripted demonstration, two episodes with a reset between them (run from `tasks/lock-picking`; generated USD
parts go to `assets/generated/`):

```bash
cd tasks/lock-picking
inspect-robots run --embodiment yam_lock_picking --policy lock_picking_scripted \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --epochs 2 --store-frames --log-dir runs/lock-demo
# Video export additionally uses imageio and imageio-ffmpeg (the optional media extra).
python tools/video.py runs/lock-demo/frames/<run> scene-0-e0 demo-e0.mp4
```

- Without installing, `python run.py run ...` with the same arguments registers the two plugins in-process.
- A policy under test runs the same command with its own `--policy`.
- `-E telemetry=<file.jsonl>` writes per-step plug angle, wrench flex, pin heights, grippers and score.
- `-P mode=oneshots --epochs 3` runs the shortcut diagnostics (pick only, self-loaded wrench, nonoverlapping use of the two tools). The last diagnostic alternates the two arms; it is not a literal one-gripper tool-changing trial.
- Feasibility diagnostic, not a demonstration: `python -m lock_picking.probe OUT_DIR` (kinematic pick, stand-in
  hand on the wrench; the embodiment's `diag=1` exists only for it).
- Fixed layout, deterministic controller, seed 0. No domain randomization is used.
- Run limit: 2,400 steps (120 s at 20 Hz). The demonstration succeeds at about 92 s.
- The demonstration is privileged: it reads simulator part poses, plug/wrench angles and pin heights for motion and phase decisions, and knows the declared binding order. It does not infer pin setting from camera images. Policies under evaluation receive the adapter's camera images and joint state; they do not receive these scripted-policy state accesses.
- Pin state is visible in the right (tension) arm's wrist camera while it holds the sleeve; the top camera is too
  coarse to read it.

## Simulation notes and limitations

- Plug: native revolute joint in the housing (0..92 degrees), return (tailpiece) spring 0.04 N m at rest,
  0.00025 N m/deg. Pins: prismatic joints (key pins in the plug, drivers in the housing) with declared springs.
- Binding order: declared hole offsets (3.5 / 0.5 / 2.0 mm for front / middle / back) on the opening side;
  exaggerated from real tolerances so the order is deterministic and each set driver rests on a 1.5 mm ledge.
- Wrench: revolute joint to the plug with a declared torsion spring (0.010 N m/deg, stops at 25 degrees); its
  flex is the turning torque, read against the orange pointer on the plug face.
- Open detent: 2 mm notches under the drivers at 90 degrees hold the plug open once released.
- Square pins, no chamfers or security pins, no wear, no wrench slip in the keyway (it is fitted to the face).
- Physics 240 Hz; control and scoring 20 Hz.

## Final model evaluation

Use the same scene, scorer, instruction, seed and 2,400-action limit as the demo, with the Dream key configured
as in the organizer walkthrough. This command makes paid API calls; it is not a development smoke test.
From this task folder, the no-install invocation is:

```bash
python run.py run --embodiment yam_lock_picking --policy dream \
  -E diag=0 -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --epochs 1 --seed 0 --store-frames --log-dir runs/lock-dream
```

Dream score and uploaded media ID remain pending in `task.json`. The local preparation is not yet a received submission.
