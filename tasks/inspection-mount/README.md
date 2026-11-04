# Aim a loaded inspection arm and lock it at the set angle

Setting up an inspection or camera fixture often takes two hands. One hand holds a loaded adjustable arm at
the required angle against gravity, and the other tightens the arm's clamp. The setting must then hold after
both hands let go. Here, one YAM gripper supports the arm's head at +15 degrees while the other turns the
clamp knob on the post. The task checks that the elevation is retained after both grippers withdraw.

What is evaluated is the arm's elevation and that it holds. No camera image from the inspection head is
evaluated.

The task is a task-local plugin for the `yam_arena` adapter of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station`;
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1). It adds the embodiment `yam_inspection_mount`, which
builds the fixture, its joints and the clamp model on top of the unchanged robot, cameras and 14-D joint
action/observation contract of `yam_arena`. It also adds the scripted demonstration policy `mount_scripted`.

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the
bases; heights above the table top. Elevation is the arm's angle above horizontal; positive raises the head.

| Part | Size | Mass | Position |
|---|---|---|---|
| post, fixed | 3 x 4 cm, 14 cm tall, on a 10 x 8 x 0.6 cm foot plate | fixed | centred at (29.5, 8) cm |
| hinge (axis along x) | — | — | (25, 8) cm, 12 cm high |
| arm bar | 1.6 x 1.2 cm, from the hinge toward -y | 60 g | — |
| inspection head (black block, orange lens) | 4 x 6 x 3.5 cm, centre 20 cm from the hinge axis | 250 g | — |
| clamp knob (orange), vertical axis | ridged round, 4 cm diameter, 2.5 cm tall, on a stem 1.2 cm above the post | 120 g | (29.5, 8) cm |

- **Start:** the arm rests on its lower stop, 26 degrees below horizontal, and the knob is loose.
- **Hinge travel:** −26 to +45 degrees.
- **Target:** +15 degrees. The head centre is then 5.2 cm above the hinge axis, 17.2 cm above the table, so the
  setting can be checked with a ruler.

For physical reconstruction, use a rigid metal post and foot plate, a lightweight hollow metal arm,
and a polymer dummy head ballasted to the stated mass; the knob is die-cast metal. These specify an
original analogue, not a calibrated commercial fixture. Simulation masses and friction are authored.

## Instruction

The text instruction is the `instruction` field of `task.json`:

1. Support the head.
2. Swing the arm to +15 degrees.
3. Keep the aim while the other gripper tightens the knob clockwise to its stop. Loosen it first if the arm is
   locked elsewhere.
4. Open both grippers and withdraw.
5. The arm must stay at +15 degrees on its own.

## Clamp model (declared, reduced-order)

- **Unlocked hinge:** an ideal revolute joint with no friction, so the loaded arm falls to its lower stop when
  released.
- **Knob:** a revolute joint about z.
  - Viscous damping 2e-4 N m s/deg, capped at 0.5 N m.
  - Stops at 0 (loose) and 170 degrees clockwise. The 170-degree stop is this prototype's chosen travel limit.
- **Indexed lock:** a 72-tooth serrated lock with 5-degree steps, a nominal original design of the
  serrated-locking-plate class.
  - Engages once the knob has been turned at least 150 degrees clockwise. The teeth mesh at the tooth nearest the
    arm's current angle.
  - Modelled as an angular spring to that tooth angle: 2 N m/deg, damping 0.02 N m s/deg, torque capped at 20 N m.
  - Turning the knob back below 130 degrees disengages it.
- **What triggers the lock:** only the knob's measured rotation. Nothing depends on the goal, a timer or the
  script phase, and nothing is teleported.
- **Limits of the model:**
  - The 20 N m cap is an authored simulation parameter, not a measured tooth strength.
  - Tooth shear, wear and breakage are not simulated.
  - The clean free-or-meshed transition is an idealisation: real serrated joints can have some drag before they
    mesh.

## Scoring

Implemented in `mount_aim/scoring.py` and reported as `info["score"]` for `--scorer task_score`. Every input
is measured:

- elevation, from the arm body's pose;
- the lock state, from the knob's rotation;
- each gripper's finger opening and its TCP distance to the head and knob centres.

| Score | Condition |
|---|---|
| 0.25 | the head has been raised within 10 degrees of +15 |
| 0.5 | the head has been within 3 degrees of +15 |
| 0.75 | while the lock is meshed with the arm within 3 degrees |
| 1.0, success, trial ends | meshed and within 3 degrees; both grippers at least 80 % open with TCPs at least 7 cm from the head and knob centres; all of this holding for 2 s with elevation deviating no more than 2 degrees from its value at the start of that window |

- Non-finite states score 0 for that step and restart the window.
- The untouched start scores 0.
- An arm locked at the wrong angle scores 0, unless it was raised near the target earlier.

CPU checks (no simulator): `python -m pytest tests` in this folder. They cover:

- the untouched start;
- a wrong angle, locked;
- the right angle while still held;
- an unlocked release;
- a full stable release;
- drift, out-of-tolerance angles and non-finite states.

## Running

Needs Linux with an NVIDIA RTX GPU. Install the pinned adapter, which installs Isaac Lab-Arena and Inspect
Robots. From the repository root:

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
uv pip install --no-deps -e tasks/inspection-mount   # registers yam_inspection_mount and mount_scripted
```

Scripted full-score demonstration. Run it from `tasks/inspection-mount`; the generated USD parts are written to
`assets/generated/`:

```bash
cd tasks/inspection-mount
inspect-robots run --embodiment yam_inspection_mount --policy mount_scripted \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir runs/mount-demo
```

- **Other policies:** a policy under test runs the same command with its own `--policy`.
- **Telemetry:** `-E telemetry=<file.jsonl>` writes a per-step trace of the elevation, knob angle, lock state,
  gripper state and score.
- **Comparisons:** `-P mode=nolock` aims and releases without tightening; `-P mode=prelock` tightens first and
  then tries to aim.
- **Run limit:** 1,200 steps (60 s at 20 Hz).
- **Layout:** one fixed layout. There are no free objects, and `object_jitter` is 0.
- **Demonstration result:** `task_score` 1.0, with success after 344 steps (17.2 s simulated). A repeat after a
  reset, two epochs in one process, also scored 1.0 in both episodes.

The demonstration is privileged: it reads the simulator's part poses, and its commands still go through the
guide's IK as ordinary joint targets.

1. The right gripper pinches the head across the hinge axis and swings the arm up in four arc steps.
2. Meanwhile the left gripper snugs the knob with a first stroke, stopping at 120 degrees, short of the
   150 degrees at which the lock meshes. Once the head is at the aim it tightens the knob to its stop with a
   second stroke while the right gripper keeps the aim.
3. Both withdraw.

The script commands +19 degrees so that the position-controlled arm, sagging under the load, settles near +15.
Both the instruction and the scorer use the actual +15 degrees.

Comparisons in this model, from scripted runs:

- **Release before locking:** with the current fixture, the arm reaches +15.5 degrees, then falls back to
  its -26-degree lower stop after release; score 0.5 for having reached the aim.
- **Lock first, then aim with one gripper:** with the current fixture, the lock meshes at -25 degrees.
  The scripted attempt reaches only -19.6 degrees before slipping, then returns to -25.3 degrees on release;
  score 0.0. Both comparisons use the same mechanism and code as the full-score demonstration.

These support the cooperative role of the two arms for this mechanism. They do not prove every one-arm
strategy impossible.

## Simulation notes

- All parts are rigid primitives. The post is a kinematic body, and the foot plate is the task file's fixed
  object.
- Physics runs at 240 Hz and control at 20 Hz. The robot uses the adapter's default contact model.

## Media

Dream-1 (`dreamlabs/dream-1-beta`): one episode, seed 0, 1,200 steps, task_score **0.0**,
terminated at the step limit without errors. It used the same text instruction, fixture, cameras and
scorer as the demonstration. The scripted demonstration scored **1.0**, including a repeat after reset.

To reproduce the Dream evaluation (requires a funded Dream API account), use the demonstration command
with `--policy dream --epochs 1 --seed 0` and a separate log directory. This calls the paid API.

The submission includes the complete Dream video, full-score demonstration video and initial setup
image. The media ID is assigned on upload; the videos show the left wrist, right wrist and overhead
cameras side by side.
