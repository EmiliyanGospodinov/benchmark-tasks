# Pump a test chamber to the pressure on its dial with a hand pump

One YAM gripper holds a free hand pump while the other works its handle. The policy must read the dial on
the gauge of the test volume and stop pumping at the specified pressure, then put the pump down and let go. The task
evaluates the chamber pressure, read on a dial, and that it is kept after release.

The task is a task-local plugin for the `yam_arena` adapter of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station`;
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1). It adds the embodiment `yam_hand_pump`, which builds
the pump, chamber and gauge on top of the unchanged robot, cameras and 14-D joint action/observation contract,
and the scripted demonstration policy `hand_pump_scripted`. Run this custom embodiment as documented
below; loading `task.py` alone with the unextended `yam_arena` embodiment does not build the mechanism.

## Instruction

The `instruction` field of `task.json` (also `instruction.txt`):

1. Take hold of the barrel with one gripper and the handle with the other; lift the pump off the mat together.
2. Keep the barrel still and pump: push the handle in, then pull it fully out; repeat.
3. Watch the dial. Stop pushing **immediately** when the black needle reaches the red 60 kPa mark (green band
   54-66 kPa), even partway through a stroke. Never exceed 80 kPa.
4. Put the pump back on the grey mat with both feet on the mat, open both grippers and move away.
5. Check the dial again: still within the green band.

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the
bases; heights above the table top.

| Part | Size | Mass | Position |
|---|---|---|---|
| rubber mat, fixed | 43 x 14 x 0.2 cm | fixed | centred (20, 0) cm |
| metal pump barrel (white, hollow; polymer end rings and feet) | 18 cm long, 4 cm diameter | 200 g | centred (12, 0) cm, axis 2.8 cm high, outlet toward the robots |
| aluminium piston rod with yellow polymer handle | handle 1.8 x 4.8 x 3.2 cm; rod 8 mm | 35 g | handle at (36, 0) cm, pulled fully out |
| polymer gauge housing with 12 cm dial, fixed | 12.8 cm diameter, 4 cm tall | fixed | (43, -16) cm |

Material descriptions are nominal construction specifications for this original design. Rigid pump contact
uses friction 0.8 static and 0.6 dynamic. Each foot is 3.0 x 3.6 x 0.9 cm, centred 6.5 cm on either side
of the barrel centre along x.

- **Pump:** single-acting, 10 mm bore, 8 cm stroke, 0.8 mL dead volume (6.3 mL per full stroke).
- **Test volume:** 30 mL of gas (internal volume, including the hose). The hose is not rendered or simulated.
  The gauge housing is a separate, larger fixed body that only carries the dial.
- **Dial:** 0-120 kPa over 270 degrees, clockwise, readable in the overhead camera.
  - Black ticks every 10 kPa, longer every 20 kPa.
  - A solid green band from 54 to 66 kPa, with a red 60 kPa mark pointing toward +x (away from the robots).
  - The black needle shows the test-volume pressure and starts at 0.
  - The wrist cameras do not see the dial while pumping; the overhead camera does.
- **Pressure per stroke:** with the barrel held, a full stroke initially adds roughly 20 kPa. The
  demonstration uses pressure feedback to stop during a compression stroke.
- One fixed layout; nothing is randomised. Evaluation seed: 0.

## Scoring

Implemented in `hand_pump/scoring.py` and reported as `info["score"]` for `--scorer task_score`. Inputs are
measured state only: the chamber pressure from the gas model (what the dial shows), each gripper's finger
opening and its TCP distance to the barrel and handle, and the barrel's height and speed.

| Score | Condition |
|---|---|
| 0.25 | the chamber has reached 20 kPa |
| 0.5 | it has been within 54-66 kPa |
| 0.75 | it is within 54-66 kPa now |
| 1.0, success, trial ends | within 54-66 kPa, pump put down on the mat (all eight corners of the two feet on the mat, barrel axis within 15 mm of its resting height, under 2 cm/s), both grippers at least 80 % open with their TCPs at least 7 cm from barrel and handle, continuously for 2 s |

- Ever exceeding 80 kPa caps the score at 0.25.
- Non-finite states score 0 for that step and restart the hold window.
- The untouched start scores 0.
- CPU checks: `python -m pytest tests` in this folder (scorer, mat placement and gas model, 18 tests).

## Running

Install the pinned adapter, then this package (from the repository root):

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
uv pip install --no-deps -e tasks/hand-pump
cd tasks/hand-pump
inspect-robots run --embodiment yam_hand_pump --policy hand_pump_scripted \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1800 --epochs 2 --seed 0 --store-frames --log-dir runs/pump-demo
```

- `-E telemetry=<file.jsonl>` writes per-step pressure, piston travel, part poses and scorer inputs.
- `-P mode=nohold` runs the pumping arm alone; `-P mode=idle` leaves the pump untouched (stability check).
- **Run limit:** 1,800 steps (90 s at 20 Hz).
- **Demonstration result:** `task_score` 1.0 in both of two episodes (with a reset between): success after
  458 and 457 steps (22.9 s and 22.85 s), 61.9 and 62.2 kPa, four strokes, pump back on the mat.

**The demonstration is privileged.** It reads part poses and the model's numerical pressure, the value
the dial displays. It does not read the gauge from the images.
1. The right gripper takes the barrel and the left takes the handle; they lift the pump 4 cm together.
2. The left arm strokes while the right counterholds.
3. Each compression is commanded 6 cm past the end stop, because the position-controlled arms are compliant
   (about 100 N/m between the two grippers in our measurements). It stops mid-stroke when the chamber
   reaches 59 kPa.
4. Both put the pump down and release.

## Simulation model

- **Mechanism:** barrel and piston are two free rigid bodies on a prismatic guide.
  - The barrel is a hollow tube, so the rod runs in a real bore.
  - Flat feet support the barrel on the mat; the piston guide supplies the internal bearing constraint.
- **Gas:** quasi-static isothermal model driven by the measured piston travel, with ideal inlet and outlet
  check valves.
  - The gas force acts equal and opposite on piston and barrel.
  - Seal drag is a 6 N s/m guide damping.
  - No leakage, heating, valve losses or hose forces, so the retained pressure is ideal.
- **Sizing:** bore, chamber volume and the 60 kPa target are chosen so the target needs about 5 N on the
  handle, which these simulated arms can apply. They are uncalibrated, nominal values for an original design,
  not a commercial pump.
- **Dial:** the needle is a display set from the model pressure each control step. It has no physical effect.
  Readability was checked on the policy's own 320x176 overhead frames at 34, 48.5, 53.4, 56.4 and 59.4 kPa: the
  needle's position relative to the green band is distinguishable, except within about 1-2 kPa of the band edges.
- **One arm (`-P mode=nohold`, diagnostic, earlier source without a saved snapshot):** the unheld pump is
  pushed about 21 cm along the table until it jams near the arm-base line, and pumping against that obstacle
  still raises the pressure. That run overshot to 74.8 kPa and scored 0.5. Its failure was overshoot, not an
  inability to pump: one-arm pumping by bracing is possible. The demonstrated procedure is cooperative
  counterholding; the task does not claim that one arm is mechanically incapable.

## Dream-1 evaluation

With Dream credentials configured locally, use the same scene, instruction, scorer and action limit:

```bash
inspect-robots run --embodiment yam_hand_pump --policy dream \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1800 --epochs 1 --seed 0 \
  --store-frames --log-dir runs/pump-dream
```

The Dream policy receives the standard three RGB views and joint state; it does not receive the
model pressure or the demonstration policy's privileged state access.

## License

Original task code and procedural geometry: MIT (see `LICENSE`). Isaac Sim, Isaac Lab Arena and the
YAM adapter remain external dependencies under their respective licenses.

## Media

Dream-1 (`dreamlabs/dream-1-beta`): **0.0**, one episode with seed 0, reaching the 1,800-step limit
without errors. The pump remained unpressurized. The scripted demonstration scored **1.0** in both
episodes, including the repeat after reset, using the same instruction, scene and scorer.

Recorded runtime on an RTX 5090: about 178 s wall time for the 90 s Dream episode (including simulator
startup and inference); about 112 s total for both scripted episodes including startup and reset.

The submission media are the complete Dream run, the full-score demonstration and the initial setup
image. Each video places left wrist, right wrist and overhead camera views side by side. The media ID
is assigned on upload.
