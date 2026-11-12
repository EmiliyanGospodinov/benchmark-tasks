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

Bring the test chamber pressure into the green 54 to 66 kPa band without ever exceeding 80 kPa. Leave the pump with both feet on the grey mat, release it and move both hands clear; the pressure must remain in the band for two seconds.

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
- CPU checks: `python -m pytest tests` in this folder (scorer, mat placement and gas model, 20 tests).

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
The observation-only demo reads the native 320x176 overhead RGB image and exposed
14-D robot joint state. The fixed setup and calibrated robot geometry determine
approach targets. While holding the barrel, forward kinematics of the right arm
estimates its x position; no simulator part poses are used. This assumes the
barrel remains in the grasp and aligned with its initial axis.

The dial reader locates the white face using its green band, excludes outer tick
marks, and finds the darkest ray through the hub. Two consecutive readable
compression frames estimated at 58–75 kPa trigger lowering and release. The broad
upper reading bound accommodates pixel uncertainty; it is not an expanded task
target. The scorer still requires actual pressure of 54–66 kPa and caps credit
if pressure ever exceeds 80 kPa. Missing/occluded readings never count as success.
Motion timing selects phases, but no stroke count or stored action trace decides
when the pressure target has been reached.

CPU tests exercise the controller with simulator access poisoned. To replay
native recorded images through the exact runtime pressure rule:
`python tools/dial_gate_v2.py /path/to/run-with-frames-and-telemetry`.
Telemetry supplies offline truth labels only, never inputs to the policy.

The revised observation-only demo scored **1.0 in both episodes**, including the
repeat after reset, with seed 0 and the instruction above. Both completed at
460 steps (23.00 s); retained pressures were 62.534 and 62.537 kPa, and each met
the continuous two-second final hold. Each episode stopped on the visible dial
at 17.85 s. The two-episode run took 175 s including startup and reset on the
local RTX 5090 system. Scene, gas model and scorer are unchanged.

The exact-rule replay on this fresh run produced no below-band stop decisions;
both first stops occurred at the retained pressures above. Four occluded frames
per episode were unreadable. Validation covers this fixed camera/layout; this
small dial reader is not a general calibrated pressure meter.

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
model pressure. The scripted policy also uses only camera images and exposed joint state.

## License

Original task code and procedural geometry: MIT (see `LICENSE`). Isaac Sim, Isaac Lab Arena and the
YAM adapter remain external dependencies under their respective licenses.

## Media

The uploaded videos match this instruction and source. Dream-1 (`dreamlabs/dream-1-beta`,
seed 0, 1,800 steps) scored 0.0: the pump stayed unpressurised. The observation-only
scripted demo scored 1.0, and 1.0 again after a reset. Each video shows the left wrist,
right wrist and overhead cameras side by side at the policy's native 320x176.
