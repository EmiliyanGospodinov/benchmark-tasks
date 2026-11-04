# Pack a parcel following a printed packing plan

As in e-commerce order picking, the robot tightly packs box-like products into a parcel to match a given
packing plan. The plan is printed on two sheets on the table: one per layer. This task evaluates matching
that fixed layout, including pushing boxes into place; it does not score optimization over alternative layouts.

The whole task is one task file, `box_packing.py`, for the `yam_arena` embodiment of
[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (YAM arms at the calibrated `station` in
Isaac Lab-Arena 0.3.1 / Isaac Lab 3.0 / Isaac Sim 6.1). It needs no other files or packages.

## Setup

Coordinates: x forward from the line between the two arm bases, y to the left, from the midpoint between the bases.

| Object | Size (x by y by z) | Mass | Material | Start (x, y) |
|---|---|---|---|---|
| carton, open, fixed to the table | inside 24 x 18 x 6.5 cm; 8 mm walls, 6 mm floor | fixed | cardboard | (26, 0) cm |
| blue box | 16 x 6 x 3.5 cm | 0.20 kg | cardboard | (26, 27) cm |
| red box | 7 x 6 x 3.5 cm | 0.08 kg | cardboard | (42, 29) cm |
| yellow box | 9 x 6 x 2.5 cm | 0.06 kg | cardboard | (36, 19) cm |
| purple box | 7 x 6 x 5 cm | 0.12 kg | cardboard | (10, 17) cm |
| green box | 12 x 5.5 x 3.5 cm | 0.15 kg | cardboard | (26, -27) cm |
| orange box | 7 x 6 x 5 cm | 0.12 kg | cardboard | (42, -25) cm |
| plan sheet, bottom layer | 15 x 12 cm | flat | paper | (47.8, 8) cm |
| plan sheet, top layer | 15 x 12 cm | flat | paper | (47.8, -8) cm |

All boxes stand on the table, axis-aligned. Each sheet shows the carton's opening at half scale and in the carton's
orientation (the sheet's far edge is the carton's far wall) as a black 3 mm outline, and each box of its layer as a
filled rectangle in the box's colour.

The plan:

- **Bottom layer (left sheet):** blue in the far-left corner; green along the far wall, right of blue; red along
  the left wall, in front of blue.
- **Top layer (right sheet):** yellow on blue, in the far-left corner.
- The blue box has 3 mm clearance from the far and left walls. Green has 3 mm clearance from the far wall
  and blue; red has 3 mm clearance from the left wall and blue. Yellow aligns with blue's far-left corner
  and rests directly on blue. Other free space is left empty.
- Purple and orange are not on the plan and stay out of the carton.

## Scoring

- 0.25 for each of blue, red, green and yellow whose axis-aligned bounding box has every face (near, far, left,
  right, bottom, top) within 5 mm of where the plan puts it.
- 0 if any part of purple or orange is inside the carton's opening below its rim.
- The trial ends early with success once all four are placed, every one at rest (under 1 cm/s and 0.5 rad/s), with
  purple and orange outside the carton.

`score`, `success` and `distance` (summed distance of the four packed boxes from their goals, in metres) are in
`box_packing.py`; run with `--scorer task_score`. Setting `BOX_PACKING_DEBUG=N` prints every N-th step how far each
packed box's faces are from their goal, in millimetres.
The numerical score is evaluated at the endpoint. Early success requires the resting/placement conditions
to persist for one second. The distractor test uses world-aligned bounding-box intersection with the opening.

## Running

Needs Linux with an NVIDIA RTX GPU (Isaac Sim needs RT cores) and about 200 GB of disk. Install the pinned
adapter, which installs Isaac Lab-Arena and Inspect Robots (from the root of this repository):

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
```

In each shell:

```bash
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
```

Scripted full-score demo (from the repository root):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=tasks/box_packing/box_packing.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("tasks/box_packing/task.json"))["instruction"])')" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir runs/box-packing-demo
```

A policy under test runs the same command with its own `--policy`. For Dream-1, store the API key as described
in the organizer walkthrough, then run one episode with the same task and instruction:

```bash
inspect-robots run --embodiment yam_arena --policy dream -E task=tasks/box_packing/box_packing.py \
  -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python -c 'import json; print(json.load(open("tasks/box_packing/task.json"))["instruction"])')" \
  --scorer task_score --max-steps 2400 --epochs 1 --store-frames --log-dir runs/box-packing-dream
```

`inspect-robots video runs/box-packing-demo/<log>.json` renders the stored frames to MP4.
Use the equivalent command with the Dream run log to export that evaluation.

- **Run limit:** 2,400 steps (2 minutes at 20 Hz).
- **Seeds and layout:** one fixed starting layout; use `-E object_jitter=0` (seed 0). With jitter, seeded resets
  shift the loose boxes by up to 3 cm.
- **Runtime:** the scripted demo succeeds after 1,634 steps (82 s simulated) with `task_score` 1.0; the run took
  about 3 minutes wall-clock on one GPU, including about 1 minute of Isaac Sim start-up.
- **Dream-1 result:** one `dreamlabs/dream-1-beta` episode reached `task_score` 0.0 at the 2,400-step limit,
  with no trial error. It took 326 seconds wall-clock, including simulator startup and API inference.
  The scripted reproduction reached 1.0 in 178 seconds wall-clock. Results use the instruction in `task.json`.

The demo packs one box at a time: it sets each box down a little in front of and right of its goal, then pushes it
left and far with the closed gripper until it stops against a wall or a packed box.
The arms act sequentially; this submission does not claim that the packing operation requires simultaneous
cooperative manipulation by both arms.

## Simulation notes

- The carton and all boxes are rigid cuboids: no cardboard deformation, flaps or tape. Contact uses PhysX's default
  friction.
- The printed plan is modelled as thin, fixed boxes on the table (1 mm paper, 1 mm ink on top).
- Physics runs at 240 Hz, control at 20 Hz.

## Media

The setup screenshot, full-score scripted video and Dream-1 evaluation video are uploaded separately to
the task board and identified by the `media` entry in `task.json`. Camera frames and evaluation logs are
generated by the commands above; they are not required to load the task.

The original demonstration was recorded at 640 x 360 per camera. The reproduced demonstration and
Dream-1 evaluation use 320 x 176 policy-camera frames. Both use the same fixed geometry and scorer.
