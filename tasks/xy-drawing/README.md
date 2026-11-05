# Two-knob XY path drawing

An original mechanical teaching plotter: one knob moves the stylus in X and the other in Y. The task is to follow two oblique segments accurately, starting at blue, passing orange and ending at purple. The teal lane is a printed reference; black ink records measured stylus motion. Ending at purple without drawing the path does not pass.

The supplied text manual explains the controls and the two segment displacements. Both controls can be turned together to produce an oblique line. Fine alternating turns with one gripper remain allowed; the scorer contains no simultaneous-hand predicate.

## Setup and execution

Use the pinned organizer adapter bootstrap on Linux with an RTX GPU. From the repository root:

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
```

Validated installed metadata: Isaac Sim6.1.0.0, Isaac Lab17.0.2, Isaac Lab Arena0.3.0, inspect-robots0.60.0, PyTorch2.11.0+cu128, yam-arena0.1.0. The adapter checkout was clean at the pinned commit above. These installed versions differ from guide branch labels; use the pinned bootstrap. Then:

```bash
cd tasks/xy-drawing
uv pip install --no-deps -e .
python -m pytest -q tests
mkdir -p eval
inspect-robots run --embodiment yam_xy_drawing --policy xy_drawing_scripted \
  -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  -E telemetry="$PWD/eval/telemetry.jsonl" \
  --instruction "$(cat instruction.txt)" --scorer task_score \
  --max-steps 6000 --epochs 2 --seed 0 --store-frames --log-dir "$PWD/eval"
```

The scripted policy emits normal 14-D absolute joint targets through the standard adapter. It does not set object poses. The alternative controls are `-P mode=coarse` and `-P mode=fine`, both using only the left arm while the right remains at READY. They use measured knob angles for turn corrections. The cooperative demo uses calibrated Cartesian waypoints and ordinary IK. No learned-model evaluation is launched by these commands.

The scene, generated USD parts and original reference markings are built locally from included code. The standard room, robots, camera poses, observations and action contract are retained. Camera resolution above is for evidence recording: the installed Dream adapter encodes its actual inputs at **320x176**, irrespective of higher recording resolution. The full trace shape is visible there; exact lane-edge judgments are limited by pixel resolution.

## Mechanism and outcome

The knobs are physical rigid bodies on revolute joints; the two stages are bodies on native prismatic joints. A declared rotary/linear transmission maps measured knob rotation to a force-driven stage target and feeds resisting torque back to the knob. This reuses the constitutive coupling approach from the mechanical-puller prototype; it is not a native rack-and-pinion joint or a commercial Etch A Sketch replica.

The stylus has a constrained writing height. Its measured motion is sampled at physics rate; a 4 mm black ribbon represents the resulting ink. The 12 mm teal reference lane is separate geometry and never counts as executed ink. Internal stage rails are visual and enclosed behind a physical colliding, optically ideal transparent lid (refraction omitted); the controller interacts with the exposed knobs. Fluid ink, deformable paper and a detailed cable transmission are not simulated.

Scoring uses only the drawn geometry and final clearance/stability, as specified in `task.json`: 0.5 for the first segment, 0.75 for both segments and correct endpoint, and 1.0 after withdrawal and a stable two-second finish. Stray or disconnected ink invalidates the geometry. Contact breaks are preserved by both renderer and scorer; missing ink is never bridged. Neither hand count nor the script phase earns credit. The tests explicitly allow a sufficiently fine one-arm staircase and reject endpoint-only shortcuts and added stray marks.

Reset restores knob and stage poses, clears ink and score history, and refreshes camera frames. Layout and seed are fixed; no task randomization. The authored instruction and full object dimensions are in `instruction.txt` and `task.json`; construction constants are in `xy_drawing/design.py` and `assets.py`.

## Evidence

Frozen runtime **b9e7103**, run `final-002`: **1.0 /1.0**, including a reset between episodes. Each episode finishes after403 actions /20.15 simulated seconds. Maximum centre-line deviations are4.671 mm and4.470 mm; both cover100% of both reference segments, have no ink-contact breaks, and finish clear/stable. All11 scorer tests pass. Source snapshots and the exact CLI instruction match the committed files byte for byte; the saved diff is empty.

The pair took approximately122 seconds of Kit-reported runtime on one RTX5090, including simulator startup and excluding GPU queue waiting. Both three-camera videos were inspected at grasp, simultaneous turns, the second slope and release. Actual SDK-encoded320x176 JPEGs were inspected, including the cleared trace after reset. The left wrist camera also sees part of the trace during the first slope; overhead lane-edge judgments remain resolution-limited.

Media handed off separately: `demo.mp4`, `demo-e1.mp4`, `setup-top.png`; `render_video.py RUN_DIRECTORY [EPISODE]` regenerates them from raw frames. `provenance.json` accompanies the local run, with runtime commit, module/instruction SHA-256, package versions and hardware. Dream-1 completed one trial with score **0.0**, using6000 actions/300 simulated seconds (590.9 seconds wall time). It approached the controls without effective knob turning or drawing; the process completed normally with no evaluation error. The submitted media include this complete model run, the1.0 scripted demonstration and setup image.

Matching pre-cover diagnostic comparisons:

| Controller | Elapsed simulated time | Maximum centre-line deviation | Coverage within6 mm |
| --- | ---: | ---: | ---: |
| Two arms together |23.8 s|4.77 mm|100% of both segments|
| One arm, coarse alternating turns |70.15 s|14.92 mm|61% /61%|
| One arm, fine alternating turns |242.3 s|7.81 mm|100% of both segments|

The fine baseline completes the route but crosses the6 mm error threshold locally. A more accurate one-arm controller may succeed; this is evidence of useful simultaneous quality/time gain, not universal two-arm necessity. The final protective cover leaves the controls unchanged; sampled CPU wrist meshes clear it by at least10.16 mm, and the carriage clears its underside by9 mm. The final covered two-episode evidence above confirms physical execution and readable output without changing knob dynamics or the scoring tolerances. These comparisons fit within the300 s task budget.

The setup specifies intended construction materials: metal frame/posts/case and die-cast knobs, a coated drawing board, a clear polycarbonate lid, an aluminium stage and a steel stylus. The simulation uses authored rigid-body masses/friction.

The geometry, reference pattern and instruction are original. Included code is MIT licensed.

## Dream-1 evaluation

With the Dream policy plugin and local credentials configured in the organizer
environment, run one episode with the same instruction and scorer:

```bash
inspect-robots run --embodiment yam_xy_drawing --policy dream \
  -E scene=station -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "$(cat instruction.txt)" --scorer task_score \
  --max-steps 6000 --epochs 1 --seed 0 --store-frames --log-dir eval-dream
```
