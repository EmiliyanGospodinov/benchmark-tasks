# Construct a retrieval tool, then recover an inaccessible cartridge

Build a working reach-extension tool from a loose shaft and hook head, then use it to recover a cartridge from a covered service channel. A spring-return release control must be held open while the head is inserted and quarter-turned; its returning gate prevents reverse rotation, while bayonet lugs retain the head axially. The completed tool must actually hook and pull the cartridge, then be returned intact after placing the cartridge in its tray.

**Scripted demonstration: 1.0 in both episodes, including a reset. Dream-1: 0.0 in one 1,500-action evaluation.**

## Mechanism and manual

The blue squeeze control is guided laterally on the loose shaft. One gripper pinches it against the fixed handle, opening the gate while supporting the tool. The other gripper seats and turns the head. Releasing the control returns the gate through a window in the head's bell. The gripper then takes the fixed rear handle for tool use, leaving the release shut. Neither shaft nor head is fixed to the table; no joint attaches them.

The single-line text manual in `instruction.txt` and `task.json` explains the release, keyed alignment, quarter turn, return and regrasp, tooth engagement, retrieval and final checks. The demonstration reads this same instruction file; its actions are privileged, as described below. This is text-manual delivery, not a visual-manual evaluation.

Manually held-open coupling sleeves provide an engineering precedent ([Parker Sleevmatic](https://www.parker.com/content/dam/Parker-com/Literature/Literature-Files/pneumatic/Literature/Telepneumatic/MRO/MRO_Quick_Coupling.pdf)). This task uses an original squeeze-control linkage, not a replica of that product. The manufacturer reference does not validate these dimensions, loads or robot grasps.

## Setup

Coordinates: x forward and y left from the midpoint between arm bases; heights above the table. One fixed layout, no jitter. The starting screenshot and `task.json` give the inventory and positions.

| Part | Nominal geometry and mass | Start |
|---|---|---|
| Shaft | Rubber-sleeved metal handle 8 x 2.4 x 3.4 cm, rod 16 x 1.6 x 1.2 cm, 150 g; vertical 1 cm pin with two lugs spanning 2.6 cm | Pin at (30,0) cm, shaft along +x |
| Hook head | 3.6 cm keyed plate, 1.6 cm square hole, 0.8 cm slots, closed bell and stops, 2.2 x 2.2 x 2 cm neck, 6 cm hook and 1 cm downward tooth; 60 g | Stand at (30,15) cm, hook toward -y |
| Release linkage | Metal links and blue polymer grip/gate, 27.1 x 6.8 x 3.4 cm overall, 50 g; 2 cm lateral travel | Returned stop on shaft |
| Head stand | Fixed grey plastic, 2.6 x 2.6 x 1.2 cm | (30,15) cm |
| Service hood | Fixed rigid plastic, 5 x 10 cm opening, 19 cm depth, closed sides/roof/rear | Mouth at x=40 cm, y=0 |
| Rear mounting block | Fixed, 3 x 12 x 2 cm | (62,0) cm |
| Cartridge | Blue plastic, 6 x 4 cm, 120 g; 1.8 cm front lip, 2 x 2.8 cm pocket, 3.6 cm rear grip body | Front at (52,0) cm |
| Tray | Fixed green plastic, 10 x 8 cm inside, 1.2 cm rim and 0.3 cm floor | Centre (24,21) cm |

The hood excludes the gripper's palm/camera housing; the cartridge lies 12 cm inside, beyond fingertip and bare-head reach. The shaft alone lacks the tooth needed to pull it. The release linkage is wide near the handle, which stays outside the hood; its forward section fits the original opening.

## Scoring

`retrieval_tool/scoring.py` reports `info["score"]` for `--scorer task_score`.

| Score | Condition |
|---|---|
| 0.2 | Head seated within 2 mm axially and 3.5 mm radially, axes within 10 degrees, relative yaw -9 to +15 degrees, and release returned within 2 mm; its guide displacement/orientation remain within 2 mm/5 degrees |
| 0.4 | At least 10 cm net cartridge travel toward the mouth while the locked tooth occupies the pocket, and every bounding-box corner outside the hood |
| 0.6 | Cartridge fully in tray: all bottom corners within 1 mm of its interior footprint, 1–10 mm above table, upright within 15 degrees, speed below 2 cm/s |
| 0.8 | Intact tool also stowed: pin within 3 cm of start, shaft height within 6 mm, tilt below 10 degrees and speed below 2 cm/s |
| 1.0 | Final conditions persist 2 s with both grippers at least 80% open and TCPs at least 7 cm from shaft/head/cartridge origins |

The untouched start scores zero. Backward cartridge motion removes unfinished pull progress; repeated oscillations cannot accumulate extraction credit. Movement without the locked hook earns no extraction credit. Invalid numeric state scores zero for that step. Partial tiers record completed progress; full success requires the final conditions together.

CPU checks: `python -m pytest -q tests` (21 tests). Tests cover physical pose bounds, extraction without a hook, oscillation banking, wrong tray placement, invalid states, control return and broken/misaligned guide states.

## Run

Install the pinned guide adapter and this task-local plugin from the repository root:

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
uv pip install --no-deps -e tasks/modular-retrieval-tool
cd tasks/modular-retrieval-tool
inspect-robots run --embodiment yam_retrieval_tool --policy retrieval_tool_scripted \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(cat instruction.txt)" --scorer task_score \
  --max-steps 1500 --epochs 2 --seed 0 --store-frames --log-dir runs/tool-demo
```

Validated installed package metadata: Isaac Sim 6.1.0.0, Isaac Lab 17.0.2, Isaac Lab Arena 0.3.0, inspect-robots 0.60.0 and PyTorch 2.11.0+cu128. Use the pinned adapter bootstrap above; these metadata versions differ from the guide branch names. The plugin retains the supplied station, YAM robot, three cameras and 14-D joint action/observation contract. Limit: 1,500 actions, 75 simulated seconds at 20 Hz. Fixed layout; seed 0 is recorded. Wall time depends on renderer/GPU load.

To evaluate Dream-1 with a configured account, use the same command with
`--policy dream --epochs 1` and a separate output directory. The published
1,500-action limit and instruction file are identical for both policies.

`-E telemetry=path.jsonl` records part poses, release travel, coupling state, grippers and scorer state. Diagnostic policy modes: `release` (support/open/lift/return), `assembly`, `onearm` (release untouched), `early` (open and release before insertion), `direct`, `bare` and `headonly`. These are diagnostics, not full-task demonstrations.

**The scripted demonstration is privileged.** It reads part poses for pickup, insertion alignment, in-grip offsets, tool steering and tooth/pocket alignment. Steering includes bounded pose-feedback corrections (8 degrees of orientation and 12 mm lateral wrist offset) to compensate for in-grip drift. Commands are ordinary robot joint targets; the policy never writes object poses. The release gate is driven only by robot contact and its passive spring, not by script phase or score.

## Simulation and evidence

Native rigid-body contacts retain the bayonet and engage the hook. A native shaft-fixed prismatic guide supports the rigid control linkage, with a 100 N/m spring, 3 mm preload, 2 N s/m damping and 20 mm travel. Shaft/linkage collision is disabled for the unmodelled guide rails; head/gate and robot contacts remain active. No shaft/head weld, pose snapping, hand-count condition or model-dependent force is used. Dimensions, masses, inertia and friction are nominal prototype values, not calibrated hardware measurements. Head, shaft and linkage use 128 position/8 velocity solver iterations; other parts use 16/4.

A support diagnostic retracts the control 20 mm and lifts the loose shaft about 21 mm with the same gripper, then demonstrates spring return. Held assembly seats and turns the head, and the gate returns after release. With the same mechanism source, untouched-release insertion remains 9.6 mm too high and 17.4 mm off-axis (0.0), and opening then releasing the control before insertion also fails (9.6 mm high, 18.5 mm off-axis, 0.0). These comparisons support the cooperative procedure; they do not prove every possible one-arm strategy impossible. The full demonstration and reset repeat each score 1.0 at 48.7 simulated seconds (975 rendered frames each), with 182.3 mm and 182.3 mm of measured hooked cartridge travel. Both finish with the retained tool stowed, cartridge fully in its tray, and both grippers open and clear for two seconds. Runtime source: `0695a34fbbbc989fde650b46a2fa4d9bc05066ce`; later documentation changes do not alter that source. One Dream-1 evaluation using the same mechanism and text manual completed 1,500 actions (75 simulated seconds) with score 0.0 and no runtime error; it did not assemble the tool or retrieve the cartridge. The full-score scripted demonstration accompanies that result.
