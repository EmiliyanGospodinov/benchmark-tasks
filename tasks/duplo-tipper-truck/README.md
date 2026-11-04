# Attach a DUPLO tipper bed and dump a load with it

Attach the loose bed to the preassembled truck, load the cube, and tip it through
the open rear into the tray. In the demonstrated procedure, one gripper stabilizes
the wheeled truck while the other attaches and operates the bed. Finish with the
hinge retained, the truck upright and both hands clear.

The four numbered steps in `manual-bed.txt` equal `task.json`'s `instruction`.
They specify orientation, attachment and the functional check. The scripted demo
uses privileged part poses and ordinary joint targets; it is not evidence of
learned visual or language understanding.

## Setup

Use the unchanged calibrated YAM `station` (table, arms and three policy cameras).
Tabletop z=747.5 mm; x forward, y left from the midpoint of the arm bases. All
initial part-frame rotations are identity. Positions below are in millimetres;
`scene_inventory.json` records exact nominal frames and masses.

| Item | Nominal geometry and mass | Initial configuration |
| --- | --- | --- |
| Chassis, LDraw 15314 | 64 x 128 x 19.2 mm; 59.9 g | Deck-frame origin (300,0,775.7), front toward -y |
| Two wheel/axle assemblies | Four tyres, diameter 32 mm, width 12 mm; 7.5 g per assembly | Axle origins (300,-32,763.5), (300,32,763.5), rotation about x |
| Blue brick, LDraw 3437 | 32 x 32 x 19.2 mm; 12.4 g | Joined at body centre (300,-40,785.3) |
| Yellow cab, original approximation | 64 x 136 x 86.4 mm; 59.25 g | Joined at chassis origin, cabin toward -y |
| Black hinge plate, LDraw 65084 | Plate body 32 x 56 x 9.6 mm plus extended ears/pin; 8.32 g | Joined at (300,9,804.5); hinge axis (300,57,806.5) |
| Blue bed, LDraw 14094 | Nominal envelope 64 x 72 x 36.4 mm; 26.23 g | Loose hinge-frame origin (200,220,748), open rear toward +y |
| Orange cube | 20 mm side; 5 g | Centre (120,170,758) |
| Grey tray | Inside 100 x 90 mm; 4 mm floor; 4 mm thick walls 12 mm above floor | Fixed/taped, centre (300,114), floor top z=751.5 |

Nominal materials: plastic truck bodies and tray, rubber tyres and wooden cube.
The chassis, brick, cab and hinge plate start connected; bed and payload are loose.
One fixed layout, seed 0, object jitter 0. Reset restores poses, detaches the bed
and clears progress/motion history. Nothing is consumed. Collider geometry is
fully specified by `duplo_tipper/assets.py` and `duplo_tipper/design.py`.

## Scoring

`duplo_tipper/scoring.py` supplies `info["score"]` to `--scorer task_score`.
The highest currently satisfied tier is returned:

| Score | Condition |
| --- | --- |
| 0 | Bed not correctly attached to the intact truck, including the untouched start |
| 0.5 | Bed correctly attached to the intact preassembled truck |
| 0.75 | Bed remains attached and a valid load has been recorded |
| 1.0 | Valid rear unloading into the tray, then released stable finish for 1 s; trial ends |

No credit for preassembled components. `task.json` states the loading, unloading,
tray containment, hand-clearance and rest thresholds. Additional geometric checks:

- Bed hinge origin within 1.5 mm of pin, residual hinge-axis rotation <=5 degrees,
  bed tip from -3 to 68 degrees. Base connections retained within 1.5 mm laterally,
  1 mm vertically and 4 degrees of their specified relative poses.
- Loaded cube centre in bed frame: |x|<=20 mm, -52<=y<=-4 mm, z=17.6 +/-4 mm.
  Bed tip <10 degrees, TCPs/pad tips >=35 mm from cube, cube speed <10 mm/s and
  <0.3 rad/s for 0.5 s.
- Valid rear exit after loading: y>4 mm and |x|<20 mm in the bed frame, bed tip
  >=20 degrees, bed still attached, truck tilt <=12 degrees, hand gap >=35 mm.
- Before exit, tipping the truck past 12 degrees, losing the bed connection, or
  removing the cube by hand invalidates the load and requires reloading. Direct
  placement into the tray cannot earn full score.
- Final: all cube corners within tray walls and bottom within 3 mm of floor;
  truck tilt <=5 degrees, chassis height within 4 mm of 775.7 mm, connections
  intact; grippers >90% open, TCPs/pad tips >=45 mm from each truck-part origin.
  Each truck part and cube must remain below 10 mm/s and 0.3 rad/s for 1 s.

Speeds use pose differences at 20 Hz. The bed may remain open at its stop.
Arm identities and simultaneous motion are not scored.

## Install and run

Linux and a supported NVIDIA RTX GPU are required. From the repository root:

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
uv pip install --no-deps -e tasks/duplo-tipper-truck
cd tasks/duplo-tipper-truck
inspect-robots run --embodiment yam_duplo_tipper --policy tipper_scripted \
  -E preassembled=brick,cab,hinge -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 -P stage=bed \
  --instruction "$(cat manual-bed.txt)" --scorer task_score \
  --max-steps 2400 --epochs 2 --seed 0 --store-frames --log-dir runs/tipper-demo
```

This task-local plugin adds parts, joints and scoring to Isaac Sim 6.1 / Isaac Lab
3.0 / Isaac Lab-Arena 0.3.1. Robot, cameras and the standard 14-D joint contract
are unchanged. Loading `task.py` alone through the unextended `yam_arena` does
not construct this mechanism. Editable installation is required; keep the task
folder writable for USD generation under `assets/generated/`. All part geometry
is bundled; the task does not download assets at run time.

CPU checks: `python -m pytest tests` (install pytest if needed). Optional telemetry:
`-E telemetry=runs/telemetry.jsonl`. For the no-counterhold diagnostic add
`-P counterhold=false --epochs 1`; the right arm stays at its ready pose.
Run limit: 2,400 steps, 120 simulated seconds at 20 Hz.

For Dream, use the same scene, instruction and limit, replace the policy with
`--policy dream`, omit `-P stage=bed`, and use `--epochs 1` and a separate log
directory. Configure credentials locally through the Dream plugin. Dream receives
RGB camera views and joint state, not privileged part poses.

## Simulation model

Rigid bodies with LDraw visuals and simplified colliders. The bed clip uses a
retention-joint approximation: alignment within 1.5 mm/5 degrees, relative
translational speed <30 mm/s and measured contact force >=2 N for 0.1 s enable a
hinge at the measured pose. No elastic clip deformation. Hinge travel is 0 to
65 degrees, with provisional break limits 15 N and 0.3 N m. No target-pose
teleport is commanded during an episode; reset restores poses between episodes.

Bearings use uncalibrated stick-slip spring/friction capped at 0.0008 N m per
axle; tyres use round capsule contacts. Physics runs at 240 Hz, control at 20 Hz.
Engagement and rest use pose changes: earlier diagnostics found disagreement
between reported body velocities and measured motion of the constrained hinge.

## Results and media

The scripted demonstration scores **1.0 in both episodes**, including the reset
repeat (733 steps / 36.65 simulated seconds each). The truck moves at most
2.5 mm from its starting pose; all connections remain intact and the cube ends
in the tray after valid rear unloading. Both episodes together took about 293 s
wall time on an RTX 5090, including startup/reset.

With the holding arm disabled, the same scripted sequence scores **0.5**: the
bed attaches and the cube loads, but tipping drags the truck about 183 mm and
tips the chassis beyond the allowed angle, invalidating unloading. This supports
the demonstrated stabilizing role; it is one comparison, not a proof that every
one-arm strategy fails. The fixed tray can also serve as a potential brace;
no optimized one-arm bracing policy has been evaluated.

Dream-1 (`dreamlabs/dream-1-beta`) scores **0.0** in one seed-0 episode, reaching
the 2,400-step limit with no errors. It does not attach the bed. The evaluation
uses the exact same instruction and scene as the demonstrations; it makes 50
requests with 50 attempts. Wall time was about 342 s on an RTX 5090, including
simulator startup and inference.

Submission media contain the complete Dream episode (120.05 s), one complete
full-score demo (36.7 s), and the original overhead setup frame. The videos show
left wrist, right wrist and overhead views side by side. Media ID is in task.json.

## License and attribution

Original code and original procedural geometry: MIT (`LICENSE`). LDraw source
geometry and derived `.npz` meshes: CC BY 4.0 (`assets/ldraw/CAlicense4.txt`).
`assets/ldraw-sources.json` records every source URL, author/history attribution
and hash. Original part files and headers are bundled byte-for-byte in
`assets/ldraw-sources.tar.gz`; the four derived meshes are directly available in
`assets/*.npz`. To regenerate meshes, first unpack the source archive with
`tar -xzf assets/ldraw-sources.tar.gz -C assets/ldraw`. The importer converts units/axes to
metres/Z-up, triangulates faces and omits edge lines; rendering uses uniform task
colours. Colliders are authored separately. External dependencies retain their
licenses. DUPLO and LEGO are trademarks of the LEGO Group; this independent task
is not endorsed by them.
