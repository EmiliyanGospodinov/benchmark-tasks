# Extract a fitted collar with a prepared two-jaw screw puller

The operator starts with both hooks engaged behind the fitted collar. The robot
counterholds the loose tool, turns its screw, recovers the removed collar in a
tray, and parks the tool. Autonomous puller placement is outside this task.
The numbered text manual in `instruction.txt` specifies the operation, turning
direction, recovery destination and final released check. Pass this exact text
as the policy instruction; no image manual is implied.

## Install and run

From the repository root, use the pinned organizer adapter bootstrap:

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
uv pip install --no-deps -e tasks/mechanical-puller
cd tasks/mechanical-puller
python -m pytest -q tests
inspect-robots run --embodiment yam_mechanical_puller \
  --policy mechanical_puller_scripted -P stage=complete \
  -E scene_kind=prepared -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(cat instruction.txt)" --scorer task_score \
  --max-steps 2000 --epochs 2 --seed 0 --store-frames --log-dir runs/demo
```

Validated installed package metadata: Isaac Sim 6.1.0.0, Isaac Lab 17.0.2,
Isaac Lab Arena 0.3.0, inspect-robots 0.60.0 and PyTorch 2.11.0+cu128. These
metadata versions differ from the guide branch labels; use the pinned bootstrap.

For the model evaluation replace `--policy mechanical_puller_scripted -P stage=complete`
with `--policy dream` and use one episode. Model evaluation incurs API costs.
The embodiment keeps the organizer's station, YAM robot, three RGB cameras and
14-D absolute joint-target action contract. The script reads privileged part
poses for control; the model receives the ordinary observation contract.
The task folder installs a custom embodiment plugin and generates original USD
geometry locally. It is not an unmodified task-loader-only scene.

Run limit: 100 simulated seconds. The verified script takes 65.85 simulated
seconds (1,317 actions). Two episodes including startup took about 342 seconds
wall time on the local GPU; rendering speed depends on hardware/load. Reset restores all loose bodies and native
joints, screw angle, stick-slip anchors and score history. Seed 0; object jitter
is disabled so the operator-prepared engagement is reproducible.

## Inventory and starting arrangement

Coordinates are metres, x forward and y left in the standard station; table
height is 0.7475 m. All tool parts are original rigid geometry, with nominal metal
flange, shaft, collar, body, ram, jaws and rest posts, a die-cast metal knob,
a black-coated side handle and a rigid green plastic recovery tray. These are
construction/material descriptions; friction and masses are authored separately. No commercial part or measured hardware is
claimed. `design.py` is the numeric inventory and `assets.py` specifies geometry,
collision material and mass properties.

| Item | Dimensions / mass | Initial placement |
| --- | --- | --- |
| Fixed flange and shaft | 100 x 100 x 10 mm flange; shaft radius 6 mm, shoulder 35 mm and centre-cup rim 38 mm above table | x 280, y 40 mm |
| Orange fitted collar | bore 13 mm; hub diameter 22 mm x 4 mm; rim diameter 44 mm x 10 mm; 30 g | hub bottom 10 mm above table |
| Free puller body | 100 x 16 x 12 mm crossbar; 30 mm boss; 110 mm side handle; 120 g | centre above shaft at 77.5 mm; upright |
| Orange screw knob | 36 mm diameter x 20 mm; 120 g | 48 mm above body centre |
| Guided ram | 8 mm tip pad, 5 mm point; 20 g | zero screw extension |
| Two sliding jaws | 6 x 14 mm arms with 3 mm hook lips; 20 g each | axes +/-25 mm; hooks pre-engaged in under-rim gap |
| Fixed tray | 110 x 90 mm interior, 4 mm floor, 14 mm rim height | centre x 280, y 180 mm |
| Fixed two-post rest | posts 6 x 16 x 89 mm, separated 90 mm centre-to-centre | centre x 280, y 180 mm; crossbar centre height 95 mm |

## Mechanics and scoring

The screw is a declared constitutive approximation: a revolute knob and native
prismatic ram are coupled with a 12 mm/turn target, 20 kN/m stiffness, 50 Ns/m
damping and a 60 N drive cap. The target is refreshed at control steps; PhysX
integrates the drive at physics steps. Elastic drive load gives equal/opposite
reaction torque on knob and body. This is not resolved thread contact.
The fitted collar uses an 8 N axial stick-slip model and a native shaft guide;
the guide disables only when its base has risen 27 mm, clearing the shaft.
Jaw slides have 3 N authored stick-slip resistance. No grasp welds, extraction
teleports or target-conditioned forces are used. Material friction is authored,
not calibrated: collar/ram 0.6/0.5 static/dynamic, body 0.8/0.7, jaws 0.9/0.8,
knob 1.0/0.9.

The measured-state scorer starts at zero. A retained two-jaw extraction with
at least two clockwise turns earns 0.25; collar fully in the tray earns 0.5;
released stable parking earns 1.0. Stability requires 2 seconds within 2 mm and
2 degrees for every tool part and collar. Tool parking tolerances are 4 mm XY,
3 mm height and 5 degrees orientation; the complete collar bounding box must
fit inside the tray and rest near its floor. Both grippers must be >=80% open
and TCPs >=80 mm from all loose part origins and the side-handle grip point. Invalid numeric states or separated
tool geometry score zero. `scoring.py` defines all checks; there is no hand-count
or scripted-phase scoring condition.

Matched prepared-scene diagnostics at a07e283 found 29.4 mm extraction with
counterholding versus 12.7 mm without it over the same eight turning strokes.
The unsupported tool rotated substantially. This supports counterholding in
this procedure; it does not prove that all longer one-arm strategies fail.

## Evidence

Frozen runtime **ec667d6**, run `final-003`: full score **1.0 / 1.0**, including
reset between episodes, each after **1,317 actions / 65.85 seconds**. All 14 scorer
tests pass. Replaying all 2,634 saved state samples reproduces the runtime scores
exactly; runtime and packaged scorer SHA-256 are
`6d8c0a934e86e3c1936cb064289b5bff4526e2c1b4b648b800b775de5244c0b9`.

Both three-camera videos were inspected at extraction, two-hand transport,
physical jaw spreading and released parking. Run media: `demo.mp4`, `demo-e1.mp4`
and `setup-top.png`; these are handed off separately, not embedded in source.
`render_video.py RUN_DIRECTORY [EPISODE]` regenerates them from saved raw frames.
The final Dream result and uploaded media identifier are recorded in task.json.

The prepared-scene counterhold comparison predates the tray-spanning cradle;
it is evidence about that diagnostic setup, not an exhaustive one-arm exclusion
for the final scene. The complete script uses both arms for extraction and
transport. A side-handle-only transport attempt dropped the tool; the successful
script supports the knob with the other gripper while carrying.

One Dream-1 evaluation of the same scene and instruction completed all 2,000 actions (100 simulated seconds) with score 0.0 and no runtime errors. It did not extract the collar. The full-score scripted demonstration accompanies this result.

## Attribution

Original task geometry and scripts. Station-wrapper and manipulation helpers
were adapted from earlier team task prototypes, using the organizer's YAM
adapter. No external mesh assets are bundled. Original task code and generated geometry are MIT-licensed; see `LICENSE`.
Upstream runtime dependencies retain their own licenses.
