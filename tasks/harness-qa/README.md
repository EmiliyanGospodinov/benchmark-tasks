# Two-probe harness pin-map verification

Measure AC, AD, BC and BD on an isolated two-conductor harness, interpret the four-bit connectivity map, record the verdict with a physical tag, and stow both probes. This is a restricted pin-map QA task, not a full insulation/short-circuit qualification or harness repair.

## Task and manual

The single-line numbered manual in `instruction.txt` is identical to `task.json`. The default concealed graph has A-D and B-C wires. Each modelled connected wire has nominal 1-ohm resistance; unconnected endpoints read OL. The supported alternatives are straight A-C+B-D, A-C only, and B-D only. Four measurements map 1001 to PASS, 0110 to CROSSED, and 1000/0001 to OPEN. Unexpected patterns require rechecking, not automatic OPEN classification.

One gripper holds each loose probe against its pad. Native contact compresses each spring tip. A reading counts only after both independent contact predicates remain valid for 0.5 s. The meter shows -- for missing contact, OL for a valid unconnected pair, and1 for continuity. The robot then places the blue result tag on P/X/O and leaves both probes stowed and released.

The upright panel has no retaining sockets, ledges or probe clips. The probes are shorter than the pad heights. Pad columns are 18 cm apart. In the release diagnostic, opening the left hand drops its compression to zero and breaks acquisition while the other tip remains compressed. This supports the demonstrated two-probe procedure, not impossibility of all alternative one-arm methods. Ordinary environment contacts remain allowed and the scorer does not count hands.

## Geometry and contact model

The inventory, coordinates, dimensions and masses are specified in `task.json`. The station, robot, cameras, action space and observation interface are unchanged. Handles are rectangular polymer bodies with metal spring tips. The panel, result areas and meter are fixed rigid fixtures. All geometry is original primitives; no commercial asset is copied.

Each tip is on a native prismatic guide with 200 N/m stiffness, 1 mm preload, 1 N s/m damping and 8 mm travel. The tip/handle guide's internal collisions are omitted; tip/panel and robot contacts remain active. Compression of 1–8.5 mm, lateral guide alignment<1mm, tip-axis alignment within 10 degrees, distal face within 2 mm of the pad plane and tip centre within 11 mm of the pad centre establish contact. Compression alone away from the panel does not count. Two valid contacts are required even for OL, so disconnected probes cannot fabricate an open-wire result.

Connectivity is calculated from the explicit declared wire graph. PhysX supplies mechanical contact, not charge transport. Lead geometry/dynamics are omitted and there is no cable-management claim. Meter glyphs are kinematic visual bodies moved behind/above an opaque face according to the live reading; they do not apply task forces. Material properties are nominal training-fixture assumptions, not calibrated measurements.

## Run

Install the guide adapter and task-local plugin:

```bash
mkdir -p .deps
git clone https://github.com/kairosiann/yam-isaac-lab-arena.git .deps/yam-isaac-lab-arena
git -C .deps/yam-isaac-lab-arena checkout 0059d02a1a91bc6834c1d0846022891df0c78730
ARENA_DIR="$PWD/.deps/IsaacLab-Arena" .deps/yam-isaac-lab-arena/setup.sh
source .deps/IsaacLab-Arena/yam.env
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
uv pip install --no-deps -e tasks/harness-qa
cd tasks/harness-qa
inspect-robots run --embodiment yam_harness_qa --policy harness_qa_scripted \
  -P mode=full -E wiring=crossed -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(cat instruction.txt)" --scorer task_score --max-steps 1100 \
  --epochs 2 --seed 0 --store-frames --log-dir runs/harness-demo
```

Validated installed metadata: Isaac Sim6.1.0.0, Isaac Lab17.0.2, Isaac Lab Arena0.3.0, inspect-robots0.60.0, PyTorch2.11.0+cu128 and yam-arena0.1.0. These are the installed runtime versions used with the pinned adapter.

The limit is 55 simulated seconds at 20 Hz; wall time depends on renderer/GPU load. `-E wiring=straight`, `open_ac` and `open_bd` select declared alternative graphs. `-E telemetry=path.jsonl` records contacts, compression, readings, part poses and scorer state.

Diagnostic modes `probe` (AC/AD) and `stage` (also release one held probe) are not full-task demonstrations. The privileged scripted policy reads simulator poses for grasping and reads the instrument's settled measurement history to choose the verdict. It never selects the verdict from the hidden wiring setting. Dream receives ordinary camera frames and joint state, without numeric meter values.

## Scoring and evidence

Four distinct valid settled measurements earn 0.15 each; the correctly positioned tag adds 0.2. Full success also requires both probes stowed, both grippers open and clear, and two seconds of stability. A wrong tag cannot earn verdict credit. CPU tests cover transient/missing contacts, valid OL, wrong/missing maps, wrong tag, held/unstowed probes, invalid values and reset.

The first crossed-harness diagnostic records AC=OL and AD=1. Repeating the same contacts with straight wiring changes them to AC=1 and AD=OL. Releasing one probe breaks acquisition. Actual Dream SDK `jpeg()` output (320x176, quality 85) is inspected during simultaneous contact; no model calls are involved in this encoding check. The final crossed-harness demonstration scores **1.0 in both episodes**, including the reset repeat, at **40.6 simulated seconds** each. Both acquire 0110, place the tag on X and finish with probes stowed and grippers open/clear for two seconds. Frozen runtime source is `11ac481`; subsequent result-documentation changes do not alter runtime or the instruction. All 18 CPU tests pass, including reversed probe identities. The straight-wiring diagnostic establishes changed AC/AD readings, not a complete alternate-verdict demonstration. Dream-1 completed one trial with score **0.0**, using1100 actions/55 simulated seconds (248.5 seconds wall time), with no evaluation error. It did not acquire the probes or record a valid measurement; no verdict was reached. Submitted media include this full model run, the1.0 scripted demonstration and setup image.

## Dream-1 evaluation

With the Dream plugin and local credentials configured in the organizer runtime:

```bash
inspect-robots run --embodiment yam_harness_qa --policy dream \
  -E scene=station -E wiring=crossed -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "$(cat instruction.txt)" --scorer task_score \
  --max-steps 1100 --epochs 1 --seed 0 --store-frames --log-dir runs/harness-dream
```
