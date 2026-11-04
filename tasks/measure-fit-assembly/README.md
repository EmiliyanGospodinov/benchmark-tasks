# Measure shaft end play and fit a split spacer

Use a contact indicator and a textual tolerance table to select a split spacer,
install it, then verify clearance and shaft rotation. The right arm supports the
indicator against the housing datum while the left moves the shaft between stops.
The fitting step uses an open-topped channel; the fixed end stop stays installed.

**Verified scripted demonstration:** two complete episodes, including a reset,
each score **1.0** in 87.7 simulated seconds from code commit `6842a9c`.
Initial gauging, physical spacer fitting, final clearance, rotation and released
stability all pass. Dream-1 scored **0.0** in one seed-0 trial using the full
2,000-step allowance (100 simulated seconds, 203.96 seconds wall time), with
no runtime errors. It attempted manipulation but did not complete referenced
gauging. Both policies received the same full text manual.

The board instruction is a compact summary. `instruction.txt` contains the full
numbered manual used for both the recorded scripted episodes and the final Dream
evaluation; the run commands below pass that exact file.

## Install and run

Use the supplied YAM Isaac Lab Arena environment (Isaac Sim 6.1 / Isaac Lab 3.0 /
Arena 0.3.1). No shared dependency upgrades are needed:

```bash
uv pip install --no-deps -e .
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
inspect-robots run --embodiment yam_measure_fit --policy measure_fit_scripted \
  -P mode=full -E scene=station -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 -E telemetry=runs/demo/telemetry.jsonl \
  --instruction "$(cat instruction.txt)" --scorer task_score \
  --max-steps 2000 --store-frames --log-dir runs/demo
```

Unique entry points: `yam_measure_fit` and `measure_fit_scripted`. The station,
robot, three cameras, 14-D joint action and observations retain the supplied
interface. A full scripted episode takes about 90 simulated seconds (roughly
3–4 minutes wall time on the development RTX 5090, excluding queue time).
`mode=coop` runs the measurement diagnostic, `mode=nohold` omits the
instrument-holding arm, and `mode=idle` checks passive stability. Reset restores
all poses, guide states and score history. No seed jitter is used.

The scripted demonstrator reads simulator state, including measured probe travel
for spacer selection. It calibrates the grasp offset and corrects the spacer pose
during insertion; reference-pad and yaw feedback aligns the handheld indicator after pickup.
It issues ordinary robot joint targets; it does not attach
objects to grippers or move objects directly. A model receives the manual through
the text instruction and reads the physical probe's remote display through the
standard cameras. The display's absolute scale has a 20 mm offset: subtracting
the two stop readings removes that offset.

For a reset repeat, add `--epochs 2` to the command above. Export each episode's
three raw camera views side by side with:

```bash
python render-run.py runs/demo
```

The final Dream evaluation uses the same scene, cameras, scorer, action limit
and full text instruction:

```bash
inspect-robots run --embodiment yam_measure_fit --policy dream \
  -E scene=station -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 -E telemetry=runs/dream/telemetry.jsonl \
  --instruction "$(cat instruction.txt)" --scorer task_score \
  --max-steps 2000 --epochs 1 --seed 0 --store-frames --log-dir runs/dream
```

The local development helper `run-station.sh NAME full 2` additionally snapshots
source, commit/diff, instruction and telemetry. It contains a workstation-specific
environment path; the command above is the portable entry point. Shared GPU users
must follow their workspace's reservation/priority launcher.

## Physical layout

Coordinates are metres, x forward from the robot bases, y left, z upward; the table
surface is z=0.7475. Rotations start as identity except the indicator/probe,
which rest at -9 degrees pitch with the bridge supported by the housing foot.
The instrument must be lifted clear of that support for gauging. Exact geometry is in
`measure_fit/assets.py`, `builder.py`, `fitting.py` and `readout.py`.

| Object | Geometry/material representation | Mass | Initial origin |
|---|---|---:|---|
| Fixed housing | Metal base 140×120×20 mm; 64 mm OD / 20 mm bore annular stop; spacer channel | 1 kg, fixed | (0.24, 0, 0.8225) |
| Shaft | Metal 16 mm diameter, 180 mm long; 56×8 mm thrust disc; 25×35×35 mm grasp block | 0.12 kg | housing origin |
| Fixed end stop | Metal annular stop, 64 mm OD / 20 mm bore, 8 mm thick, supported on a post | 0.05 kg, fixed | housing origin; inner face at local x=20 mm |
| Indicator body | Rigid 35×35 mm handle, open probe bore, two reference bridge pads; approximately 150×60×39 mm envelope | 0.18 kg | (0.06665, 0, 0.77394) |
| Spring probe | 6 mm stem, 12 mm contact tip; native ±20 mm prismatic travel, 20 N/m spring | 0.012 kg | (0.13488, 0, 0.78475) |
| Blue / yellow / red spacers | Polymer training shims, 6 / 9 / 14 mm axial thickness; 60 mm width; 95.5 mm height including 26×28×25 mm grasp tabs | 0.035 kg each | x=0.14 / 0.26 / 0.38, y=0.18, z=0.7715 |
| Parts mat | 320×120×4 mm rigid mat | supplied Box | (0.26, 0.18, 0.7495) |
| Remote readout | 85×150×16 mm fixed housing with non-colliding display segments | 0.2 kg, fixed | (0.41, -0.25, 0.7555) |

The split spacer changes the actual colliding axial stop. No object identity changes
joint limits or triggers a clamp. The shaft's native guide permits axial motion and
rotation while approximating bearing support. Display segments are visual only and
are updated from probe displacement. The fixture and friction are authored training
models, not calibrated industrial hardware.

## Scoring and checks

Initial score is zero. Referenced gauging requires probe alignment/contact geometry,
both reference pads within 0.5 mm of the housing datum, and stable readings at both
physical stops. The scorer checks initial measurement, installed geometry, final
2–4 mm clearance and its remeasurement, actual angular sweep of at least 60°, and
two seconds of released stability. Releasing means open grippers at least 6 cm from
the relevant grasp regions; the indicator must be returned to table height.

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

CPU tests cover false measurements, invalid values, wrong clearance, visual readout
encoding, and the need for post-fitting verification, rotation and release. A test
pass is not evidence that robot manipulation succeeds. See run evidence for that.

In a separate physically loaded geometry check, the 6 mm spacer left 5.994 mm
of shaft travel, the 9 mm spacer left 2.995 mm, and the 14 mm spacer could not
seat (it remained about 34 mm above the installed position). These are mechanism
checks; the two full robot demonstrations use the fixed layout described above.
Omitting the instrument-holding arm from the same measuring procedure produced
no valid referenced measurement and scored 0.0 in a separate 27.5-second trial.
This comparison tests that procedure, not every possible one-arm strategy.

## Provenance

Assets are original compound geometry; no downloaded mechanical models. Motion
and USD helpers reuse earlier task scaffolding under the included MIT license.
No sibling task is imported at runtime. Generated USD and run outputs are excluded
from version control.
