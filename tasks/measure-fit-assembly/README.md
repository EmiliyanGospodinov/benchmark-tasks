# Measure shaft end play and fit a split spacer

Use a contact indicator and a textual tolerance table to select a split spacer,
install it, then verify clearance and shaft rotation. The right arm supports the
indicator against the housing datum while the left moves the shaft between stops.
The fitting step uses an open-topped channel; the fixed end stop stays installed.

**Tolerances.** The indicator display resolves 0.01 mm. The three selective
split spacers are 8.80, 9.00 and 9.20 mm (0.2 mm steps) and the accepted final
end play is 2.90–3.10 mm, so exactly one spacer fits the measured 12.00 mm travel
and each neighbour misses the window by 0.10 mm. Real end-play shimming works in
hundredths to tenths of a millimetre; the steps here are coarser than typical
real shims (often 0.05–0.1 mm) because the robot-held indicator repeats to about
0.04 mm in simulation, and the 3 mm nominal end play is far larger than a real
small gearbox's so the robot can insert the spacer without a lead-in. The
initial travel is fixed, so the 9.00 mm spacer is the correct one after every reset.

**Scripted demonstration (revised tolerances):** two complete episodes,
including a reset, each score **1.0** in 87.7 simulated seconds. The held indicator
measured 11.98 mm initial travel in both episodes (true travel 12.00 mm), selecting
the 9.00 mm spacer; remeasured end play was 2.99 mm against 3.00 mm physical.
Dream-1 on the revised task scored 0.0 (seed 0, 2,000 steps).

The board instruction is a compact summary. `instruction.txt` contains the full
numbered manual; the run commands below pass that exact file.

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
standard cameras. The display (00.00–40.00, 0.01 mm steps) has a 20 mm offset: subtracting
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

Shared GPU users must follow their workspace's reservation/priority launcher.

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
| Blue / yellow / red spacers | Polymer selective split spacers, 8.80 / 9.00 / 9.20 mm axial thickness; 60 mm width; 95.5 mm height including 26×28×25 mm grasp tabs | 0.035 kg each | x=0.14 / 0.26 / 0.38, y=0.18, z=0.7715 |
| Parts mat | 320×120×4 mm rigid mat | supplied Box | (0.26, 0.18, 0.7495) |
| Remote readout | 85×150×16 mm fixed housing with non-colliding display segments | 0.2 kg, fixed | (0.41, -0.25, 0.7555) |

The split spacer changes the actual colliding axial stop. No object identity changes
joint limits or triggers a clamp. The shaft's native guide permits axial motion and
rotation while approximating bearing support. Display segments are visual only and
are updated from probe displacement. The fixture and friction are authored training
models, not calibrated industrial hardware.

## Scoring and checks

Initial score is zero. Referenced gauging requires probe alignment/contact geometry,
both reference pads within 0.5 mm of the housing datum, the shaft within 0.05 mm of
each physical stop, and readings steady within 0.02 mm for 0.25 s. The initial travel
must be within 0.10 mm of 12.00 mm. The scorer checks installed geometry, final
2.90–3.10 mm clearance and its remeasurement (agreeing within 0.05 mm), actual angular sweep of at least 60°, and
two seconds of released stability. Releasing means open grippers at least 6 cm from
the relevant grasp regions; the indicator must be returned to table height.

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

CPU tests cover false measurements, invalid values, wrong clearance, visual readout
encoding, and the need for post-fitting verification, rotation and release. A test
pass is not evidence that robot manipulation succeeds. See run evidence for that.

Selection margin: in the recorded held-indicator episodes, steady readings at the
stops matched true shaft travel within about 0.04 mm, against the 0.10 mm margin to
each selection boundary. The adjacent spacers would leave 3.20 or 2.80 mm, outside
the accepted window. The earlier loaded-geometry check (6/9/14 mm spacers) and the
one-arm comparison below were made on the previous, looser version and were not
repeated.
Omitting the instrument-holding arm from the same measuring procedure produced
no valid referenced measurement and scored 0.0 in a separate 27.5-second trial.
This comparison tests that procedure, not every possible one-arm strategy.

## Provenance

Assets are original compound geometry; no downloaded mechanical models. Motion
and USD helpers reuse earlier task scaffolding under the included MIT license.
No sibling task is imported at runtime. Generated USD and run outputs are excluded
from version control.
