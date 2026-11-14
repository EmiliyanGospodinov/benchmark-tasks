# Extract a fitted collar with a prepared two-jaw screw puller

The operator starts with both hooks engaged behind the fitted collar. The robot
counterholds the loose tool, turns its screw, recovers the removed collar in a
tray, and parks the tool. Autonomous puller placement is outside this task.
`instruction.txt` states only the goal and the scored constraints (screw
extraction, collar in the tray, intact tool level on its rest in its starting
orientation, both grippers open and at least 8 cm clear, everything still for two
seconds). Pass this exact text as the policy instruction.

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
14-D absolute joint-target action contract. The scripted demonstration uses the
same observation contract as the model (see "Demonstration controller").
The task folder installs a custom embodiment plugin and generates original USD
geometry locally. It is not an unmodified task-loader-only scene.

Run limit: 100 simulated seconds. The corrected nominal observation-only script takes 79.4
simulated seconds (1,588 actions) per episode. Reset restores all loose bodies and native
joints, screw angle, stick-slip anchors and score history. Seed 0; object jitter
is disabled so the operator-prepared engagement is reproducible.

## Inventory and starting arrangement

Coordinates are metres, x forward and y left in the standard station; table
height is 0.7475 m. All tool parts are original rigid geometry: metal flange, ground steel shaft,
body, ram, jaws and rest posts; an integral iglidur J polymer collar; a die-cast metal knob,
a black-coated side handle and a rigid green plastic recovery tray. These are
construction/material descriptions; friction and masses are authored separately. No commercial part or measured hardware is
claimed. `design.py` is the numeric inventory and `assets.py` specifies geometry,
collision material and mass properties.

| Item | Dimensions / mass | Initial placement |
| --- | --- | --- |
| Fixed flange and shaft | 100 x 100 x 10 mm flange; shaft radius 6 mm, shoulder 35 mm and centre-cup rim 38 mm above table | x 280, y 40 mm |
| Yellow integral iglidur J collar | lower 2 mm land bore 11.990 +/-0.002 mm; remainder bore 13 mm; hub diameter 22 mm x 4 mm; rim diameter 44 mm x 10 mm; calculated 22.21 g | hub bottom 10 mm above table |
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
The integral polymer collar has a 10 micrometre nominal diametral interference
on a 12.000 +/-0.001 mm steel shaft. Its lower 2 mm land has an 11.990 +/-0.002 mm
free bore; the rest is relieved to 13 mm. It is custom machined from iglidur J
stock, not a catalogue bushing mounted in its usual housing. No housing or bond
restrains it. An offline axisymmetric linear-elastic calculation includes the
stepped flange, free outer boundary and unilateral shaft contact; radial deformation
is reduced to an axial friction force. Nominal full-overlap resistance is 14.61 N;
a tabulated normal load decreases with remaining overlap. The shaft guide releases
when the physical land clears the 38 mm cup rim at 28 mm rise. A 13 mm rigid collision proxy prevents
micron-scale rigid interpenetration; it is not the physical free bore. The last
3 mm of shaft is the centre cup: outer diameter12 mm and inner diameter7 mm,
not a solid journal. Its radial compliance is approximated using the hollow-cylinder
Lamé expression; solid and cup contact lengths weight their compliance in series
with the polymer. This reduces initial14.704 N to14.609 N, and full cup-only
contact to14.471 N. Mixed-contact end effects are not resolved; the compliance
correction is under2% of the rigid-steel estimate. Resistance stays active through
the rim, rather than disappearing at the lower35 mm shoulder.

Material source: [igus J data](https://www.igus.com/plastic-bearings/resources/iglide-material-j),
E=2.4 GPa, density=1.49 g/cm3, dynamic steel friction 0.06-0.18.
[NSK fits guide](https://www.nsk.com/content/dam/nsk/am/pt_br/product/roller-bearings-/needle-roller-bearings/doc%202.pdf),
section8.4, describes extraction force from contact pressure, area and friction.
`tools/axisymmetric_fit.py` and `tools/fit_estimates.json` reproduce the calculation.
Poisson ratio0.35 and static friction equal to nominal dynamic0.12 are assumptions;
this is not force calibration. No creep, wear or breakaway aging is represented.
Mesh refinement from0.5 to0.25 to0.125 mm gives15.15,14.83,14.70 N; the last change
is0.84%. This is a numerical estimate, not a rigorous error bound. At bore/shaft
tolerance extremes7-13 micrometre interference, E +/-20%, nu0.30-0.45 and mu0.06-0.18,
rigid-steel calculated full-overlap resistance spans4.01-34.98 N; steel compliance
reduces these estimates by less than2%. This parameter study does not
bound real static breakout. Assembly and dimensions refer to20 C; use measured
parts at that temperature and record departures.
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

Historical diagnostics with the old 8 N model at a07e283 found 29.4 mm extraction with
counterholding versus 12.7 mm without it over the same eight turning strokes.
The unsupported tool rotated substantially. This supports counterholding in
this procedure; it does not prove that all longer one-arm strategies fail.

## Demonstration controller (observation-only)

`mechanical_puller/policy.py` reads only: the 14-D measured joint state, the
native 320x176 overhead image `top_cam`, elapsed time, its own memory, the
calibrated arm bases with forward/inverse kinematics, and the published fixed
layout (start pose of the prepared puller, screw lead, tray and rest position).
It never reads simulator poses or joints, the embodiment or the scorer;
`tests/test_obs_boundary.py` runs the whole plan with those inaccessible.

- Tool pose: the published start pose until the right gripper closes on the end
  grip; afterwards right-hand FK of the measured joints composed with the grasp
  transform captured at that moment. Arms are commanded by position only.
- Extraction: all eleven bounded strokes execute. Measured wrist rotation times
  screw lead guides the holder upward, but can overestimate actual advance when
  the knob slips. It is a following heuristic, not an extraction measurement.
  The camera gate must independently confirm the collar left the shaft before
  release; simulator state is not read by the policy.
- Camera gate: after the carry, `shaft_view.py` must read the shaft flange as
  visible and free of the yellow collar for five consecutive frames before the tool is
  lowered and the jaws are spread. Unreadable or occluded frames never count; if
  this is not confirmed within 4 s, the arms hold and nothing is released. It was
  checked against the prior failed extraction: the revised mask reads the yellow
  collar as present in all 340 sampled late-episode frames where it remained on the shaft.
- Parking relies on the grasp transform and fixed rest position, including a
  +2 mm x command offset. Corrected nominal final body errors were (+1.83,-2.47)
  and (+0.23,-2.94) mm in x/y against the 4 mm tolerance; height errors were
  +0.04/+0.03 mm. There is no visual correction of parked position.
- The demonstrated layout and seed are fixed. Reset succeeds, but the trajectories
  are not identical: final screw turns were2.937 and3.513.

## Evidence

Corrected nominal run `fit-nominal-003`, runtime commit `d6609a4`: **1.0 / 1.0**,
including reset repeat, zero errors. Both episodes succeed at79.4 s (1,588 steps)
within the2,000-step limit. Collar clearance occurs at measured rises28.129 and
28.017 mm, after clearing the38 mm cup rim. Both final grippers are open, collar
rests inside the tray, and the unchanged scorer confirms two seconds of released
stability. All22 CPU checks pass. Videos show left wrist | right wrist | top at
960x176,20 fps,H.264/yuv420p,1,589 frames/79.45 s each. The setup image is the
native320x176 top camera. Matching Dream-1 (`dreamlabs/dream-1-beta`) evaluation scored **0.0** in one error-free
2,000-step episode, seed 0, terminating at the action limit. Its runtime source
and instruction match this demonstration; the full model-run video is supplied.

The parameter study is not demonstrated hardware tolerance: an isolated upper
case (13 micrometre interference,E2.88 GPa,nu0.30,mu0.18, calculated34.71 N)
scored0.0 in one error-free2,000-step episode. Collar rise reached only0.292 mm;
the camera correctly refused release. Peak simulated screw-coupling force40.37 N
does not establish force delivered to the collar or successful tool load transfer.
The60 N model cap is not a demonstrated robot capability. Only the nominal
calculated fit has a full-score demonstration; actual breakout, wear and material
variation are not calibrated by this evidence.

Earlier results (`rev-004`1.0/1.0 and its Dream0.0) used the old unexplained8 N
fit and are historical, not evidence for this revision. The provisional25 mm
release versions `fit-nominal-001`0.0/1.0 and `fit-nominal-002`1.0/1.0 also are
superseded by the corrected28 mm source. Earlier counterhold comparisons predate
both this fit and the tray rest; they do not establish an exhaustive one-arm
exclusion for the present task.

`render_video.py RUN_DIRECTORY [EPISODE]` renders raw frames. It is an optional
development helper requiring `imageio` and its FFmpeg backend, already present
in the validated stack; they are not needed by task physics or scoring.

## Attribution

Original task geometry and scripts. Station-wrapper and manipulation helpers
were adapted from earlier team task prototypes, using the organizer's YAM
adapter. No external mesh assets are bundled. Original task code and generated geometry are MIT-licensed; see `LICENSE`.
Upstream runtime dependencies retain their own licenses.

## Response to the fit review

The [latest review](https://github.com/robocurve/benchmark-tasks/pull/369#issuecomment-6872582744)
correctly identified that a 13 mm metal bore around a 12 mm shaft could not
produce the previously specified 8 N resistance. This revision replaces that
unsupported fit with the specified integral polymer interference land and
geometry-derived elastic/contact model described above. The 13 mm collision
proxy is explicitly distinguished from the physical 11.990 mm free land bore.
Nominal extraction and released recovery are demonstrated through the complete
38 mm shaft/cup geometry. Force estimates are calculated, not measured.

Matching model-run, full-score demonstration and setup image: media `M-YHNVC3`.
