#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${YAM_ENV:-}" ]]; then source "$YAM_ENV"; fi
command -v inspect-robots >/dev/null || { echo "Activate the organizer environment or set YAM_ENV." >&2; exit 1; }
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
name="${1:?}"; stage="${2:-draft}"
instruction='Engage both jaws behind the collar. Hold the puller aligned while turning its screw clockwise to extract the collar.'
if [[ "${3:-draft}" == prepared ]]; then
 instruction='Both puller jaws are already engaged behind the fitted collar. Hold the loose puller aligned and turn its screw clockwise to extract the collar.'
fi
if [[ "$stage" == complete ]]; then instruction="$(cat instruction.txt)"; fi
mkdir "runs/$name"
cp -r mechanical_puller "runs/$name/source"
cp task.py instruction.txt task.json "runs/$name/"
printf '%s\n' "$instruction" > "runs/$name/instruction.txt"
git rev-parse HEAD > "runs/$name/source-commit.txt"
git diff -- . > "runs/$name/source-diff.patch"
inspect-robots run --embodiment yam_mechanical_puller --policy mechanical_puller_scripted -P stage="$stage" -E scene_kind="${3:-draft}" -E telemetry="$PWD/runs/$name/telemetry.jsonl" -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$instruction" --scorer task_score --max-steps 2000 --epochs "${4:-1}" --seed 0 --store-frames --log-dir "$PWD/runs/$name"
