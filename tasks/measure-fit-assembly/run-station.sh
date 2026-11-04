#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
cd "$(dirname "$0")"
source /home/emiliyan/repos/robocurve-work/IsaacLab-Arena/yam.env
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
run_name="${1:?unique run name required}"
mkdir "runs/$run_name"
cp -r measure_fit "runs/$run_name/source"
cp task.py instruction.txt "runs/$run_name/"
git rev-parse HEAD > "runs/$run_name/source-commit.txt"
git diff -- . > "runs/$run_name/source-diff.patch"
instruction="$(cat instruction.txt)"
inspect-robots run --embodiment yam_measure_fit --policy measure_fit_scripted -P mode="${2:-coop}" -E scene=station -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 -E telemetry="$PWD/runs/$run_name/telemetry.jsonl" --instruction "$instruction" --scorer task_score --epochs "${3:-1}" --max-steps 2000 --store-frames --log-dir "$PWD/runs/$run_name"
