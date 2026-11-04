#!/usr/bin/env bash
# Run one trial of the tipper-truck task with the scripted demonstration.
# Usage: bash run.sh --log-dir RUN_DIR [extra inspect-robots args]   (ARENA_ENV: path to yam.env)
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ARENA_ENV:?set ARENA_ENV to the IsaacLab-Arena yam.env}"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export ISAACSIM_ASSET_ROOT="${ISAACSIM_ASSET_ROOT:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0}"
cd "$here"
instruction="${INSTRUCTION:-$(cat "$here/manual-bed.txt")}"
launcher=()
command -v run-experiment >/dev/null && launcher=(run-experiment)
exec "${launcher[@]}" inspect-robots run --embodiment yam_duplo_tipper --policy tipper_scripted \
    -E preassembled=brick,cab,hinge -E control_hz=20 -E cam_width=320 -E cam_height=176 \
    --instruction "$instruction" --scorer task_score --max-steps 2400 --store-frames "$@"
