#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/emiliyan/repos/robocurve-work/IsaacLab-Arena/yam.env
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
python probe.py "$@"
