"""Stitch stored policy frames (inspect-robots --store-frames, .npy per camera and step) into one H.264 video.

    python tools/video.py FRAMES_DIR EPISODE_PREFIX OUT.mp4     # e.g. scene-0-e0
Panels: left wrist | right wrist | top, 20 fps.
"""
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np

frames, prefix, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cams = ("left_cam", "right_cam", "top_cam")
steps = sorted(int(p.stem.rsplit("_", 1)[1]) for p in frames.glob(f"{prefix}_top_cam_*.npy"))
with iio.get_writer(out, fps=20, codec="libx264", pixelformat="yuv420p", macro_block_size=8,
                    output_params=["-movflags", "+faststart"]) as w:
    for k in steps:
        w.append_data(np.concatenate([np.load(frames / f"{prefix}_{c}_{k:06d}.npy") for c in cams], axis=1))
print(out, len(steps), "frames")
