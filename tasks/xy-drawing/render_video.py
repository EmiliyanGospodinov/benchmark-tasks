"""Render the adapter's raw RGB frames: python render_video.py RUN_DIR [EPISODE]."""
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np

root=Path(sys.argv[1]);episode=int(sys.argv[2]) if len(sys.argv)>2 else 0
cameras=('left_cam','right_cam','top_cam')
files=[sorted((root/'frames').rglob(f'*-e{episode}_{cam}_*.npy')) for cam in cameras]
if not files[0] or len({len(x) for x in files})!=1:
    raise SystemExit('Missing or unequal camera frames')
output=root/('demo.mp4' if episode==0 else f'demo-e{episode}.mp4')
with imageio.get_writer(output,fps=20,codec='libx264',macro_block_size=None,ffmpeg_params=['-movflags','+faststart']) as out:
    for i,paths in enumerate(zip(*files)):
        frames=[np.load(p) for p in paths]
        out.append_data(np.concatenate(frames,axis=1))
        if i==0 and episode==0:imageio.imwrite(root/'setup-top.png',frames[2])
        if i%100==0:imageio.imwrite(root/f'review-e{episode}-{i:04d}.png',np.concatenate(frames,axis=1))
print(output)
