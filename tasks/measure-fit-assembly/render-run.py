"""Render each episode's raw policy frames without overlays or rescaling."""
import sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
p=Path(sys.argv[1])
frames=next((p/"frames").iterdir())
prefixes=sorted({f.name.split("_top_cam_")[0] for f in frames.glob("*_top_cam_*.npy")})
assert prefixes, "No stored frames"
for prefix in prefixes:
 tops=sorted(frames.glob(prefix+"_top_cam_*.npy"))
 suffix="" if prefix=="scene-0-e0" else "-"+prefix
 output=p/(p.name+suffix+"-all-cameras.mp4")
 with imageio.get_writer(str(output),fps=20,codec="libx264",pixelformat="yuv420p",macro_block_size=1,output_params=["-movflags","+faststart"]) as writer:
  for top in tops:
   images=[np.load(top.with_name(top.name.replace("top_cam",cam))) for cam in ("left_cam","right_cam","top_cam")]
   writer.append_data(np.concatenate(images,axis=1).astype(np.uint8))
 imageio.imwrite(p/("setup-top"+suffix+".png"),np.load(tops[0]))
 for index in (100,160,220,280,320,380,500,650,750,950,1100,1250,1500,1650,1750):
  if index<len(tops):
   top=tops[index]
   imageio.imwrite(p/f"review{suffix}-{index}.png",np.concatenate([np.load(top.with_name(top.name.replace("top_cam",cam))) for cam in ("left_cam","right_cam","top_cam")],axis=1))
 print(output)
