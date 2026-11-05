"""Conservative world-AABB mesh sweep of wrists, with sampled nominal IK."""
import json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh
from yam_arena import YAM_URDF,station,ik
from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS,FINGER_OPEN
from yam_arena.urdf import forward_kinematics,rpy_matrix
from xy_drawing.policy import plan
from xy_drawing.motion import Script
root=ET.parse(YAM_URDF).getroot();meshes={}
for link in root.findall('link'):
 name=link.get('name')
 if name not in ('link4','link5','gripper','tip_left','tip_right'):continue
 points=[]
 for visual in link.findall('visual'):
  m=visual.find('geometry/mesh');v=trimesh.load(Path(YAM_URDF).parent/m.get('filename'),process=False).vertices
  o=visual.find('origin');xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3)
  rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ') if o is not None else np.zeros(3)
  v=v@rpy_matrix(rpy).T+xyz;points.extend(v)
 meshes[name]=np.array(points)
scripts={s:Script(plan(i),station.BASES[s]) for i,s in enumerate(('left','right'))}
minimum=1.;worst=0.;minpair=None;overlaps=[];lid_gap=1.;lid_overlaps=[]
for t in np.arange(0,22,.1):
 bounds={}
 for side,script in scripts.items():
  a,label=script.act(None,float(t));worst=max(worst,script.worst_mm)
  qs=dict(zip(ARM_JOINTS,a[:6]));qs.update({j:a[-1]*FINGER_OPEN for j in FINGER_JOINTS})
  fk=forward_kinematics(YAM_URDF,qs);bounds[side]={}
  for n,v in meshes.items():
   w=v@fk[n][:3,:3].T+fk[n][:3,3]+station.BASES[side];bounds[side][n]=(w.min(0),w.max(0))
   lo,hi=w.min(0),w.max(0)
   if hi[0]>=.24 and lo[0]<=.48 and hi[1]>=-.18 and lo[1]<=.18:
    gap=lo[2]-(.7475+.060);lid_gap=min(lid_gap,float(gap))
    if gap<=0 and hi[2]>=.7475+.058:lid_overlaps.append([float(t),side,n])
 for n,(lo,hi) in bounds['left'].items():
  for m,(lo2,hi2) in bounds['right'].items():
   distance=np.linalg.norm(np.maximum(0,np.maximum(lo-hi2,lo2-hi)))
   if distance<minimum:minimum=float(distance);minpair=[float(t),n,m]
   if distance==0:overlaps.append([round(float(t),2),n,m])
print(json.dumps({'max_ik_error_mm':worst,'minimum_between_wrist_mesh_AABBs_mm':minimum*1000,'minimum_at':minpair,'lid_vertical_gap_mm':lid_gap*1000,'lid_overlap_count':len(lid_overlaps),'overlap_count':len(overlaps),'first_overlaps':overlaps[:4],'scope':'sampled nominal trajectories; AABBs are conservative, cameras and compliance not included'},indent=2))
