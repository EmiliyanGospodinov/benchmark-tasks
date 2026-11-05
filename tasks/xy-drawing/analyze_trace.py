"""Diagnostic geometric measurements; not the task scorer. No simulated success inferred."""
import json,sys
from pathlib import Path
import numpy as np
from xy_drawing import design as d

def point_to_segments(points,vertices):
 p=np.asarray(points,float);best=np.full(len(p),np.inf)
 for a,b in zip(vertices[:-1],vertices[1:]):
  v=b-a;den=v@v
  if den<1e-16:continue
  u=np.clip((p-a)@v/den,0,1)
  best=np.minimum(best,np.linalg.norm(p-a-u[:,None]*v,axis=1))
 return best
if __name__=='__main__':
 r=Path(sys.argv[1]);rows=[json.loads(l) for l in (r/'telemetry.jsonl').read_text().splitlines() if '"t"' in l]
 if (r/'trace-final.json').exists():
  p=(np.array(json.loads((r/'trace-final.json').read_text()))[:,:2]-d.START[:2])*1000
 else:p=np.array([x['xy_mm'] for x in rows])
 targets=d.POINTS*1000;error=point_to_segments(p,targets)
 coverage=[]
 for a,b in zip(targets[:-1],targets[1:]):
  q=a+np.linspace(0,1,101)[:,None]*(b-a)
  coverage.append(float(np.mean(point_to_segments(q,p)<=6)))
 print(json.dumps({'time_s':rows[-1]['t'],'end_mm':p[-1].tolist(),'max_deviation_mm':float(error.max()),'coverage_within6mm':coverage,'diagnostic_only':True},indent=2))
