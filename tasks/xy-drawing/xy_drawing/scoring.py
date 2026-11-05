"""Outcome-only geometry of the visible trace; no simultaneous-hand predicate."""
from collections import deque
import numpy as np
from . import design as d
TOL=.006

def distances(points,vertices):
 p=np.asarray(points,float).reshape(-1,2);v=np.asarray(vertices,float).reshape(-1,2)
 best=np.full(len(p),np.inf)
 if len(v)==1:return np.linalg.norm(p-v[0],axis=1)
 for a,b in zip(v[:-1],v[1:]):
  delta=b-a;den=delta@delta
  if den<1e-16:continue
  u=np.clip((p-a)@delta/den,0,1)
  best=np.minimum(best,np.linalg.norm(p-a-u[:,None]*delta,axis=1))
 return best

def dense(vertices,spacing=.001):
 out=[]
 for a,b in zip(vertices[:-1],vertices[1:]):
  n=max(1,int(np.ceil(np.linalg.norm(b-a)/spacing)))
  out.extend(a+np.arange(n)[:,None]/n*(b-a))
 if len(vertices):out.append(vertices[-1])
 return np.asarray(out).reshape(-1,2)

def connected(strokes,width=.004):
 """Connected components of the final round-capped physical ink strokes."""
 if len(strokes)<2:return True
 graph=[set() for _ in strokes]
 def intersects(a,b):
  # Endpoint-to-segment distance plus true crossings gives minimum segment distance.
  if min(distances(a,b).min(),distances(b,a).min())<=width+1e-9:return True
  if len(a)<2 or len(b)<2:return False
  cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
  q=b[:-1];v=b[1:]-q
  for p,r in zip(a[:-1],np.diff(a,axis=0)):
   den=cross(r,v);valid=np.abs(den)>1e-12
   if not np.any(valid):continue
   t=cross(q[valid]-p,v[valid])/den[valid];u=cross(q[valid]-p,r)/den[valid]
   if np.any((t>=0)&(t<=1)&(u>=0)&(u<=1)):return True
  return False
 for i,a in enumerate(strokes):
  for j in range(i):
   if intersects(a,strokes[j]):graph[i].add(j);graph[j].add(i)
 seen={0};todo=[0]
 while todo:
  i=todo.pop()
  for j in graph[i]-seen:seen.add(j);todo.append(j)
 return len(seen)==len(strokes)

def geometry(world_points,breaks=()):
 p=np.asarray(world_points,float)
 result={'geometric_score':0.,'coverage':[0.,0.],'max_error_mm':None,'endpoint_error_mm':None,'connected':False}
 if p.ndim!=2 or len(p)<2 or p.shape[1]<2 or not np.isfinite(p).all():return result
 p=p[:,:2]-d.START[:2];target=d.POINTS
 cuts=[0,*sorted(set(i for i in breaks if 0<i<len(p))),len(p)]
 strokes=[p[a:b] for a,b in zip(cuts[:-1],cuts[1:])]
 sample=np.concatenate([dense(s) for s in strokes]);error=distances(sample,target);worst=float(error.max())
 coverage=[]
 for a,b in zip(target[:-1],target[1:]):
  q=dense(np.array([a,b]));coverage.append(float(np.mean(np.min([distances(q,s) for s in strokes],axis=0)<=TOL)))
 end=float(np.linalg.norm(p[-1]-target[-1]));joined=connected(strokes)
 result.update(coverage=coverage,max_error_mm=worst*1000,endpoint_error_mm=end*1000,connected=joined)
 if not joined or worst>TOL or np.linalg.norm(p[0]-target[0])>TOL:return result
 if coverage[0]>=.95:result['geometric_score']=.5
 if min(coverage)>=.95 and end<=TOL:result['geometric_score']=.75
 return result

class DrawingScore:
 def __init__(self):self.history=deque();self.last=None
 def update(self,points,grips,t,breaks=()):
  info=geometry(points,breaks);score=info['geometric_score']
  endpoint=np.asarray(points[-1][:2]) if len(points) else np.array([np.nan,np.nan])
  self.history.append((float(t),endpoint))
  while self.history and self.history[0][0]<t-2.05:self.history.popleft()
  # Final clearance is only for exposing the completed output; neither arm is required to have been used.
  clear=all(all(np.linalg.norm(np.asarray(g['tcp'])-k)>=.10 for k in d.KNOBS) for g in grips.values()) and len(grips)==2
  stable=False
  if len(self.history)>1 and self.history[-1][0]-self.history[0][0]>=2.-1e-6:
   h=np.array([p for _,p in self.history]);stable=bool(np.isfinite(h).all() and np.linalg.norm(np.ptp(h,axis=0))<=.001)
  if score==.75 and stable and clear:score=1.
  info.update(score=score,success=score==1.,clear=clear,stable=stable)
  self.last=info;return info
