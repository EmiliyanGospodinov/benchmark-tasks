import numpy as np
TOP=.7475
PAD_X=.36
PADS={'A':np.array([PAD_X,.09,TOP+.10]),'B':np.array([PAD_X,.09,TOP+.15]),'C':np.array([PAD_X,-.09,TOP+.10]),'D':np.array([PAD_X,-.09,TOP+.15])}
WIRES={'straight':(('A','C'),('B','D')),'crossed':(('A','D'),('B','C')),'open_ac':(('A','C'),),'open_bd':(('B','D'),)}
def quat_to_matrix(q):
 x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def start_poses():
 p={n:np.r_[v,[0,0,0,1.]] for n,v in {'probe_l':(.19,.15,TOP+.023),'probe_r':(.19,-.15,TOP+.023),'panel':(.365,0,TOP),'meter':(.46,.335,TOP+.12),'result_board':(.24,.22,TOP),'pointer':(.16,0,TOP+.018)}.items()}
 for s in ['l','r']:p['tip_'+s]=p['probe_'+s].copy()
 for name in ['dash','ol','one']:
  p['display_'+name]=p['meter'].copy();p['display_'+name][2]+=.008 if name=='dash' else -.02
 return p

def contact(probe,tip):
 if not np.all(np.isfinite(np.r_[probe,tip])) or abs(np.linalg.norm(probe[3:])-1)>.02 or abs(np.linalg.norm(tip[3:])-1)>.02:return None,0.
 r=quat_to_matrix(probe[3:]);rt=quat_to_matrix(tip[3:]);offset=r.T@(tip[:3]-probe[:3]);compression=-float(offset[0]);end=tip[:3]+rt@np.array([.065,0,0])
 # Compression comes from the native passive spring; distal face lies on the pad.
 # No proximity-only contact: require spring load, guide alignment, and face alignment.
 if not (.001<=compression<=.0085 and np.linalg.norm(offset[1:])<.001 and rt[0,0]>.985):return None,compression
 for label,p in PADS.items():
  if abs(end[0]-p[0])<.002 and np.linalg.norm((end-p)[1:])<.011:return label,compression
 return None,compression

def connected(wiring,a,b):
 graph={k:set() for k in PADS}
 for x,y in WIRES[wiring]:graph[x].add(y);graph[y].add(x)
 seen={a};pending=[a]
 while pending:
  x=pending.pop()
  for y in graph[x]-seen:seen.add(y);pending.append(y)
 return b in seen

def canonical_pair(labels):
 """Continuity is undirected: either probe may contact either terminal column."""
 pair=''.join(sorted(label or '-' for label in labels))
 return pair if pair in ('AC','AD','BC','BD') else '--'
