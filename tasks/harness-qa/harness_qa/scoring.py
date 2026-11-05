"""Settled acquisition is independent of electrical continuity (OL can be valid)."""
import numpy as np
from . import design as d
PAIRS=('AC','AD','BC','BD')
ZONES={'PASS':np.array([.14,.22]),'CROSSED':np.array([.24,.22]),'OPEN':np.array([.34,.22])}
def verdict(readings):
 if set(readings)!=set(PAIRS) or any(type(v) is not bool for v in readings.values()):return None
 bits=''.join('1' if readings[k] else '0' for k in PAIRS)
 return {'1001':'PASS','0110':'CROSSED','1000':'OPEN','0001':'OPEN'}.get(bits)
class HarnessScore:
 def __init__(self,dt=.05):self.dt=dt;self.reset()
 def reset(self):self.pair=None;self.value=None;self.settle=0.;self.readings={};self.hold=0.;self.success=False;self.result=None
 def update(self,pair,value,poses,opening,tcps,speeds):
  values=np.r_[*[np.asarray(p).ravel() for p in poses.values()],opening,np.asarray(tcps).ravel(),list(speeds.values())]
  if not np.isfinite(values).all() or any(abs(np.linalg.norm(p[3:])-1)>.02 for p in poses.values()):self.pair=None;self.settle=0.;self.hold=0.;self.success=False;return 0.
  valid=pair in PAIRS and type(value) is bool
  self.settle=self.settle+self.dt if valid and pair==self.pair and value==self.value else self.dt if valid else 0.
  self.pair=pair if valid else None;self.value=value if valid else None
  if self.settle>=.5-1e-9:self.readings[pair]=value
  self.result=verdict(self.readings)
  score=.15*len(self.readings)
  tag=poses['pointer'];tag_ok=self.result is not None and np.max(np.abs(tag[:2]-ZONES[self.result]))<=.025 and abs(tag[2]-(d.TOP+.022))<.004 and d.quat_to_matrix(tag[3:])[2,2]>.966
  if tag_ok:score+=.2
  stowed=all(np.linalg.norm(poses['probe_'+s][:2]-d.start_poses()['probe_'+s][:2])<.03 and abs(poses['probe_'+s][2]-(d.TOP+.023))<.004 and d.quat_to_matrix(poses['probe_'+s][3:])[2,2]>.966 for s in ['l','r'])
  released=min(opening)>.8 and all(np.linalg.norm(np.asarray(t)-poses[n][:3])>.07 for t in tcps for n in ['probe_l','probe_r','pointer'])
  stable=max(speeds.values())<.02
  self.hold=self.hold+self.dt if tag_ok and stowed and released and stable else 0.
  self.success=self.hold>=2.-1e-9
  return 1. if self.success else score
