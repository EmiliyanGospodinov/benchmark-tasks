"""Privileged joint-target feasibility controller. No object pose changes."""
import math
import numpy as np
from inspect_robots.types import Action,ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena import ik,station
from yam_arena.constants import FINGER_OPEN,READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy
from .motion import Seg,Script
from . import design as d
YAW0=math.radians(-50)
def point(i,yaw,dz=0):return lambda sim:(d.KNOBS[i]+[0,0,.003+dz],grip_rotation(yaw))
def plan(i):
 s=[Seg(3.,point(i,YAW0,.04),FINGER_OPEN,'approach'),Seg(1.,point(i,YAW0),FINGER_OPEN,'descend'),Seg(.7,point(i,YAW0),-.012,'grip')]
 for k,xy in enumerate(d.POINTS[1:]):s.append(Seg(6.,point(i,YAW0+xy[i]/d.RATIO),-.012,'segment'+str(k)))
 s.extend([Seg(1.,None,-.012,'hold'),Seg(.6,None,FINGER_OPEN,'release'),Seg(1.,point(i,YAW0+d.POINTS[-1,i]/d.RATIO,.06),FINGER_OPEN,'retreat')])
 s.append(Seg(2.5,lambda sim:ik.tcp_pose(np.array(READY),station.BASES['left' if i==0 else 'right']),FINGER_OPEN,'ready'))
 return s
def sequential_plan(step_mm):
 """One real left gripper alternates controls; right stays READY. Privileged angle feedback."""
 s=[]
 for a,b in zip(d.POINTS[:-1],d.POINTS[1:]):
  n=int(np.ceil(np.max(np.abs(b-a))*1000/step_mm))
  for k in range(1,n+1):
   xy=a+(b-a)*k/n
   for i in (0,1):
    target=float(xy[i]/d.RATIO)
    s.extend([Seg(1.5,point(i,YAW0,.035),FINGER_OPEN,'transfer'),
     Seg(.5,point(i,YAW0),FINGER_OPEN,'descend'),Seg(.4,point(i,YAW0),-.012,'grip'),
     Seg(1.,lambda sim,i=i,target=target:point(i,YAW0+target-sim.angles[i])(sim),-.012,'turn'),
     Seg(.3,None,FINGER_OPEN,'release'),
     Seg(.4,lambda sim,i=i:(d.KNOBS[i]+[0,0,.038],sim._policy_scripts['left'].prev[1]),FINGER_OPEN,'clear')])
 s.append(Seg(2.5,lambda sim:ik.tcp_pose(np.array(READY),station.BASES['left']),FINGER_OPEN,'ready'))
 return s
class DrawingPolicy(YamScriptedPolicy):
 def __init__(self,mode='coop',**kwargs):super().__init__(**kwargs);self.mode=mode
 def reset(self,scene):
  if self.mode in ('coarse','fine'):
   self.scripts={'left':Script(sequential_plan(20. if self.mode=='coarse' else 5.),self._bases['left'])}
  else:self.scripts={side:Script(plan(i),self._bases[side]) for i,side in enumerate(('left','right')) if self.mode=='coop' or (self.mode=='xonly' and i==0)}
  from .embodiment import ACTIVE
  if ACTIVE is not None:ACTIVE._policy_scripts=self.scripts
  self.end=max(s.duration for s in self.scripts.values())+2.;self.printed=None
 def act(self,observation):
  from .embodiment import ACTIVE
  t=float(observation.state_time)+self._dt;out=[];labels=[]
  for side in ('left','right'):
   act,label=self.scripts[side].act(ACTIVE,t) if side in self.scripts else (np.r_[READY,1.],'-')
   out.append(act);labels.append(label)
  if labels!=self.printed:print('[xy-demo]',round(t,2),labels,flush=True);self.printed=labels
  return ActionChunk(actions=[Action(data=pack(*out),meta={'request_stop':True} if t>self.end else {})],control_hz=1/self._dt)
