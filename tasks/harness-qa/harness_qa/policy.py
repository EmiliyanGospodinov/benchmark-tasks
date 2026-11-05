import numpy as np
from inspect_robots.types import Action,ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena.constants import FINGER_OPEN,READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy
from .motion import Script,Seg
from . import design as d

def target(p):
    p=np.array(p,float)
    return lambda sim:(p,grip_rotation(0.))

def plan(side,mode):
    y=.15 if side=='left' else -.15
    start=np.array([.19,y,d.TOP+.040])
    seq=[Seg(3,target(start+[0,0,.06]),FINGER_OPEN,'approach'),Seg(1.2,target(start),FINGER_OPEN,'descend'),Seg(.7,None,-.006,'grasp'),Seg(1,target(start+[0,0,.08]),-.006,'lift')]
    labels=['A','A','B','B'] if side=='left' else ['C','D','C','D']
    if mode in ['probe','stage']:labels=labels[:2]
    for label in labels:
        pad=d.PADS[label];contact=pad+np.array([-.061,0,.017])
        seq +=[Seg(1.2,target(contact+[-.04,0,0]),-.006,'align '+label),Seg(1.3,target(contact),-.006,'contact '+label),Seg(1.5,None,-.006,'measure '+label),Seg(.8,target(contact+[-.035,0,0]),-.006,'withdraw '+label)]
    if mode=='stage':
        pad=d.PADS[labels[-1]];p=pad+[-.061,0,.017]
        seq +=[Seg(1,target(p),-.006,'stage contact'),Seg(1,None,FINGER_OPEN if side=='left' else -.006,'release left'),Seg(2,target(p+[-.06,0,.04]) if side=='left' else None,FINGER_OPEN if side=='left' else -.006,'staging diagnostic')]
    seq +=[Seg(2,target(start+[0,0,.06]),-.006,'return'),Seg(1,target(start),-.006,'lower'),Seg(.6,None,FINGER_OPEN,'release'),Seg(1,target(start+[0,0,.08]),FINGER_OPEN,'clear')]
    if mode=='full':
        tag=np.array([.16,0,d.TOP+.030])
        def place(sim,up=0):
            from .scoring import verdict,ZONES
            label=verdict(sim.readings)
            if label is None:raise RuntimeError('Incomplete or unexpected measured connectivity map')
            return np.r_[ZONES[label],d.TOP+.034+up],grip_rotation(0.)
        if side=='left':
            seq += [Seg(2,target(tag+[0,0,.07]),FINGER_OPEN,'tag approach'),Seg(1,target(tag),FINGER_OPEN,'tag descend'),Seg(.7,None,-.006,'tag grasp'),Seg(1,target(tag+[0,0,.07]),-.006,'tag lift'),Seg(2,lambda sim:place(sim,.07),-.006,'record measured verdict'),Seg(1,place,-.006,'place tag'),Seg(.6,None,FINGER_OPEN,'release tag'),Seg(1,lambda sim:place(sim,.09),FINGER_OPEN,'clear tag')]
    return seq

class HarnessPolicy(YamScriptedPolicy):
    def __init__(self,mode='full',**kwargs):super().__init__(**kwargs);self.mode=mode
    def reset(self,scene):
        self.scripts={a:Script(plan(a,self.mode),self._bases[a]) for a in ['left','right']};self.end=max(s.duration for s in self.scripts.values())+2;self.last=None
    def act(self,observation):
        from .embodiment import ACTIVE
        t=float(observation.state_time)+self._dt;aa=[];labels=[]
        for arm in ['left','right']:
            a,l=self.scripts[arm].act(ACTIVE,t);aa.append(a);labels.append(l)
        if labels!=self.last:print('[demo]',t,labels,'IK mm',max(s.worst_mm for s in self.scripts.values()),flush=True);self.last=labels
        return ActionChunk(actions=[Action(data=pack(*aa),meta={'request_stop':True} if t>self.end+self.hold_s else {})],control_hz=1/self._dt)
