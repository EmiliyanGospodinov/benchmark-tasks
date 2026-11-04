"""Privileged measurement diagnostic, ordinary joint-target actions only."""
import numpy as np
from inspect_robots.types import Action,ActionChunk
from inspect_robots_yam.packing import pack
from yam_arena.constants import FINGER_OPEN,READY
from yam_arena.demo import grip_rotation
from yam_arena.scripted import YamScriptedPolicy
from .motion import Seg,Script
from . import design as d

def point(p):
    return lambda sim:(np.asarray(p,float),grip_rotation(0.))

def held_gauge_point(p,feedback=False):
    # Privileged robot-target correction: reference-pad gaps and actual body yaw.
    # No object poses/forces are written. Needed after grasp/settling variations.
    from scipy.spatial.transform import Rotation
    dx=0.;yaw=0.
    def target(sim):
        nonlocal dx,yaw
        if feedback and sim is not None:
            poses=sim.poses();g=poses['gauge'];h=poses['housing']
            r=Rotation.from_quat(g[3:]).as_matrix()
            gap=h[0]-.008-(g[0]+r[0,0]*.132)
            dx=float(np.clip(dx+.10*(gap-.00012),-.003,.003))
            actual_yaw=np.arctan2(r[1,0],r[0,0])
            yaw=float(np.clip(yaw-.10*np.degrees(actual_yaw),-3.,3.))
        return (np.asarray(p,float)+[.010+dx,.002,.001],
            Rotation.from_euler("zy",[-1.+yaw,8.],degrees=True).as_matrix()@grip_rotation(0.))
    return target

def plans():
    gauge=d.GAUGE_ORIGIN+np.array([0,0,.013])
    shaft=d.ORIGIN+np.array([.095,0,.013])
    out={}
    for side,p in (("right",gauge-[.03,0,.051]),("left",shaft)):
        out[side]=[Seg(3.,point(p+[0,0,.04]),FINGER_OPEN,"approach"),
                   Seg(1.5,point(p),FINGER_OPEN,"descend"),
                   Seg(1.,point(p),-.014,"grasp")]
    out["right"] += [Seg(2.,held_gauge_point(gauge-[.03,0,0]),-.014,"lift instrument"),Seg(2.,held_gauge_point(gauge),-.014,"contact shaft"),Seg(9.,held_gauge_point(gauge,feedback=True),-.014,"hold instrument",freeze=False,blend_seconds=.5),Seg(2.,held_gauge_point(gauge-[.03,0,0]),-.014,"withdraw probe"),Seg(2.,point(gauge-[.03,0,.051]),-.014,"lower instrument")]
    out["left"] += [Seg(4.,None,-.014,"wait for instrument"),Seg(2.,point(shaft+[-.022,0,0]),-.014,"near stop"),
                    Seg(1.,None,-.014,"measure near"),
                    Seg(3.,point(shaft+[.020,0,0]),-.014,"far stop"),
                    Seg(1.,None,-.014,"measure far"),
                    Seg(2.,point(shaft+[-.022,0,0]),-.014,"return to near stop"),Seg(4.,None,-.014,"wait for instrument") ]
    for side,p in (("right",gauge-[.03,0,.051]),("left",shaft)):
        out[side] += [Seg(1.,None,FINGER_OPEN,"release"),Seg(2.,point(p+[0,0,.04]),FINGER_OPEN,"retreat")]
    return out

class MeasureFitScriptedPolicy(YamScriptedPolicy):
    def __init__(self,mode="coop",**kwargs):
        super().__init__(**kwargs); self.mode=mode
    def reset(self,scene):
        print("[a11-init] policy reset",flush=True)
        from . import full_policy
        full_policy._DIAGNOSTIC_CHOICE=None
        if self.mode=="fitprobe":p=full_policy.fit_probe()
        elif self.mode=="full":
            from .full_policy import extend
            p=extend(plans)
        else:p=plans()
        if self.mode=="idle": p={}
        elif self.mode=="nohold": del p["right"]
        self._scripts={arm:Script(segs,self._bases[arm]) for arm,segs in p.items()}
        self._end=max([s.duration for s in self._scripts.values()] or [5.])+2.
        self._printed=None
        print("[a11-init] policy ready",flush=True)
    def act(self,observation):
        from .embodiment import ACTIVE
        t=float(observation.state_time)+self._dt
        actions=[]; labels=[]
        for arm in ("left","right"):
            a,label=self._scripts[arm].act(ACTIVE,t) if arm in self._scripts else (np.r_[READY,1.],"idle")
            actions.append(a); labels.append(label)
        if labels!=self._printed:
            print(f"[measure-demo] {t:.2f} {labels} IK={max([s.worst_mm for s in self._scripts.values()] or [0]):.1f}mm",flush=True)
            self._printed=labels
        return ActionChunk(actions=[Action(data=pack(*actions),meta={"request_stop":True} if t>self._end else {})],control_hz=1/self._dt)
