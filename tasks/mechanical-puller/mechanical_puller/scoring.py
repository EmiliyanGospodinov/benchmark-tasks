"""Measured extraction, recovery and released parking; no policy-state inputs."""
from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation
from . import design as d


def matrix(p):
    return Rotation.from_quat(p[3:]).as_matrix()


def collar_in_tray(p):
    # Conservative bounding box enclosing the faceted ring, transformed by pose.
    corners=np.array([[x,y,z] for x in (-.023,.023) for y in (-.023,.023)
                      for z in (0.,d.HUB_H+d.RIM_H)]) @ matrix(p).T+p[:3]
    return bool(np.all(np.abs(corners[:,:2]-d.TRAY_XY) <= np.array(d.TRAY_INNER)/2)
                and corners[:,2].min() >= d.TOP+d.TRAY_FLOOR-.002
                and corners[:,2].max() <= d.TOP+d.TRAY_FLOOR+.030)


def tool_parked(p):
    body=p['body'];rb=matrix(body)
    return bool(np.linalg.norm(body[:2]-d.REST_XY)<=.004
                and abs(body[2]-d.TOP-d.PARK_Z)<=.003
                and Rotation.from_matrix(rb).magnitude()<=math.radians(5))


def intact(p):
    body=p['body'];rb=matrix(body)
    local={k:rb.T@(v[:3]-body[:3]) for k,v in p.items()}
    if np.linalg.norm(local['handle']-np.array([0.,0.,d.KNOB_Z]))>.004:
        return False
    if np.linalg.norm(local['ram'][:2])>.004 or not -.062<=local['ram'][2]<=.004:
        return False
    for name,sign in [('jaw_pos',1),('jaw_neg',-1)]:
        a=local[name]
        if np.linalg.norm(a[1:])>.004 or not .017<=sign*a[0]<=.044:
            return False
    return True


class PullerScore:
    def __init__(self,hz=20):
        self.hz=hz;self.history=deque(maxlen=round(2*hz)+1)
        self.extracted=False;self.success=False;self.reason='extract collar';self.score=0.

    def update(self,p,turns,free,grip_open,tcps):
        self.success=False
        arrays=[*p.values(),np.asarray(grip_open),np.asarray(tcps),np.asarray(turns)]
        if (not all(np.all(np.isfinite(a)) for a in arrays)
                or any(np.linalg.norm(v[3:])<1e-12 for v in p.values())
                or not intact(p)):
            self.history.clear();self.reason='invalid or separated tool';self.score=0.;return self.score
        b=p['body'];rb=matrix(b)
        jaw=[rb.T@(p[k][:3]-b[:3]) for k in ('jaw_pos','jaw_neg')]
        # Extraction must be observed with retained jaws and screw advancement.
        collar_local=rb.T@(p["collar"][:3]-b[:3])
        retained=(np.linalg.norm(collar_local[:2])<=.004 and abs(collar_local[2]-d.HOOK_Z)<=.006)
        if free and retained and turns>=2. and all(.018<=abs(j[0])<=.029 for j in jaw):
            self.extracted=True
        recovered=self.extracted and collar_in_tray(p['collar'])
        parked=tool_parked(p)
        grip_point=b[:3]+rb@np.array([0.,-d.BOSS[1]/2-d.HANDLE[1]-d.GRIP_BLOCK[1]/2+.005,d.GRIP_Z])
        points=np.array([*[v[:3] for k,v in p.items() if k!='fixture'],grip_point])
        clear=float(np.min(np.linalg.norm(np.asarray(tcps)[:,None,:]-points[None,:,:],axis=2)))>=.080
        released=min(grip_open)>=.8 and clear
        self.score=.5 if recovered else .25 if self.extracted else 0.
        self.reason='park tool and withdraw' if recovered else 'recover collar in tray' if self.extracted else 'extract collar'
        if recovered and parked and released:
            self.history.append({k:v.copy() for k,v in p.items() if k!='fixture'})
            stable=len(self.history)==self.history.maxlen
            if stable:
                for k,v in self.history[0].items():
                    for h in self.history:
                        if np.linalg.norm(h[k][:3]-v[:3])>.002 or (Rotation.from_quat(h[k][3:])*Rotation.from_quat(v[3:]).inv()).magnitude()>math.radians(2):
                            stable=False;break
                    if not stable:break
            if stable:
                self.score=1.;self.success=True;self.reason='recovered and released stable for 2 s'
        else:self.history.clear()
        return self.score
