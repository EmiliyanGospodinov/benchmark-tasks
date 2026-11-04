"""Measurement-selected spacer fitting and physical rotation verification plan."""
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik,station
from yam_arena.constants import READY,FINGER_OPEN
from yam_arena.demo import grip_rotation
from .motion import Seg
from . import design as d
from .fitting import SPACERS,STARTS,GRIP_Z,seated_pose

_DIAGNOSTIC_CHOICE=None

def selected(sim):
    if _DIAGNOSTIC_CHOICE is not None:return _DIAGNOSTIC_CHOICE
    if sim is None:return 'shim_9'  # CPU reach preflight only
    measured=sim.scorer.initial_travel_mm
    if measured is None:raise RuntimeError('No valid initial measurement; refusing to select a spacer')
    thickness=d.SPEC.choose(measured/1000)
    return next(name for name,t in SPACERS.items() if abs(t-thickness)<1e-9)

def shim_point(kind,height=0):
    def target(sim):
        name=selected(sim)
        p=STARTS[name][:3].copy() if kind=='pick' else seated_pose(SPACERS[name])[:3].copy()
        p[2]+=GRIP_Z+.013+height
        return p,grip_rotation(0.)
    return target

class ShimPlacement:
    """Privileged grasp calibration and pose-error correction, based on A9's realignment approach."""
    def __init__(self):
        self.offset=None; self.orientation=None; self.bias=np.zeros(3)
    def target(self,height=0,y=0,mode='hold'):
        def f(sim):
            name=selected(sim);desired=seated_pose(SPACERS[name])[:3].copy()
            desired[1]=y;desired[2]+=height
            if sim is None:return desired+[0,0,GRIP_Z+.013],grip_rotation(0.)
            if mode=='released' and self.offset is not None:
                return desired+self.offset+self.bias,self.orientation
            from yam_arena.inspect_embodiment import ARMS
            ids=sim._ids[ARMS[0]][0]
            q=sim._base.scene[ARMS[0]].data.joint_pos.torch[0][ids].cpu().numpy()
            hp,hr=ik.tcp_pose(q,station.BASES['left'])
            hp+=sim._base.scene.env_origins[0].cpu().numpy()
            obj=sim.poses()[name]
            if self.offset is None:
                inv=Rotation.from_quat(obj[3:]).as_matrix().T
                self.offset=inv@(hp-obj[:3]);self.orientation=inv@hr
            local=Rotation.from_quat(obj[3:]).as_matrix().T@(hp-obj[:3])
            if np.linalg.norm(local-self.offset)>.025:
                raise RuntimeError('Spacer no longer follows the grasp; stop the diagnostic')
            if mode=='correct':
                self.bias+=desired-obj[:3]
                self.orientation=Rotation.from_quat(obj[3:]).as_matrix().T@self.orientation
            elif mode=='align':self.bias+=.12*(desired-obj[:3])
            elif mode=='servo_xz':self.bias+=.15*(desired-obj[:3])*[1,0,1]
            elif mode=='servo_xy':self.bias+=.15*(desired-obj[:3])*[1,1,0]
            if mode in ("align","servo_xz","servo_xy"):
                correction=Rotation.from_rotvec(-.08*Rotation.from_quat(obj[3:]).as_rotvec()).as_matrix()
                self.orientation=correction@self.orientation
            self.bias=np.clip(self.bias,-.02,.02)
            return desired+self.offset+self.bias,self.orientation
        return f

def installation():
    place=ShimPlacement()
    return [Seg(2.5,shim_point('pick',.04),FINGER_OPEN,'approach selected shim'),
            Seg(1.5,shim_point('pick'),FINGER_OPEN,'descend to shim'),
            Seg(1.,None,-.010,'grasp shim'),
            Seg(2.,shim_point('pick',.04),-.010,'lift shim'),
            Seg(3.,place.target(.04,.07),-.010,'carry outside channel'),
            Seg(.8,place.target(.04,.07,'correct'),-.010,'realign shim'),
            Seg(.8,place.target(.04,.07,'correct'),-.010,'realign shim again'),
            Seg(1.5,place.target(.04,0,'servo_xz'),-.010,'move over channel',freeze=False),
            Seg(1.5,place.target(.04,0,'align'),-.010,'center above channel',freeze=False),
            Seg(5.,place.target(0,0,'servo_xy'),-.010,'insert split spacer',freeze=False),
            Seg(1.,None,FINGER_OPEN,'release spacer'),
            Seg(2.,place.target(.04,mode='released'),FINGER_OPEN,'clear spacer')]

def rotor_point(deg=0,height=0):
    def target(sim):
        center=(sim.poses()['shaft'][:3] if sim is not None else d.ORIGIN)+[.095,0,0]
        r=Rotation.from_euler('x',deg,degrees=True).as_matrix()
        return center+r@np.array([0,0,.013+height]),r@grip_rotation(0.)
    return target

def ready(side):
    return lambda sim:ik.tcp_pose(np.array(READY),station.BASES[side])

def extend(measurement_plans):
    p=measurement_plans()
    install=installation()
    p['left']+=install
    p['right']+=[Seg(sum(s.duration for s in install),None,FINGER_OPEN,'wait for spacer fitting')]
    again=measurement_plans()
    for arm in p:p[arm]+=again[arm]
    turn=[Seg(2.,rotor_point(0,.04),FINGER_OPEN,'approach shaft test grip'),
          Seg(1.5,rotor_point(),FINGER_OPEN,'descend to shaft'),
          Seg(1.,None,-.014,'grasp shaft for rotation')]
    for angle in (-20,-40,-60,-80):turn.append(Seg(1.3,rotor_point(angle),-.014,'turn fitted shaft'))
    turn += [Seg(1.,None,FINGER_OPEN,'release rotated shaft'),
             Seg(2.,rotor_point(-80,.06),FINGER_OPEN,'withdraw from shaft'),
             Seg(3.,ready('left'),FINGER_OPEN,'return ready'),Seg(3.,None,FINGER_OPEN,'released verification')]
    p['left']+=turn
    p['right']+=[Seg(3.,ready('right'),FINGER_OPEN,'return ready'),
                 Seg(sum(s.duration for s in turn)-3,None,FINGER_OPEN,'released verification')]
    return p

def fit_probe():
    global _DIAGNOSTIC_CHOICE
    _DIAGNOSTIC_CHOICE='shim_9'
    p=installation()+[Seg(3.,ready('left'),FINGER_OPEN,'return ready'),Seg(2.,None,FINGER_OPEN,'observe retention')]
    return {'left':p,'right':[Seg(sum(s.duration for s in p),ready('right'),FINGER_OPEN,'idle')]}
