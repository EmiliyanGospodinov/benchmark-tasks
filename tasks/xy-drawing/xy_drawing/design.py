"""Original teaching plotter. Declared rotary/linear transmission, no commercial replica."""
import numpy as np
TOP=.7475
KNOBS=[np.array([.23,.13,TOP+.080]),np.array([.23,-.13,TOP+.080])]
START=np.array([.29,-.07,TOP+.028])
POINTS=np.array([[0.,0.],[.08,.08],[.14,.04]])
RATIO=.070 # metres per radian: equivalent 70 mm cable drum radius
K=300.; DAMP=3.; MAX_FORCE=5.
PARTS=('frame','knob_x','knob_y','stage_x','stylus')
def layout():
 p={'frame':np.array([.30,0,TOP+.012]),'knob_x':KNOBS[0],'knob_y':KNOBS[1],
    'stage_x':START+np.array([0,0,.018]),'stylus':START+np.array([0,0,.009])}
 return {k:np.r_[v,0,0,0,1.] for k,v in p.items()}
