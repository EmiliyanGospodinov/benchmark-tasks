import numpy as np
from xy_drawing import design as d
from xy_drawing.scoring import geometry,DrawingScore,dense

def world(p):return np.c_[np.asarray(p)+d.START[:2],np.full(len(p),d.START[2])]
def staircase(step):
 p=[d.POINTS[0].copy()]
 for a,b in zip(d.POINTS[:-1],d.POINTS[1:]):
  n=int(np.ceil(max(abs(b-a))/step))
  for k in range(1,n+1):
   goal=a+(b-a)*k/n;p.append(np.array([goal[0],p[-1][1]]));p.append(goal)
 return world(p)

def test_empty_and_start_have_no_credit():
 assert geometry([])['geometric_score']==0
 assert geometry(world([[0,0]]))['geometric_score']==0

def test_first_segment_partial():assert geometry(world(dense(d.POINTS[:2])))['geometric_score']==.5

def test_both_segments_need_final_clear_stable():
 p=world(dense(d.POINTS));assert geometry(p)['geometric_score']==.75
 s=DrawingScore();g={'left':{'tcp':[0,.3,1.1]},'right':{'tcp':[0,-.3,1.1]}}
 assert s.update(p,g,0)['score']==.75
 for t in np.arange(.05,2.1,.05):out=s.update(p,g,t)
 assert out['score']==1

def test_endpoint_shortcut_rejected():assert geometry(world(d.POINTS[[0,2]]))['geometric_score']==0

def test_fine_one_arm_staircase_allowed():assert geometry(staircase(.005))['geometric_score']==.75

def test_coarse_staircase_rejected_geometrically():assert geometry(staircase(.02))['geometric_score']==0

def test_extra_ink_is_not_erased_by_correct_finish():
 p=np.vstack([d.POINTS[0],[0,.05],d.POINTS]);assert geometry(world(p))['geometric_score']==0

def test_clearance_cannot_replace_drawing():
 s=DrawingScore();g={'left':{'tcp':[0,.3,1.1]},'right':{'tcp':[0,-.3,1.1]}}
 assert s.update([],g,10)['score']==0

def test_holding_knob_keeps_complete_trace_at_partial():
 s=DrawingScore();g={'left':{'tcp':d.KNOBS[0]},'right':{'tcp':[0,-.3,1.1]}};p=world(dense(d.POINTS))
 for t in np.arange(0,2.1,.05):out=s.update(p,g,t)
 assert out['score']==.75


def test_lifted_gap_not_bridged_even_if_tolerance_coverage_passes():
 p=world([[0,0],[.035,.035],[.04,.04],[.08,.08],[.14,.04]])
 assert geometry(p)['geometric_score']==.75
 broken=geometry(p,breaks=[2])
 assert min(broken['coverage'])>=.95
 assert not broken['connected']
 assert broken['geometric_score']==0

def test_contact_break_at_same_point_leaves_connected_visible_line():
 p=world([[0,0],[.04,.04],[.04,.04],[.08,.08],[.14,.04]])
 assert geometry(p,breaks=[2])['geometric_score']==.75
