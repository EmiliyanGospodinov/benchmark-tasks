import numpy as np
from mechanical_puller import design as d
from mechanical_puller.scoring import PullerScore,collar_in_tray,tool_parked


def final_poses():
    p=d.prepared_layout();old=p['body'][:3].copy();at=np.r_[d.REST_XY,d.TOP+d.PARK_Z]
    for k in ('body','handle','ram','jaw_pos','jaw_neg'):p[k][:3]+=at-old
    p['ram'][2]-=.031
    p['jaw_pos'][0]+=.014;p['jaw_neg'][0]-=.014
    p['collar'][:3]=[*d.TRAY_XY,d.TOP+d.TRAY_FLOOR]
    return p


def extracted(s):
    p=d.prepared_layout()
    for k in ('body','handle','ram','jaw_pos','jaw_neg','collar'):p[k][2]+=.029
    p['ram'][2]-=.031
    return s.update(p,2.6,True,[0.,1.],[[0,0,0],[0,0,0]])


def test_initial_zero():
    s=PullerScore();assert s.update(d.prepared_layout(),0,False,[1,1],[[0,0,0],[0,0,0]])==0

def test_extraction_partial():
    assert extracted(PullerScore())==.25

def test_no_screw_no_extraction():
    s=PullerScore();p=d.prepared_layout();p['collar'][2]+=.03
    assert s.update(p,0,True,[1,1],[[0,0,0],[0,0,0]])==0

def test_final_requires_history():
    s=PullerScore();p=final_poses()
    for _ in range(45):assert s.update(p,2.6,True,[1,1],[[0,0,0],[0,0,0]])==0

def test_full_after_dwell():
    s=PullerScore();extracted(s);p=final_poses()
    for _ in range(40):assert s.update(p,2.6,True,[1,1],[[0,0,0],[0,0,0]])==.5
    assert s.update(p,2.6,True,[1,1],[[0,0,0],[0,0,0]])==1

def test_still_held_not_success():
    s=PullerScore();extracted(s);p=final_poses()
    for _ in range(45):assert s.update(p,2.6,True,[0,1],[[0,0,0],[0,0,0]])==.5

def test_collar_off_tray():
    p=final_poses()['collar'];p[0]+=.05;assert not collar_in_tray(p)

def test_not_parked():
    p=final_poses();p['body'][0]+=.01;assert not tool_parked(p)

def test_invalid_zero():
    s=PullerScore();extracted(s);p=final_poses();p['collar'][0]=float('nan')
    assert s.update(p,2.6,True,[1,1],[[0,0,0],[0,0,0]])==0

def test_detached_handle_zero():
    s=PullerScore();extracted(s);p=final_poses();p['handle'][0]+=.02
    assert s.update(p,2.6,True,[1,1],[[0,0,0],[0,0,0]])==0


def test_open_hand_at_side_handle_is_not_withdrawn():
    s=PullerScore();extracted(s);p=final_poses()
    handle=p['body'][:3]+np.array([0.,-.135,d.GRIP_Z])
    for _ in range(45):assert s.update(p,2.6,True,[1,1],[[0,0,0],handle])==.5

def test_moving_tool_cannot_complete_dwell():
    s=PullerScore();extracted(s);p=final_poses()
    for i in range(45):
        q={k:v.copy() for k,v in p.items()}
        for k in ('body','handle','ram','jaw_pos','jaw_neg'):q[k][0]+=.003 if i%2 else 0
        assert s.update(q,2.6,True,[1,1],[[0,0,0],[0,0,0]])==.5


def test_unrelated_collar_and_turned_tool_do_not_earn_extraction():
    s=PullerScore();p=d.prepared_layout();p['collar'][:3]+=[.10,0,.03]
    assert s.update(p,2.6,True,[1,1],[[0,0,0],[0,0,0]])==0


def test_zero_quaternion_returns_zero_without_exception():
    for key in final_poses():
        s=PullerScore();extracted(s);p=final_poses();p[key][3:]=0
        assert s.update(p,2.6,True,[1,1],[[0,0,0],[0,0,0]])==0
