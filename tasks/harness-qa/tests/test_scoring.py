import numpy as np
from harness_qa.scoring import HarnessScore,verdict,ZONES
from harness_qa import design as d

def state():return d.start_poses(),[1.,1.],[[0,0,1.2],[0,0,1.3]],{n:0. for n in ['probe_l','tip_l','probe_r','tip_r','pointer']}
def sample(s,pair,value,n=10,args=None):
 args=state() if args is None else args
 for _ in range(n):score=s.update(pair,value,*args)
 return score

def test_no_contact_ol_earns_nothing():
 s=HarnessScore();assert sample(s,'--',None,100)==0;assert s.readings=={}
def test_transient_not_accepted():
 s=HarnessScore();sample(s,'AC',False,9);sample(s,'--',None,1);sample(s,'AC',False,9);assert s.readings=={}
def test_valid_open_is_measurement():
 s=HarnessScore();assert sample(s,'AC',False)==.15;assert s.readings=={'AC':False}
def test_four_patterns_and_recheck():
 for bits,result in [('1001','PASS'),('0110','CROSSED'),('1000','OPEN'),('0001','OPEN'),('1111',None),('0100',None)]:
  assert verdict(dict(zip(['AC','AD','BC','BD'],[v=='1' for v in bits])))==result
 assert verdict({'AC':True}) is None

def measured():
 s=HarnessScore()
 for p,v in zip(['AC','AD','BC','BD'],[False,True,True,False]):sample(s,p,v)
 return s

def test_correct_tag_release_stow():
 s=measured();args=state();args[0]['pointer'][:3]=[*ZONES['CROSSED'],d.TOP+.022];assert sample(s,'--',None,40,args)==1

def test_wrong_tag():
 s=measured();args=state();args[0]['pointer'][:3]=[*ZONES['PASS'],d.TOP+.022];assert sample(s,'--',None,40,args)==.6

def test_held_probe_not_success():
 s=measured();args=state();args[0]['pointer'][:3]=[*ZONES['CROSSED'],d.TOP+.022];args[1][0]=.2;assert sample(s,'--',None,40,args)==.8

def test_unstowed_probe_not_success():
 s=measured();args=state();args[0]['pointer'][:3]=[*ZONES['CROSSED'],d.TOP+.022];args[0]['probe_l'][0]+=.1;assert sample(s,'--',None,40,args)==.8

def test_nonfinite_breaks_hold():
 s=measured();args=state();args[0]['pointer'][0]=np.nan;assert sample(s,'--',None,1,args)==0;assert s.hold==0

def test_reset_clears_measurements():
 s=measured();s.reset();assert s.readings=={} and not s.success

def test_invalid_orientation_breaks_hold():
 s=measured();args=state();args[0]['pointer'][3:]=0;assert sample(s,'--',None,1,args)==0;assert not s.success
