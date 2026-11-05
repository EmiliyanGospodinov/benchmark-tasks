import numpy as np
from harness_qa import design as d

def poses(label,compression=.004):
 pad=d.PADS[label];base=np.r_[pad+[-.065+compression,0,0],[0,0,0,1]];tip=base.copy();tip[0]-=compression
 return base,tip

def test_each_pad_requires_compression():
 for label in d.PADS:
  a,b=poses(label);assert d.contact(a,b)[0]==label
  a,b=poses(label,0);assert d.contact(a,b)[0] is None

def test_air_alignment_is_not_contact():
 a,b=poses('A');b[0]-=.004;a[0]-=.004;assert d.contact(a,b)[0] is None

def test_spring_compressed_away_from_pad_invalid():
 a,b=poses('A');a[2]+=.03;b[2]+=.03;assert d.contact(a,b)[0] is None

def test_nonfinite_invalid():
 a,b=poses('A');b[0]=float('nan');assert d.contact(a,b)[0] is None

def test_explicit_connectivity():
 assert [d.connected('crossed',*p) for p in ['AC','AD','BC','BD']]==[False,True,True,False]
 assert [d.connected('straight',*p) for p in ['AC','AD','BC','BD']]==[True,False,False,True]
 assert [d.connected('open_ac',*p) for p in ['AC','AD','BC','BD']]==[True,False,False,False]

def test_corrupted_tip_orientation_invalid():
 a,b=poses('A');b[3:]=0;assert d.contact(a,b)[0] is None

def test_swapped_probes_have_identical_pair_and_reading():
 for pair in ['AC','AD','BC','BD']:
  left=d.contact(*poses(pair[0]))[0];right=d.contact(*poses(pair[1]))[0]
  assert d.canonical_pair([left,right])==d.canonical_pair([right,left])==pair
  for wiring in d.WIRES:assert d.connected(wiring,left,right)==d.connected(wiring,right,left)
 assert d.canonical_pair(['A',None])=='--'
 assert d.canonical_pair(['A','B'])=='--'
