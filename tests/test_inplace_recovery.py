import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('inplace',Path(__file__).resolve().parents[1]/'scripts/prepare_inplace_recovery.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def motion():
 p=np.zeros((30,14,3));p[:,0,2]=np.r_[np.full(10,.2),np.ones(20)];q=np.zeros((30,14,4));q[:,:,0]=1
 target=dict(body_pos_w=np.repeat(p[-1:],11447,axis=0))
 return dict(body_pos_w=p,body_quat_w=q),target

def test_stops_at_first_sustained_rise_before_following_walk():
 source,target=motion();source['body_pos_w'][20:,0,0]=4
 n,_=m.recovery_cut(source,target,'lying');assert n==15

def test_rejects_locomotion_instead_of_warping_it_back_to_origin():
 source,target=motion();source['body_pos_w'][:15,0,0]=np.linspace(0,4,15)
 with pytest.raises(ValueError,match='footprint'):m.recovery_cut(source,target,'seated')

def test_half_squat_does_not_keep_source_turning_tail():
 source,target=motion();n,reason=m.recovery_cut(source,target,'half_squat');assert n==2 and 'direct' in reason
