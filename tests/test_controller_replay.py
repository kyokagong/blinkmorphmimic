import pytest
import torch
from morphmimic.model.controller import ControllerState, ControllerRecord
from morphmimic.train.replay import ReplayBuffer, ReplayCfg
from morphmimic.train.fixed_replay import GraphReplaySampler


def add(r,i,clock):
    z=lambda *s:torch.zeros(2,*s)
    state=ControllerState(torch.full((2,11,29),float(i)),torch.full((2,4),i,dtype=torch.long))
    after=ControllerState(state.queue+1,state.meta+1)
    record=ControllerRecord(state,after,torch.full((2,10,29),float(i+2)),torch.full((2,10),i,dtype=torch.long),torch.full((2,),clock))
    r.add_step(z(240),z(93),z(29),z(),torch.zeros(2,dtype=torch.bool),torch.full((2,),clock),z(33),torch.full((2,),i),torch.full((2,),i),controller=record)


def test_real_history_and_contiguous_clock_survive_ring_wrap_both_samplers():
    r=ReplayBuffer(ReplayCfg(2,capacity=5,controller_horizon=3))
    for i in range(9):add(r,i,i%3==0)
    for sampler in (r,GraphReplaySampler(r)):
        starts=sampler.sample_starts(32,1) if sampler is r else sampler.starts(32,1)
        torch.testing.assert_close(starts.controller.queue[:,0,0],starts.frame.float())
        assert starts.reset_clock.shape==(32,3)
        assert all(tuple(x) in {(False,False,True),(False,True,False),(True,False,False)} for x in starts.reset_clock.tolist())
        batch=sampler.sample_windows(32,2) if sampler is r else sampler.windows(32,2)
        torch.testing.assert_close(batch.controller.targets[:,:,0,0],batch.frame[:,:-1].float()+2)
        torch.testing.assert_close(batch.controller.after.queue[:,:,0,0],batch.frame[:,:-1].float()+1)


def test_missing_real_history_and_insufficient_clock_rejected_legacy_unchanged():
    r=ReplayBuffer(ReplayCfg(2,capacity=5,controller_horizon=3))
    add(r,0,False)
    with pytest.raises(RuntimeError,match='clock'):r.sample_starts(2,1)
    with pytest.raises(RuntimeError,match='clock'):GraphReplaySampler(r).check_starts(1)
    assert ReplayBuffer(ReplayCfg(2,capacity=5)).controller_before_queue is None


def test_manual_reset_invalidates_bootstrap_timeline_without_fabricating_clock():
    r=ReplayBuffer(ReplayCfg(2,capacity=5,controller_horizon=3))
    for i in range(4):add(r,i,True)
    r.check_controller_clock();r.clear_controller_clock()
    with pytest.raises(RuntimeError,match='clock'):r.sample_starts(2,1)
    for i in range(3):add(r,i,False)
    assert not r.sample_starts(10,1).reset_clock.any()
