"""Fixed-shape, uniform replay sampling suitable for CUDA graph capture."""
from __future__ import annotations
import torch
from morphmimic.train.replay import WindowBatch,StartBatch


class GraphReplaySampler:
    """Enumerate all ring ages; call check methods before every graph replay.

    Sampling uses device RNG and replacement from all valid rows. Checks deliberately
    run outside a graph: replay contents can change after capture.
    """
    def __init__(self,replay):
        self.replay=replay
        self.env=torch.arange(replay.num_envs,device=replay.device).repeat_interleave(replay.capacity)
        self.age=torch.arange(replay.capacity,device=replay.device).repeat(replay.num_envs)

    def _window_mask(self,k,env_range):
        r=self.replay
        if not 1<=k<r.capacity:raise ValueError('k must be within replay capacity')
        if env_range is not None and not 0<=env_range[0]<env_range[1]<=r.num_envs:
            raise ValueError('Invalid replay environment range')
        offsets=torch.arange(k+1,device=r.device)
        slots=r._slot(self.env[:,None],self.age[:,None]+offsets)
        steps=r.ep_step[self.env[:,None],slots].long()
        valid=(self.age+k<r.size[self.env]) & (steps==steps[:,:1]+offsets).all(dim=1)
        valid &= ~r.no_action[self.env[:,None],slots[:,:-1]].any(dim=1)
        if env_range is not None:valid &= (self.env>=env_range[0]) & (self.env<env_range[1])
        return valid

    def _start_mask(self,hist_len,accept_frame):
        r=self.replay
        if not 1<=hist_len<=r.capacity:raise ValueError('hist_len must be within replay capacity')
        slot=r._slot(self.env,self.age)
        episode_start=self.age-r.ep_step[self.env,slot].long()
        earliest=torch.maximum(self.age+1-hist_len,episode_start)
        valid=(self.age<r.size[self.env]) & ~r.no_action[self.env,slot] & (earliest>=0)
        if accept_frame is not None:
            accepted=accept_frame(r.frame[self.env,slot])
            if accepted.shape!=valid.shape or accepted.dtype!=torch.bool:raise ValueError('accept_frame must return a boolean mask')
            valid &= accepted
        return valid

    def _check(self,valid,what):
        if valid.device.type=='cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Validate replay availability outside graph capture')
        if not bool(valid.any()):raise RuntimeError('replay buffer holds no valid '+what)

    def check_windows(self,k,env_range=None):self._check(self._window_mask(k,env_range),'windows')
    def check_starts(self,hist_len,accept_frame=None):
        self.replay.check_controller_clock()
        self._check(self._start_mask(hist_len,accept_frame),'start rows')

    def _sample(self,mask,batch,what):
        if batch<1:raise ValueError('batch must be positive')
        if mask.device.type!='cuda' or not torch.cuda.is_current_stream_capturing():self._check(mask,what)
        index=torch.multinomial(mask.float(),batch,replacement=True)
        return self.env[index],self.age[index]

    def windows(self,batch,k,env_range=None):
        r=self.replay;env,age=self._sample(self._window_mask(k,env_range),batch,'windows')
        offsets=torch.arange(k+1,device=r.device);env2=env[:,None]
        slots=r._slot(env2,age[:,None]+offsets)
        return WindowBatch(state=r.state[env2,slots],action=r.action[env2,slots[:,:-1]],
            reward=r.reward[env2,slots[:,:-1]],terminated=r.terminated[env2,slots[:,:-1]],
            ctx=r.ctx[env,slots[:,0]],frame=r.frame[env2,slots],controller=r.controller_windows(env2,slots[:,:-1]))

    def starts(self,batch,hist_len,accept_frame=None):
        r=self.replay
        if r.device.type != "cuda" or not torch.cuda.is_current_stream_capturing(): r.check_controller_clock()
        env,age=self._sample(self._start_mask(hist_len,accept_frame),batch,'start rows')
        slot=r._slot(env,age);start=age-r.ep_step[env,slot].long()
        offsets=torch.arange(1-hist_len,1,device=r.device)
        hist=r._slot(env[:,None],torch.maximum(age[:,None]+offsets,start[:,None]))
        return StartBatch(state=r.state[env,slot],obs_hist=r.obs[env[:,None],hist],ctx=r.ctx[env,slot],frame=r.frame[env,slot],
            controller=r.controller_start(env,slot),reset_clock=r.sample_controller_clock(batch))
