"""Explicit CUDA graph stages; inputs and randomness are supplied by the caller."""
from __future__ import annotations
import copy
import math
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


def clone_tree(tree):
    leaves, spec = tree_flatten(tree)
    if any(x is not None and not isinstance(x, torch.Tensor) for x in leaves):
        raise ValueError('Graph trees must contain only tensors or None')
    return tree_unflatten([x.detach().clone() if x is not None else None for x in leaves],spec)


def copy_tree(destination, source):
    dst, ds = tree_flatten(destination); src, ss = tree_flatten(source)
    if ds != ss or len(dst) != len(src):raise ValueError('Graph input tree differs')
    for a,b in zip(dst,src):
        if a is None and b is None:continue
        if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device:
            raise ValueError('Graph input shape, dtype or device differs')
        a.copy_(b)


class CapturedBackward:
    """Capture a pure function returning a scalar loss, or a tuple with loss first.

    Caller owns sampling, stochastic tensors, updates and distributed synchronization.
    Parameter gradients must keep their buffers between backward graph replays.
    A shared pool requires one common capture stream and serialized graph replays.
    Returned outputs are copied before another graph can reuse its workspace;
    inputs and gradient buffers are allocated outside the shared pool.
    """
    def __init__(self, fn, parameters, *, pool=None, stream=None):
        self.fn=fn;self.params=list(parameters);self.graph=None
        if not self.params:raise ValueError('Graph needs parameters')
        self.device=self.params[0].device
        if self.device.type!='cuda' or any(p.device!=self.device for p in self.params):
            raise ValueError('Graph parameters must share one CUDA device')
        if (pool is None)!=(stream is None):
            raise ValueError('Shared graph pool and common capture stream must be supplied together')
        if stream is not None and stream.device!=self.device:
            raise ValueError('Capture stream must match the parameter device')
        self.pool=pool;self.stream=stream

    def __call__(self,*args):
        if self.graph is None:self._capture(args)
        copy_tree(self.inputs,args)
        for p,g in zip(self.params,self.grad_buffers):p.grad=g
        self.graph.replay()
        return clone_tree(self.outputs)

    def _capture(self,args):
        self.inputs=clone_tree(args)
        for p in self.params:
            if p.grad is None:p.grad=torch.zeros_like(p)
        self.grad_buffers=[p.grad for p in self.params]
        original=[p.detach().clone() for p in self.params]
        original_grads=[p.grad.detach().clone() for p in self.params]
        stream=self.stream if self.stream is not None else torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        def run():
            for p in self.params:p.grad.zero_()
            out=self.fn(*self.inputs)
            loss=out[0] if isinstance(out,tuple) else out
            if not isinstance(loss,torch.Tensor) or loss.numel()!=1:raise ValueError('First result must be scalar loss')
            loss.backward()
            return out
        try:
            with torch.random.fork_rng(devices=[self.device]):
                with torch.cuda.stream(stream):run()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream,pool=self.pool):outputs=run()
                torch.cuda.current_stream(self.device).wait_stream(stream)
            if any(not torch.equal(p.detach(),v) for p,v in zip(self.params,original)):
                raise ValueError('Captured backward function changed parameters')
            self.graph,self.outputs=graph,outputs
        finally:
            with torch.no_grad():
                for p,v,g in zip(self.params,original,original_grads):p.copy_(v);p.grad.copy_(g)


class CapturedOptimizerStep:
    """Capture Adam or the project MuonAdam with a device-side learning rate."""
    def __init__(self,opt,params,clip_fn):
        from morphmimic.train.optim import MuonAdam
        if not isinstance(opt,(torch.optim.Adam,MuonAdam)):
            raise ValueError('Captured optimizer requires Adam or MuonAdam')
        self.opt=opt;self.params=list(params);self.clip_fn=clip_fn;self.graph=None
        if not self.params or any(p.device.type!='cuda' for p in self.params):raise ValueError('CUDA parameters required')
        self.device=self.params[0].device
        if any(p.device!=self.device for p in self.params):raise ValueError('One CUDA device required')
        adam=opt.adam if isinstance(opt,MuonAdam) else opt
        if any(not isinstance(g['lr'],torch.Tensor) or g['lr'].device!=self.device for g in opt.param_groups):
            raise ValueError('Optimizer requires device tensor learning rates')
        if any(not g.get('capturable') for g in adam.param_groups):
            raise ValueError('Adam requires capturable=True')

    def step(self):
        external=[p.grad for p in self.params]
        if any(g is None for g in external):raise ValueError('Every captured parameter needs a gradient')
        if self.graph is None:self._capture(external)
        for dst,src in zip(self.grads,external):dst.copy_(src)
        try:
            for p,g in zip(self.params,self.grads):p.grad=g
            self.graph.replay()
            return clone_tree(self.clip_result) if self.clip_result is not None else None
        finally:
            for p,g in zip(self.params,external):p.grad=g

    def _step_optimizer(self):
        from morphmimic.train.optim import MuonAdam
        if not isinstance(self.opt,MuonAdam):
            self.opt.step();return
        # Public Muon converts tensor lr to a host scalar and cannot be graph captured.
        from torch.optim._muon import _zeropower_via_newtonschulz
        with torch.no_grad():
            for group in self.opt.muon.param_groups:
                lr=group['lr'];momentum=group['momentum']
                for p in group['params']:
                    grad=p.grad
                    if grad is None:continue
                    state=self.opt.muon.state[p]
                    if 'momentum_buffer' not in state:state['momentum_buffer']=torch.zeros_like(p)
                    buf=state['momentum_buffer'];buf.lerp_(grad,1-momentum)
                    update=grad.lerp(buf,momentum) if group['nesterov'] else buf
                    update=_zeropower_via_newtonschulz(update,group['ns_coefficients'],group['ns_steps'],group['eps'])
                    mode=group['adjust_lr_fn'];a,b=p.shape
                    ratio=math.sqrt(max(1,a/b)) if mode in (None,'original') else .2*math.sqrt(max(a,b)) if mode=='match_rms_adamw' else 1.
                    p.mul_(1-lr*group['weight_decay'])
                    p.add_(update.to(p.dtype)*(-lr*ratio))
            self.opt.adam.step()

    def _capture(self,external):
        self.grads=[g.detach().clone() for g in external]
        saved_params=[p.detach().clone() for p in self.params]
        saved_state={p:{k:(v.clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v)) for k,v in state.items()}
                     for p,state in self.opt.state.items()}
        stream=torch.cuda.Stream(device=self.device);stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            for p,g in zip(self.params,self.grads):p.grad=g
            with torch.random.fork_rng(devices=[self.device]):
                with torch.cuda.stream(stream):self.clip_fn();self._step_optimizer()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    clip_result=self.clip_fn();self._step_optimizer()
                torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph,self.clip_result=graph,clip_result
        finally:
            with torch.no_grad():
                for p,value in zip(self.params,saved_params):p.copy_(value)
                for p,state in self.opt.state.items():
                    old=saved_state.get(p,{})
                    for key,value in state.items():
                        if isinstance(value,torch.Tensor):
                            value.copy_(old[key]) if key in old else value.zero_()
                        elif key in old:state[key]=old[key]
                        else:raise ValueError('Cannot restore non-tensor lazy optimizer state')
                for p,g in zip(self.params,external):p.grad=g
