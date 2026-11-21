"""Shared backward memory pools preserve gradients, RNG and cloned output lifetime."""
import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
from morphmimic.train.cuda_graphs import CapturedBackward


def test_shared_pool_alternating_graphs_gradients_and_old_outputs():
    device=torch.device('cuda:0');pool=torch.cuda.graph_pool_handle();stream=torch.cuda.Stream(device=device)
    p=torch.nn.Parameter(torch.tensor([2.,3.],device=device))
    q=torch.nn.Parameter(torch.tensor([4.,5.],device=device))
    def first(x):
        hidden=torch.sin((p*x).repeat(256,1))
        return hidden.square().sum(),p*x
    def second(x):
        hidden=torch.cos((q*x).repeat(384,1))
        return hidden.square().sum(),q*x
    a=CapturedBackward(first,[p],pool=pool,stream=stream)
    b=CapturedBackward(second,[q],pool=pool,stream=stream)
    held=[]
    for stage,param,fn,xvalue in [(a,p,first,1.),(b,q,second,2.),(a,p,first,3.),(a,p,first,4.),(b,q,second,5.),(a,p,first,6.)]:
        x=torch.full((2,),xvalue,device=device)
        eager=param.detach().clone().requires_grad_(True)
        hidden=(torch.sin((eager*x).repeat(256,1)) if stage is a else torch.cos((eager*x).repeat(384,1)))
        expected=(hidden.square().sum(),eager*x);expected[0].backward();gradient=eager.grad.clone()
        param.grad=None;loss,out=stage(x)
        torch.testing.assert_close(loss,expected[0]);torch.testing.assert_close(param.grad,gradient)
        torch.testing.assert_close(out,param*x)
        held.append((out,out.clone()))
        for old,snapshot in held:torch.testing.assert_close(old,snapshot)


def test_shared_pool_randomness_advances_without_warm_capture_consumption():
    device=torch.device('cuda:0');pool=torch.cuda.graph_pool_handle();stream=torch.cuda.Stream(device=device)
    p=torch.nn.Parameter(torch.ones(8,device=device));q=torch.nn.Parameter(torch.ones(8,device=device))
    def make(param):
        def fn():
            sample=torch.rand_like(param)
            return (param*sample).sum(),sample
        return fn
    torch.cuda.manual_seed(428)
    a=CapturedBackward(make(p),[p],pool=pool,stream=stream)
    b=CapturedBackward(make(q),[q],pool=pool,stream=stream)
    # Warmup/capture themselves must preserve the generator state; replay advances it.
    before=torch.cuda.get_rng_state().clone();a._capture(())
    assert torch.equal(before,torch.cuda.get_rng_state())
    b._capture(());assert torch.equal(before,torch.cuda.get_rng_state())
    first=a()[1];middle=b()[1];last=a()[1]
    assert not torch.equal(first,last) and not torch.equal(first,middle)
    torch.testing.assert_close(p.grad,last);torch.testing.assert_close(q.grad,middle)


def test_pool_without_common_stream_is_rejected():
    p=torch.nn.Parameter(torch.ones(1,device='cuda'))
    try:CapturedBackward(lambda: p.sum(),[p],pool=torch.cuda.graph_pool_handle())
    except ValueError:pass
    else:raise AssertionError('shared pools require an explicit shared stream')
