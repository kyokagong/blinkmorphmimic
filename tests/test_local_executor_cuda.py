"""Small real CUDA graph differential; run only after GPU scheduling."""
import copy
import pytest
import torch
from torch import nn
from morphmimic.train.update_phases import UpdatePhase,run_equal_phases,share_parameters
from morphmimic.train.cuda_graphs import CapturedBackward,CapturedOptimizerStep
from morphmimic.train.imagination import guarded_clip_grad_norm_

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires explicitly scheduled GPU')


def test_three_robot_shared_pool_backward_and_optimizer_match_eager():
    class Body(nn.Module):
        def __init__(self,scale):
            super().__init__();self.net=nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Linear(4,1));self.register_buffer('scale',torch.tensor(scale,device='cuda'))
        def forward(self,x):return self.net(x)*self.scale
    torch.manual_seed(37)
    bodies=[Body(i+1).cuda() for i in range(3)]
    for b in bodies[1:]:share_parameters(b,bodies[0])
    refs=[copy.deepcopy(b) for b in bodies]
    for b in refs[1:]:share_parameters(b,refs[0])
    opt=torch.optim.Adam(bodies[0].parameters(),lr=torch.tensor(.002,device='cuda'),capturable=True)
    reference_opt=torch.optim.Adam(refs[0].parameters(),lr=.002,capturable=True)
    batches=[torch.randn(n,2,device='cuda') for n in (3,7,9)]
    pool=torch.cuda.graph_pool_handle();stream=torch.cuda.Stream()
    graphs=[CapturedBackward(lambda x,b=b:b(x).square().mean(),b.parameters(),pool=pool,stream=stream) for b in bodies]
    params=tuple(bodies[0].parameters())
    optimizer=CapturedOptimizerStep(opt,params,lambda:guarded_clip_grad_norm_(params,1e6))
    saved=[]
    def phases(index):
        for _ in range(3):
            opt.zero_grad(set_to_none=False)
            loss=graphs[index](batches[index]);saved.append(loss.clone())
            yield UpdatePhase('actor',params,opt,optimizer.step)
    run_equal_phases([phases(i) for i in range(3)])
    for _ in range(3):
        reference_opt.zero_grad();sum(b(x).square().mean()/3 for b,x in zip(refs,batches)).backward();reference_opt.step()
    for p,q in zip(bodies[0].parameters(),refs[0].parameters()):
        torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)
        for name in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(opt.state[p][name],reference_opt.state[q][name],rtol=2e-5,atol=2e-6)
    assert len(saved)==9 and all(torch.isfinite(v) for v in saved)


def test_local_backward_graph_randomness_advances_without_changing_parameters():
    p=nn.Parameter(torch.ones(8,device='cuda'))
    graph=CapturedBackward(lambda:(p*torch.rand_like(p)).square().mean(),[p])
    before=p.detach().clone();a=graph();b=graph()
    assert not torch.equal(a,b)
    torch.testing.assert_close(p,before,rtol=0,atol=0)
