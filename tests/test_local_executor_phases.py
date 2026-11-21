"""CPU algebra and ownership tests; no simulation or GPU dependency."""
import copy
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from morphmimic.train.update_phases import UpdatePhase, run_equal_phases, run_phases, share_parameters
from morphmimic.train.normalizers import RunningNormalizer

class LocalPhaseTests(unittest.TestCase):
    def test_equal_robot_gradients_and_one_adam_step_for_unequal_batches(self):
        torch.manual_seed(51)
        owner=nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Linear(4,1))
        reference=copy.deepcopy(owner)
        models=[owner,copy.deepcopy(owner),copy.deepcopy(owner)]
        for m in models[1:]:share_parameters(m,owner)
        batches=[(torch.randn(n,2),torch.randn(n,1)) for n in (2,7,3)]
        opt=torch.optim.Adam(owner.parameters(),lr=.01)
        refopt=torch.optim.Adam(reference.parameters(),lr=.01)
        observed=[]
        def phases(m,x,y):
            for i in range(3):
                opt.zero_grad(set_to_none=True)
                loss=(m(x)-y).square().mean();loss.backward()
                result=yield UpdatePhase('actor',tuple(m.parameters()),opt,lambda: (opt.step(),17)[1])
                observed.append((i,result))
            return 3
        gens=[phases(m,*b) for m,b in zip(models,batches)]
        self.assertEqual(run_equal_phases(gens),[3]*3)
        for _ in range(3):
            refopt.zero_grad();sum((reference(x)-y).square().mean()/3 for x,y in batches).backward();refopt.step()
        for a,b in zip(owner.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
            for k in ('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(opt.state[a][k],refopt.state[b][k])
        self.assertEqual(observed,[(i,17) for i in range(3) for _ in range(3)])

    @unittest.skipUnless(hasattr(torch.optim,'Muon'), 'requires pinned Torch Muon')
    def test_muon_equal_gradient_update_matches_joint_loss(self):
        from morphmimic.train.optim import make_optimizer
        torch.manual_seed(7)
        model=nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Linear(4,4),nn.Tanh(),nn.Linear(4,1))
        reference=copy.deepcopy(model)
        opt=make_optimizer(model.parameters(),[model],kind='muon',lr=.002)
        refopt=make_optimizer(reference.parameters(),[reference],kind='muon',lr=.002)
        batches=[torch.randn(n,2) for n in (3,7,9)]
        def phases(x):
            for _ in range(3):
                opt.zero_grad();model(x).square().mean().backward()
                yield UpdatePhase('wm',tuple(model.parameters()),opt,opt.step)
        run_equal_phases([phases(x) for x in batches])
        for _ in range(3):
            refopt.zero_grad();sum(reference(x).square().mean()/3 for x in batches).backward();refopt.step()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-6)
            for key in opt.state[a]:torch.testing.assert_close(opt.state[a][key],refopt.state[b][key],rtol=2e-4,atol=2e-6)

    def test_external_accumulation_survives_zeroed_reused_grad_storage(self):
        p=nn.Parameter(torch.tensor(2.));p.grad=torch.zeros_like(p)
        opt=torch.optim.SGD([p],lr=.1)
        def one(g):
            p.grad.zero_();p.grad.add_(g)
            yield UpdatePhase('wm',(p,),opt,opt.step)
        run_equal_phases([one(1.),one(2.),one(9.)])
        torch.testing.assert_close(p.detach(),torch.tensor(1.6))

    def test_nonfinite_not_sanitized_before_global_guard(self):
        p=nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1)
        captured=[]
        def guard():captured.append(p.grad.clone());p.grad.zero_();opt.step();return 'bad'
        def one(g):
            p.grad=torch.tensor(g)
            result=yield UpdatePhase('wm',(p,),opt,guard)
            return result
        self.assertEqual(run_equal_phases([one(1.),one(float('nan')),one(9.)]),['bad']*3)
        self.assertTrue(torch.isnan(captured[0]));self.assertEqual(float(p),1.)

    def test_ownership_requires_same_parameter_names_and_leaves_buffers_local(self):
        class Robot(nn.Module):
            def __init__(self,n):
                super().__init__();self.net=nn.Linear(2,1);self.register_buffer('morph',torch.tensor(n))
        a,b=Robot(1),Robot(2);share_parameters(b,a)
        self.assertIs(a.net.weight,b.net.weight);self.assertIsNot(a.morph,b.morph)
        self.assertEqual(int(b.morph),2)
        with self.assertRaises(ValueError):share_parameters(nn.Linear(3,1),a.net)

    def test_ema_interpolates_shared_parameters_once_and_copies_local_buffers(self):
        from morphmimic.train.optim import PolicyEMA
        class Robot(nn.Module):
            def __init__(self,n):
                super().__init__();self.w=nn.Parameter(torch.tensor(0.));self.register_buffer('morph',torch.tensor(n))
        robots=[Robot(i) for i in range(3)]
        for m in robots[1:]:share_parameters(m,robots[0])
        avgs=[PolicyEMA(m,.9) for m in robots]
        for avg in avgs[1:]:share_parameters(avg.module,avgs[0].module)
        with torch.no_grad():robots[0].w.fill_(10.)
        avgs[0].update_parameters(robots[0])
        for avg,m in zip(avgs,robots):avg.copy_buffers(m)
        for i,avg in enumerate(avgs):
            self.assertEqual(float(avg.module.w),1.);self.assertEqual(int(avg.module.morph),i)
        self.assertEqual(avgs[0].updates,1)

    def test_stage_mismatch_rejected_before_any_optimizer_step(self):
        p=nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1)
        def one(name):
            p.grad=torch.tensor(1.)
            yield UpdatePhase(name,(p,),opt,opt.step)
        with self.assertRaisesRegex(ValueError,'phase'):run_equal_phases([one('wm'),one('actor')])
        self.assertEqual(float(p),1.)

    def test_normalizer_union_is_count_weighted_and_empty_safe(self):
        torch.manual_seed(1);norm=RunningNormalizer(3);ref=RunningNormalizer(3)
        history=torch.randn(9,3);norm.update(history);ref.update(history)
        batches=[torch.randn(2,3)+4,torch.empty(0,3),torch.randn(19,3)-2]
        norm.update_many(batches);ref.update(torch.cat(batches))
        for k in ('mean','var','count'):torch.testing.assert_close(getattr(norm,k),getattr(ref,k))
        old=copy.deepcopy(norm.state_dict());norm.update_many([torch.empty(0,3)]*3)
        for k,v in old.items():torch.testing.assert_close(norm.state_dict()[k],v)

if __name__=='__main__':unittest.main()
