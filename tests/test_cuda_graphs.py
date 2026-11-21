import unittest
import torch

class TreeTests(unittest.TestCase):
    def test_static_copy_checks_tree_structure_shape_and_dtype(self):
        from morphmimic.train.cuda_graphs import clone_tree, copy_tree
        x={'a':torch.tensor([1.,2.]),'b':(torch.ones(1,dtype=torch.long),)}
        static=clone_tree(x)
        copy_tree(static,{'a':torch.tensor([3.,4.]),'b':(torch.zeros(1,dtype=torch.long),)})
        self.assertTrue(torch.equal(static['a'],torch.tensor([3.,4.])))
        with self.assertRaises(ValueError):copy_tree(static,{'a':torch.ones(1),'b':(torch.zeros(1,dtype=torch.long),)})
        with self.assertRaises(ValueError):copy_tree(static,{'a':torch.ones(2),'b':(torch.zeros(1),)})
        with self.assertRaises(ValueError):copy_tree(static,{'a':torch.ones(2)})

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GraphTests(unittest.TestCase):
    def test_backward_refreshes_inputs_preserves_rng_and_output_lifetime(self):
        from morphmimic.train.cuda_graphs import CapturedBackward
        p=torch.nn.Parameter(torch.tensor([2.],device='cuda'))
        graph=CapturedBackward(lambda x: ((p*x).square().sum(),(p*x).clone()),[p])
        x=torch.tensor([3.],device='cuda');state=torch.cuda.get_rng_state().clone()
        loss,out=graph(x)
        self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))
        self.assertEqual(loss.item(),36.)
        self.assertEqual(p.grad.item(),36.)
        graph(torch.tensor([4.],device='cuda'))
        self.assertEqual(out.item(),6.)
        self.assertEqual(p.grad.item(),64.)

    def test_adam_step_matches_eager_and_lr_change(self):
        from morphmimic.train.cuda_graphs import CapturedOptimizerStep
        a=torch.nn.Parameter(torch.tensor([1.,2.],device='cuda'));b=torch.nn.Parameter(a.detach().clone())
        ga=torch.optim.Adam([a],lr=torch.tensor(.01,device='cuda'),capturable=True)
        gb=torch.optim.Adam([b],lr=torch.tensor(.01,device='cuda'),capturable=True)
        staged=CapturedOptimizerStep(ga,[a],lambda:(a.grad.norm(),torch.isfinite(a.grad).all()))
        for lr in (.01,.02,.005,0.):
            ga.param_groups[0]['lr'].fill_(lr);gb.param_groups[0]['lr'].fill_(lr)
            a.grad=torch.tensor([.2,-.3],device='cuda');b.grad=a.grad.clone()
            norm,finite=staged.step();gb.step()
            self.assertTrue(finite.item());self.assertAlmostEqual(norm.item(),(.2**2+.3**2)**.5,places=6)
            torch.testing.assert_close(a,b)
            for k in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(ga.state[a][k],gb.state[b][k])

class GraphGuards(unittest.TestCase):
    def test_non_cuda_backward_rejected(self):
        from morphmimic.train.cuda_graphs import CapturedBackward
        p=torch.nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError,'CUDA'):
            CapturedBackward(lambda x:(p*x).sum(),[p])

@unittest.skipUnless(torch.cuda.is_available() and hasattr(torch.optim,'Muon'),'CUDA and pinned Muon required')
class MuonGraphTests(unittest.TestCase):
    def test_muon_adam_matches_three_eager_steps_and_lr_change(self):
        from morphmimic.train.cuda_graphs import CapturedOptimizerStep
        from morphmimic.train.optim import MuonAdam
        torch.manual_seed(2)
        left=torch.nn.Sequential(torch.nn.Linear(5,6),torch.nn.Linear(6,6),torch.nn.Linear(6,2)).cuda()
        import copy
        right=copy.deepcopy(left)
        a=MuonAdam(left.parameters(),[left],lr=.001,betas=(.9,.999),fused=True)
        b=MuonAdam(right.parameters(),[right],lr=.001,betas=(.9,.999),fused=True)
        for opt in (a,b):
            for g in opt.param_groups:g['lr']=torch.tensor(.001,device='cuda')
            for g in opt.adam.param_groups:g['capturable']=True
        stage=CapturedOptimizerStep(a,left.parameters(),lambda:None)
        for lr in (.001,.002,.0005,0.):
            for opt in (a,b):
                for g in opt.param_groups:g['lr'].fill_(lr)
            for x,y in zip(left.parameters(),right.parameters()):
                x.grad=torch.randn_like(x);y.grad=x.grad.clone()
            stage.step();b.step()
            for x,y in zip(left.parameters(),right.parameters()):
                torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
                for key in a.state[x]:torch.testing.assert_close(a.state[x][key],b.state[y][key],atol=2e-6,rtol=2e-5)

@unittest.skipUnless(torch.cuda.is_available() and hasattr(torch.optim,'Muon'),'CUDA and pinned Muon required')
class NonfiniteGraphTests(unittest.TestCase):
    def test_nonfinite_guard_keeps_muon_adam_parameters_and_moments_finite(self):
        from morphmimic.train.cuda_graphs import CapturedOptimizerStep
        from morphmimic.train.imagination import guarded_clip_grad_norm_
        from morphmimic.train.optim import MuonAdam
        net=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,4),torch.nn.Linear(4,2)).cuda()
        params=list(net.parameters());opt=MuonAdam(params,[net],lr=.001,betas=(.9,.999),fused=True)
        for group in opt.param_groups:group['lr']=torch.tensor(.001,device='cuda')
        for group in opt.adam.param_groups:group['capturable']=True
        for p in params:p.grad=torch.ones_like(p)
        params[0].grad.fill_(float('nan'))
        stage=CapturedOptimizerStep(opt,params,lambda:guarded_clip_grad_norm_(params,1.))
        _,bad=stage.step()
        self.assertEqual(bad.item(),1)
        for p in params:
            self.assertTrue(torch.isfinite(p).all().item())
            for value in opt.state[p].values():
                if isinstance(value,torch.Tensor):self.assertTrue(torch.isfinite(value).all().item())

class OptionalTreeTests(unittest.TestCase):
    def test_none_tree_leaves_preserved_and_other_scalars_rejected(self):
        from collections import namedtuple
        from morphmimic.train.cuda_graphs import clone_tree,copy_tree
        Result=namedtuple('Result','value bootstrap recon_err')
        tree=Result(torch.tensor([1.]),None,None)
        out=clone_tree(tree);copy_tree(out,Result(torch.tensor([3.]),None,None))
        self.assertEqual(out.value.item(),3.)
        self.assertIsNone(out.bootstrap)
        with self.assertRaises(ValueError):copy_tree(out,Result(torch.tensor([1.]),None,2))
        with self.assertRaises(ValueError):clone_tree(Result(torch.tensor([1.]),None,2))

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class PersistentGradientTests(unittest.TestCase):
    def test_zero_grad_to_none_between_captured_replays(self):
        from collections import namedtuple
        from morphmimic.train.cuda_graphs import CapturedBackward
        Result=namedtuple('Result','loss value bootstrap recon_err')
        p=torch.nn.Parameter(torch.tensor([2.],device='cuda'))
        def fn(x):return Result((p*x).square().sum(),p*x,None,None)
        graph=CapturedBackward(fn,[p]);x=torch.tensor([3.],device='cuda')
        result=graph(x)
        self.assertIsNone(result.bootstrap)
        self.assertEqual(p.grad.item(),36.)
        p.grad=None
        graph(torch.tensor([4.],device='cuda'))
        self.assertIsNotNone(p.grad)
        self.assertEqual(p.grad.item(),64.)
