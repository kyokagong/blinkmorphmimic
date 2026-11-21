import copy
import io
import unittest
import torch
from torch import nn


class OptimizerRuntimeTests(unittest.TestCase):
    def test_cpu_adam_restores_flags_step_and_learning_rate(self):
        from morphmimic.train.optim import restore_optimizer_runtime
        p=nn.Parameter(torch.tensor([1.,2.]))
        opt=torch.optim.Adam([p],lr=.012,fused=False)
        p.grad=torch.ones_like(p);opt.step()
        opt.param_groups[0]['fused']=True
        opt.param_groups[0]['capturable']=True
        restore_optimizer_runtime(opt,fused=False,capturable=False)
        self.assertIs(opt.param_groups[0]['fused'],False)
        self.assertIs(opt.param_groups[0]['capturable'],False)
        self.assertEqual(opt.state[p]['step'].device.type,'cpu')
        self.assertEqual(opt.state[p]['step'].item(),1.)
        self.assertEqual(opt.param_groups[0]['lr'],.012)

    def test_cpu_device_rejected_for_graph_capture(self):
        from morphmimic.train.optim import restore_optimizer_runtime
        p=nn.Parameter(torch.ones(1));opt=torch.optim.Adam([p])
        with self.assertRaisesRegex(ValueError,'CUDA'):
            restore_optimizer_runtime(opt,fused=False,capturable=True)


@unittest.skipUnless(torch.cuda.is_available() and hasattr(torch.optim,'Muon'),'CUDA and pinned Muon required')
class ResumeGraphTests(unittest.TestCase):
    def test_cpu_checkpoint_resumes_muon_and_adam_with_changing_lr(self):
        from morphmimic.train.optim import MuonAdam,restore_optimizer_runtime
        from morphmimic.train.cuda_graphs import CapturedOptimizerStep
        for kind in ('adam','muon'):
            with self.subTest(kind=kind):
                torch.manual_seed(13)
                a=nn.Sequential(nn.Linear(5,6),nn.Linear(6,6),nn.Linear(6,2)).cuda()
                b=copy.deepcopy(a)
                def make(model):
                    if kind=='muon':return MuonAdam(model.parameters(),[model],lr=.001,betas=(.9,.999),fused=False)
                    return torch.optim.Adam(model.parameters(),lr=.001,fused=False)
                eager=make(a)
                for p in a.parameters():p.grad=torch.full_like(p,.1)
                eager.step()
                b.load_state_dict(a.state_dict())
                for group in eager.param_groups:group['lr']=0.
                if kind=='muon':
                    for group in eager.muon.param_groups:group.update(fused=True,capturable=True)
                data=io.BytesIO();torch.save(eager.state_dict(),data);data.seek(0)
                loaded=torch.load(data,map_location='cpu',weights_only=False)
                resumed=make(b);resumed.load_state_dict(loaded)
                restore_optimizer_runtime(resumed,fused=False,capturable=True)
                for group in resumed.param_groups:
                    self.assertEqual(group['lr'].device.type,'cuda')
                    self.assertEqual(group['lr'].item(),0.)
                adam=resumed.adam if kind=='muon' else resumed
                self.assertTrue(all(not g['fused'] and g['capturable'] for g in adam.param_groups))
                self.assertTrue(all(s['step'].device.type=='cuda' for s in adam.state.values()))
                if kind=='muon':
                    self.assertTrue(all('fused' not in g and 'capturable' not in g for g in resumed.muon.param_groups))
                graph=CapturedOptimizerStep(resumed,b.parameters(),lambda:None)
                for lr in (.001,0.,.002):
                    for g in eager.param_groups:g['lr']=lr
                    for g in resumed.param_groups:g['lr'].fill_(lr)
                    for x,y in zip(a.parameters(),b.parameters()):
                        x.grad=torch.randn_like(x);y.grad=x.grad.clone()
                    eager.step();graph.step()
                    for x,y in zip(a.parameters(),b.parameters()):
                        torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-5)
                        for key in eager.state[x]:
                            torch.testing.assert_close(eager.state[x][key],resumed.state[y][key],atol=3e-6,rtol=3e-5,check_device=False)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RuntimeModeTests(unittest.TestCase):
    def test_loaded_adam_step_device_tracks_fused_or_capturable(self):
        from morphmimic.train.optim import restore_optimizer_runtime
        for fused,capturable in ((False,False),(True,False),(False,True),(True,True)):
            with self.subTest(fused=fused,capturable=capturable):
                p=nn.Parameter(torch.ones(2,device='cuda'))
                source=torch.optim.Adam([p],lr=.025,fused=False)
                p.grad=torch.ones_like(p);source.step()
                stream=io.BytesIO();torch.save(source.state_dict(),stream);stream.seek(0)
                opt=torch.optim.Adam([p],lr=.1,fused=False);opt.load_state_dict(torch.load(stream,map_location='cpu',weights_only=False))
                restore_optimizer_runtime(opt,fused=fused,capturable=capturable)
                g=opt.param_groups[0]
                self.assertEqual((g['fused'],g['capturable']),(fused,capturable))
                self.assertEqual(opt.state[p]['step'].device.type,'cuda' if fused or capturable else 'cpu')
                if capturable:self.assertEqual(g['lr'].device.type,'cuda')
                self.assertAlmostEqual(float(g['lr']),.025,places=6)
                p.grad=torch.ones_like(p);opt.step()
                self.assertEqual(opt.state[p]['step'].item(),2.)
