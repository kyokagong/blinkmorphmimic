import unittest
import torch
from morphmimic.train.replay import ReplayBuffer, ReplayCfg


def fixture():
    b=ReplayBuffer(ReplayCfg(num_envs=2,capacity=6,device='cpu'))
    b.ptr[:]=torch.tensor([5,2]);b.size[:]=torch.tensor([5,6]);b.valid[:]=True
    # env0 has episode 0,1,2 then a terminal row 3, then reset 0.
    b.ep_step[0,:5]=torch.tensor([0,1,2,3,0]);b.no_action[0,3]=True
    # env1 has wrapped; chronological slots 2,3,4,5,0,1 with overwritten history.
    b.ep_step[1]=torch.tensor([6,7,2,3,4,5]);b.no_action[1,1]=True
    for e in range(2):
        for s in range(6):
            b.frame[e,s]=100*e+s
            b.obs[e,s]=b.frame[e,s]
            b.state[e,s]=b.frame[e,s]
    return b

class FixedReplayTests(unittest.TestCase):
    def test_windows_obey_wrap_terminals_and_range(self):
        from morphmimic.train.fixed_replay import GraphReplaySampler
        s=GraphReplaySampler(fixture());torch.manual_seed(10)
        w=s.windows(3000,2,env_range=(0,1))
        self.assertEqual(set(w.frame[:,0].tolist()),{0,1})
        self.assertTrue(torch.equal(w.frame[:,1],w.frame[:,0]+1))
        self.assertTrue(torch.equal(w.frame[:,2],w.frame[:,0]+2))
        self.assertGreater((w.frame[:,0]==0).sum().item(),1200)
        self.assertLess((w.frame[:,0]==0).sum().item(),1800)
    def test_uniform_across_all_valid_rows_not_uniform_per_environment(self):
        from morphmimic.train.fixed_replay import GraphReplaySampler
        torch.manual_seed(91);s=GraphReplaySampler(fixture())
        frame=s.windows(12000,2).frame[:,0]
        self.assertEqual(set(frame.tolist()),{0,1,102,103,104,105})
        for value in (0,1,102,103,104,105):
            count=(frame==value).sum().item()
            self.assertGreater(count,1750);self.assertLess(count,2250)
    def test_starts_history_accept_filter_and_rng_advance(self):
        from morphmimic.train.fixed_replay import GraphReplaySampler
        s=GraphReplaySampler(fixture());accept=lambda x:x<100
        a=s.starts(128,3,accept_frame=accept);b=s.starts(128,3,accept_frame=accept)
        self.assertFalse(torch.equal(a.frame,b.frame))
        self.assertTrue((a.frame<100).all())
        for frame,hist in zip(a.frame.tolist(),a.obs_hist[:,:,0].tolist()):
            self.assertEqual(hist,[float(max(0,frame-2)),float(max(0,frame-1)),float(frame)] if frame<3 else [4.,4.,4.])
    def test_overwritten_histories_are_never_selected(self):
        from morphmimic.train.fixed_replay import GraphReplaySampler
        s=GraphReplaySampler(fixture());torch.manual_seed(40)
        result=s.starts(2000,3)
        self.assertEqual(set(result.frame.tolist()),{0,1,2,4,100,104,105})
        self.assertTrue((result.obs_hist[:,0,0]<=result.obs_hist[:,2,0]).logical_or(result.frame>=100).all().item())
    def test_empty_rejected_outside_graph(self):
        from morphmimic.train.fixed_replay import GraphReplaySampler
        b=ReplayBuffer(ReplayCfg(num_envs=1,capacity=5,device='cpu'))
        s=GraphReplaySampler(b)
        with self.assertRaisesRegex(RuntimeError,'no valid'):s.windows(2,2)
        with self.assertRaisesRegex(RuntimeError,'no valid'):s.starts(2,2)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class FixedReplayCudaTests(unittest.TestCase):
    def test_graph_replay_draws_fresh_valid_samples(self):
        from morphmimic.train.fixed_replay import GraphReplaySampler
        b=fixture();b.device=torch.device('cuda')
        for key,value in vars(b).items():
            if isinstance(value,torch.Tensor):setattr(b,key,value.cuda())
        s=GraphReplaySampler(b);s.check_windows(2,(0,1));s.check_starts(3,lambda f:f<100)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):s.windows(256,2,(0,1));s.starts(256,3,lambda f:f<100)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            windows=s.windows(256,2,(0,1));starts=s.starts(256,3,lambda f:f<100)
        torch.cuda.current_stream().wait_stream(stream)
        s.check_windows(2,(0,1));s.check_starts(3,lambda f:f<100)
        graph.replay();first=windows.frame.clone();first_starts=starts.frame.clone()
        graph.replay();second=windows.frame.clone();second_starts=starts.frame.clone()
        self.assertFalse(torch.equal(first,second))
        self.assertFalse(torch.equal(first_starts,second_starts))
        self.assertTrue(((second[:,0]==0)|(second[:,0]==1)).all().item())
        self.assertTrue((second_starts<100).all().item())
        b.no_action[0,0]=True
        s.check_windows(2,(0,1));s.check_starts(3,lambda f:f<100)
        graph.replay()
        self.assertTrue((windows.frame[:,0]==1).all().item())
