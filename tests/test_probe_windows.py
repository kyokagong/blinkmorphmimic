import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

try:
    from morphmimic.env.commands import MultiMotionCommand
    from morphmimic.eval import benchmark
    from morphmimic.constants import GOAL_DIM,HUMAN_ENV_DIM,PROPRIO_DIM,R_MAX
except ImportError:
    MultiMotionCommand=None

@unittest.skipIf(MultiMotionCommand is None,'project runtime required')
class WindowTests(unittest.TestCase):
    def test_explicit_clip_local_start_and_bounds(self):
        lib=SimpleNamespace(clip_start=torch.tensor([0,10]),clip_len=torch.tensor([10,20]),
                            frame_bin=lambda frames:frames,clip_end=lambda frames:torch.where(frames<10,9,29))
        command=SimpleNamespace(env_clip=torch.zeros(2,dtype=torch.long),env_clip_start=torch.zeros(2,dtype=torch.long),
                                device='cpu',library=lib,cfg=SimpleNamespace(sampling='start',failure_reset=0),num_envs=2)
        ids=torch.tensor([0,1]);clips=torch.tensor([0,1]);starts=torch.tensor([3,7])
        MultiMotionCommand.assign_clips(command,ids,clips,starts)
        frames,_=MultiMotionCommand._sample_start(command,ids)
        torch.testing.assert_close(frames,torch.tensor([3,17]))
        with self.assertRaises(ValueError):MultiMotionCommand.assign_clips(command,ids,clips,torch.tensor([9,0]))
        command.cfg.failure_reset=.1
        with self.assertRaises(ValueError):MultiMotionCommand.assign_clips(command,ids,clips,starts)
        command.cfg.failure_reset=0
        MultiMotionCommand.assign_clips(command,ids,clips)
        torch.testing.assert_close(command.env_clip_start,torch.zeros(2,dtype=torch.long))

    def test_window_completion_failure_mapping_and_partial_reset(self):
        lib=SimpleNamespace(has_human=False,num_clips=3,clip_names=['a','b','c'],
            clip_len=torch.tensor([20,20,20]),clip_start=torch.tensor([0,20,40]),
            source_has_phases=torch.tensor([False,False,False]),source_start=torch.tensor([0,0,0]),
            source_stop=torch.tensor([20,20,20]),frame_clip=torch.repeat_interleave(torch.arange(3),20))
        command=MultiMotionCommand.__new__(MultiMotionCommand);command.library=lib
        command.cfg=SimpleNamespace(sampling='start');command.global_frame=torch.zeros(2,dtype=torch.long)
        command.env_clip=torch.zeros(2,dtype=torch.long);local_start=torch.zeros(2,dtype=torch.long);resets=[]
        def assign(ids,clips,starts=None):
            command.env_clip[ids]=clips;local_start[ids]=0 if starts is None else starts
        command.assign_clips=assign
        obs={'policy':torch.zeros(2,PROPRIO_DIM),'goal':torch.zeros(2,GOAL_DIM),'goal_kp':torch.zeros(2,HUMAN_ENV_DIM)}
        def reset(env_ids):
            resets.append(env_ids.tolist());command.global_frame[env_ids]=lib.clip_start[command.env_clip[env_ids]]+local_start[env_ids]+1
            return obs,{}
        def step(action):
            local=command.global_frame-lib.clip_start[command.env_clip]
            terminated=(command.env_clip==1)&(local==6);truncated=local==19
            command.global_frame+=1
            return obs,torch.where(command.env_clip==1,R_MAX*.5,R_MAX),terminated,truncated,{}
        env=SimpleNamespace(device='cpu',num_envs=2,step_dt=.02,reset=reset,step=step,
                            command_manager=SimpleNamespace(get_term=lambda _:command))
        trace=SimpleNamespace(record=lambda _:None,take=lambda ids,steps:[(0,)]*len(ids))
        with patch.object(benchmark,'DriftTrace',return_value=trace),patch.object(benchmark,'summarize_drift',return_value={k:0. for k in benchmark.DRIFT_SUMMARY_COLUMNS}):
            result=benchmark.run_benchmark(env,lambda history,goals:torch.zeros(2,29),clip_ids=[0,1,2],windows=[(2,5),(4,8),(1,3)])
        self.assertEqual(result['steps'].tolist(),[3,2,2])
        self.assertEqual(result['fail_frame'].tolist(),[-1,6,-1])
        self.assertEqual(result['window_completed'].tolist(),[True,False,True])
        self.assertEqual(result['window_start_frame'].tolist(),[2,4,1])
        self.assertEqual(result['window_end_frame'].tolist(),[5,8,3])
        self.assertNotIn('norm_return',result.columns)
        torch.testing.assert_close(torch.tensor(result['window_norm_return'].tolist()),torch.tensor([1.,.5,1.]))
        self.assertTrue(any(len(ids)==1 for ids in resets[1:]))
        with patch.object(benchmark,'DriftTrace',return_value=trace):
            with self.assertRaises(TimeoutError):
                benchmark.run_benchmark(env,lambda history,goals:torch.zeros(2,29),clip_ids=[0],windows=[(2,5)],deadline=0.)

    def test_window_bounds_reject_out_of_range_and_unselected(self):
        lib=SimpleNamespace(clip_len=torch.tensor([10,20]))
        self.assertEqual(benchmark.validate_windows(lib,[0,1],[(2,9),(0,19)]),[(2,9),(0,19)])
        for bounds in ([(0,10),(0,19)],[(2,2),(0,19)],[(0,1)],[(True,2),(0,19)]):
            with self.assertRaises(ValueError):benchmark.validate_windows(lib,[0,1],bounds)
