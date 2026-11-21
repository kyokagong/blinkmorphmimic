"""Fixed-shape replay writes must preserve every nonterminal environment."""
import unittest
import torch
from morphmimic.constants import ACTION_DIM,CTX_DIM,PROPRIO_DIM,STATE_DIM
from morphmimic.train.replay import ReplayBuffer,ReplayCfg


class MaskedTerminalTests(unittest.TestCase):
    def test_none_partial_and_all_match_dynamic_terminal_writes_after_wrap(self):
        torch.manual_seed(9)
        for mask in (torch.tensor([False,False,False]),torch.tensor([True,False,True]),torch.tensor([True,True,True])):
            with self.subTest(mask=mask.tolist()):
                a=ReplayBuffer(ReplayCfg(num_envs=3,capacity=2));b=ReplayBuffer(ReplayCfg(num_envs=3,capacity=2))
                for step in range(3):
                    values=dict(state=torch.randn(3,STATE_DIM),obs=torch.randn(3,PROPRIO_DIM),
                        action=torch.randn(3,ACTION_DIM),reward=torch.randn(3),terminated=torch.tensor([False,True,False]),
                        timeout=torch.zeros(3,dtype=torch.bool),ctx=torch.randn(3,CTX_DIM),frame=torch.arange(3),ep_step=torch.full((3,),step))
                    a.add_step(**values);b.add_step(**values)
                terminal=dict(state=torch.randn(3,STATE_DIM),obs=torch.randn(3,PROPRIO_DIM),ctx=torch.randn(3,CTX_DIM),
                    frame=torch.arange(3),ep_step=torch.tensor([2,3,4]))
                ids=mask.nonzero().flatten()
                a.add_terminal_rows(ids,**{key:value[ids] for key,value in terminal.items()})
                b.add_terminal_rows_masked(mask,**terminal)
                for name in ('state','obs','action','reward','terminated','timeout','ctx','frame','ep_step','no_action','valid','ptr','size'):
                    torch.testing.assert_close(getattr(a,name),getattr(b,name),msg=name)

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_COLLECTOR_MIXED_SUBSET'), 'requires GPU graph fixture')
class MaskedTerminalGpuTests(unittest.TestCase):
    def test_mask_changes_across_graph_replays(self):
        device='cuda:0';n=3
        a=ReplayBuffer(ReplayCfg(num_envs=n,capacity=3,device=device))
        b=ReplayBuffer(ReplayCfg(num_envs=n,capacity=3,device=device))
        terminal=dict(state=torch.randn(n,STATE_DIM,device=device),obs=torch.randn(n,PROPRIO_DIM,device=device),
            ctx=torch.randn(n,CTX_DIM,device=device),frame=torch.arange(n,device=device),ep_step=torch.arange(n,device=device))
        mask=torch.zeros(n,dtype=torch.bool,device=device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): b.add_terminal_rows_masked(mask,**terminal)
        for bits in ([False,False,False],[True,False,True],[True,True,True],[False,True,False]):
            mask.copy_(torch.tensor(bits,device=device));ids=mask.nonzero().flatten()
            a.add_terminal_rows(ids,**{key:value[ids] for key,value in terminal.items()})
            graph.replay()
            for name in ('state','obs','action','reward','terminated','timeout','ctx','frame','ep_step','no_action','valid','ptr','size'):
                torch.testing.assert_close(getattr(a,name),getattr(b,name),msg=name)

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class GraphDelayRingTests(unittest.TestCase):
    def test_ring_wrap_and_partial_reset_match_mjlab(self):
        from mjlab.utils.buffers import CircularBuffer
        from morphmimic.env.graph_delay import GraphCircularBuffer
        old=CircularBuffer(11,3,'cpu');new=GraphCircularBuffer(11,3,'cpu')
        for step in range(30):
            if step in (5,19): old.reset([1]);new.reset([1])
            if step==20:
                data=torch.full((3,2),77.)
                ids=torch.tensor([1]);old.backfill(data,ids);new.backfill(data,ids)
            data=torch.full((3,2),float(step)) + torch.arange(3)[:,None]
            old.append(data);new.append(data)
            lag=torch.tensor([step%11,(step+2)%11,(step+4)%11])
            torch.testing.assert_close(new[lag],old[lag])
            torch.testing.assert_close(new.current_length,old.current_length)
            torch.testing.assert_close(new.buffer,old.buffer)

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class GraphDelayLagTests(unittest.TestCase):
    def test_lag_updates_and_partial_resets_match_runtime(self):
        from mjlab.utils.buffers import DelayBuffer
        from morphmimic.env.graph_delay import GraphDelayBuffer
        old=DelayBuffer(min_lag=0,max_lag=10,batch_size=3,device='cpu',update_period=1000,per_env_phase=False)
        new=GraphDelayBuffer(min_lag=0,max_lag=10,batch_size=3,device='cpu',update_period=1000,per_env_phase=False)
        for step in range(30):
            if step in (5,19): old.reset([1]);new.reset([1])
            data=torch.full((3,2),float(step)) + torch.arange(3)[:,None]
            old.append(data);new.append(data)
            torch.manual_seed(1000+step);a=old.compute()
            torch.manual_seed(1000+step);b=new.compute()
            torch.testing.assert_close(a,b)
            torch.testing.assert_close(old.current_lags,new.current_lags)

class MaskedHistoryTests(unittest.TestCase):
    def test_none_partial_all_match_selected_reset(self):
        from morphmimic.train.obs_history import ObsHistory
        for mask in (torch.tensor([False,False,False]),torch.tensor([True,False,True]),torch.tensor([True,True,True])):
            with self.subTest(mask=mask.tolist()):
                a=ObsHistory(3,4,2,'cpu');b=ObsHistory(3,4,2,'cpu')
                initial=torch.arange(6.).reshape(3,2)
                a.reset(None,initial);b.reset(None,initial)
                a.push(initial+10);b.push(initial+10)
                new=initial+50;ids=mask.nonzero().flatten()
                a.reset(ids,new[ids]);b.reset_masked(mask,new)
                torch.testing.assert_close(a.buf,b.buf)

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_COLLECTOR_MIXED_SUBSET'), 'requires GPU physics fixture')
class MaskedNativeResetTests(unittest.TestCase):
    def test_native_reset_matches_selected_worlds(self):
        import os
        import warp as wp
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        subset=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'];root=os.environ['BLINKMORPH_HUMAN_ROOT'];device='cuda:0'
        def cfg():
            value=morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=True,human_root=root)
            value.commands['motion'].allow_missing_human=True
            return value
        a=MorphMimicEnv(cfg(),device=device)
        stream=torch.cuda.Stream(device=device);b=MorphMimicEnv(cfg(),device=device,graph_stream=stream)
        try:
            for bits in ([False,False],[True,False],[True,True]):
                a.reset();b.reset();action=torch.linspace(-.1,.1,29,device=device).repeat(2,1)
                a.step(action)
                with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
                    b.step_no_reset_tensor(action)
                torch.cuda.synchronize(device)
                mask=torch.tensor(bits,device=device);ids=mask.nonzero().flatten()
                a.sim.reset(ids)
                with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
                    b.reset_native_masked(mask)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(a.sim.data.qpos,b.sim.data.qpos,atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(a.sim.data.qvel,b.sim.data.qvel,atol=1e-4,rtol=1e-4)
        finally:
            a.close();b.close()

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_COLLECTOR_MIXED_SUBSET'), 'requires GPU physics fixture')
class MaskedNativeCaptureTests(unittest.TestCase):
    def test_native_reset_mask_changes_on_graph_replay(self):
        import os,warp as wp
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        subset=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'];root=os.environ['BLINKMORPH_HUMAN_ROOT'];device='cuda:0'
        def cfg():
            value=morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=True,human_root=root)
            value.commands['motion'].allow_missing_human=True
            return value
        eager=MorphMimicEnv(cfg(),device=device)
        stream=torch.cuda.Stream(device=device);captured=MorphMimicEnv(cfg(),device=device,graph_stream=stream)
        try:
            eager.reset();captured.reset()
            mask=torch.zeros(2,dtype=torch.bool,device=device)
            with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
                captured.reset_native_masked(mask)
            torch.cuda.synchronize(device)
            graph=torch.cuda.CUDAGraph();ws=wp.stream_from_torch(stream)
            with wp.ScopedStream(ws,sync_enter=False,sync_exit=False):
                with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
                    wp.capture_begin(stream=ws,external=True)
                    try: captured.reset_native_masked(mask)
                    finally: warp_graph=wp.capture_end(stream=ws)
            for bits in ([False,False],[True,False],[True,True]):
                mask.copy_(torch.tensor(bits,device=device));stream.wait_stream(torch.cuda.current_stream(device))
                eager.sim.reset(mask.nonzero().flatten())
                with torch.cuda.stream(stream): graph.replay()
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.testing.assert_close(eager.sim.data.qpos,captured.sim.data.qpos)
                torch.testing.assert_close(eager.sim.data.qvel,captured.sim.data.qvel)
            self.assertIsNotNone(warp_graph)
        finally:
            eager.close();captured.close()


@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedDelayTests(unittest.TestCase):
    def test_masked_delay_reset_preserves_unselected_history_lags_and_phase(self):
        from morphmimic.env.graph_delay import GraphDelayBuffer
        for mask in (torch.tensor([False,False,False]),torch.tensor([True,False,True]),torch.tensor([True,True,True])):
            with self.subTest(mask=mask.tolist()):
                torch.manual_seed(33);a=GraphDelayBuffer(min_lag=0,max_lag=4,batch_size=3,device='cpu',update_period=13,per_env_phase=True)
                torch.manual_seed(33);b=GraphDelayBuffer(min_lag=0,max_lag=4,batch_size=3,device='cpu',update_period=13,per_env_phase=True)
                for i in range(7):
                    values=torch.arange(6.).reshape(3,2)+i
                    a.append(values);b.append(values)
                    torch.manual_seed(80+i);a.compute()
                    torch.manual_seed(80+i);b.compute()
                ids=mask.nonzero().flatten()
                torch.manual_seed(57);a.reset(ids)
                torch.manual_seed(57);b.reset_masked(mask)
                for name in ('_current_lags','_step_count','_phase_offsets'):
                    torch.testing.assert_close(getattr(a,name),getattr(b,name),msg=name)
                torch.testing.assert_close(a._buffer._buffer,b._buffer._buffer)
                torch.testing.assert_close(a._buffer._num_pushes,b._buffer._num_pushes)

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedReturnTests(unittest.TestCase):
    def test_none_partial_all_preserve_duplicate_bin_ema(self):
        from morphmimic.env.commands import fold_returns,return_sampling_probs
        from morphmimic.env.masked_reset import fold_finished_masked
        g=torch.tensor([.3,.6,.9]);past=torch.tensor([.2,.5,.3]);bins=torch.tensor([0,0,2]);returns=torch.tensor([.4,.8,.2]);finished=torch.tensor([1.,1.,0.])
        for mask in (torch.tensor([False,False,False]),torch.tensor([True,True,False]),torch.tensor([True,True,True])):
            with self.subTest(mask=mask.tolist()):
                ids=mask.nonzero().flatten();expected=g.clone();expected_past=past.clone()
                if len(ids):
                    expected=fold_returns(g,bins[ids],returns[ids],finished[ids],.1)
                    _,soft=return_sampling_probs(expected,past,1.,(.3,.5,.2),None)
                    expected_past=.9*past+.1*soft
                actual,actual_past=fold_finished_masked(g,past,bins,returns,finished,mask,.1,1.,(.3,.5,.2),None)
                torch.testing.assert_close(actual,expected);torch.testing.assert_close(actual_past,expected_past)

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedCandidateTests(unittest.TestCase):
    def test_start_candidates_keep_source_per_environment_and_do_not_mutate(self):
        from types import SimpleNamespace
        from morphmimic.env.masked_reset import sample_start_candidates
        class Library:
            clip_start=torch.tensor([0,100])
            def frame_bin(self,frames): return frames//10
            def clip_end(self,frames): return torch.where(frames<100,torch.tensor(99),torch.tensor(199))
        cmd=SimpleNamespace(num_envs=2,device='cpu',library=Library(),env_clip=torch.tensor([0,1]),
            cfg=SimpleNamespace(sampling='start',failure_reset=0.),consecutive_local=torch.tensor([2,3]),
            restart_frame=torch.tensor([9,19]),global_frame=torch.tensor([10,110]))
        frames,bins,streak,entropy=sample_start_candidates(cmd)
        torch.testing.assert_close(frames,torch.tensor([0,100]))
        torch.testing.assert_close(bins,torch.tensor([0,10]))
        torch.testing.assert_close(streak,torch.tensor([2,3]))
        torch.testing.assert_close(cmd.restart_frame,torch.tensor([9,19]))
        self.assertIsNone(entropy)

    def test_full_batch_return_candidates_match_existing_sampler(self):
        from types import SimpleNamespace
        from morphmimic.env.commands import MultiMotionCommand
        from morphmimic.env.masked_reset import sample_start_candidates
        class Library:
            num_bins=3
            def sample_frames_in_bins(self,bins): return bins*10+(torch.rand(bins.shape)*9).long()
            def frame_bin(self,frames): return frames//10
            def clip_end(self,frames): return torch.full_like(frames,100)
        cfg=SimpleNamespace(sampling='return',failure_reset=0.,softmax_temp=1.,mix=(.5,.4,.1),softmax_cap=None)
        cmd=SimpleNamespace(num_envs=5,device='cpu',library=Library(),cfg=cfg,
            G_b=torch.tensor([.2,.6,.8]),past_softmax=torch.tensor([.4,.3,.3]),
            probe_probabilities=None,probe_rho=0.,consecutive_local=torch.zeros(5,dtype=torch.long),
            global_frame=torch.zeros(5,dtype=torch.long),metrics={'sampling_entropy':torch.zeros(5)})
        torch.manual_seed(12);expected=MultiMotionCommand._sample_start(cmd,torch.arange(5))
        torch.manual_seed(12);frames,bins,_,entropy=sample_start_candidates(cmd)
        torch.testing.assert_close(frames,expected[0]);torch.testing.assert_close(bins,expected[1])
        torch.testing.assert_close(entropy,cmd.metrics['sampling_entropy'][0])

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedReferenceWriteTests(unittest.TestCase):
    def test_selected_native_joints_and_world_root_velocity_only(self):
        from types import SimpleNamespace
        from morphmimic.env.masked_reset import write_reference_masked
        qpos=torch.arange(18.).reshape(2,9);qvel=torch.arange(16.).reshape(2,8)
        oldpos=qpos.clone();oldvel=qvel.clone()
        data=SimpleNamespace(data=SimpleNamespace(qpos=qpos,qvel=qvel),
            soft_joint_pos_limits=torch.tensor([[[-1.,1.],[-2.,2.]]]).expand(2,-1,-1))
        indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7),free_joint_v_adr=torch.arange(6),
            joint_q_adr=torch.tensor([7,8]),joint_v_adr=torch.tensor([6,7]))
        cmd=SimpleNamespace(robot=SimpleNamespace(data=data,indexing=indexing))
        root_pos=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        quat=torch.tensor([[.70710678,0.,0.,.70710678],[1.,0.,0.,0.]])
        linear=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        angular=torch.tensor([[1.,0.,0.],[0.,0.,0.]])
        joints=torch.tensor([[3.,-3.],[.2,.3]])
        velocity=torch.tensor([[.4,.5],[.6,.7]])
        write_reference_masked(cmd,torch.tensor([True,False]),root_pos,quat,linear,angular,joints,velocity)
        torch.testing.assert_close(qpos[0,:3],root_pos[0]);torch.testing.assert_close(qpos[0,3:7],quat[0])
        torch.testing.assert_close(qpos[0,7:],torch.tensor([1.,-2.]))
        torch.testing.assert_close(qvel[0,:3],linear[0]);torch.testing.assert_close(qvel[0,3:6],torch.tensor([0.,-1.,0.]),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(qvel[0,6:],velocity[0])
        torch.testing.assert_close(qpos[1],oldpos[1]);torch.testing.assert_close(qvel[1],oldvel[1])

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedActionManagerTests(unittest.TestCase):
    def test_fixed_mask_matches_selected_action_history_reset(self):
        from types import SimpleNamespace
        from mjlab.managers.action_manager import ActionManager
        from morphmimic.env.masked_reset import reset_action_manager_masked
        class Term:
            def __init__(self): self._raw_actions=torch.arange(12.).reshape(3,4)
            def reset(self,env_ids): self._raw_actions[env_ids]=0
        def manager():
            return SimpleNamespace(_prev_action=torch.ones(3,4),_prev_prev_action=torch.ones(3,4)*2,
                _action=torch.ones(3,4)*3,_terms={'joint_pos':Term()})
        for mask in (torch.tensor([False,False,False]),torch.tensor([True,False,True]),torch.tensor([True,True,True])):
            with self.subTest(mask=mask.tolist()):
                a,b=manager(),manager();ActionManager.reset(a,mask.nonzero().flatten());reset_action_manager_masked(b,mask)
                for name in ('_prev_action','_prev_prev_action','_action'):
                    torch.testing.assert_close(getattr(a,name),getattr(b,name))
                torch.testing.assert_close(a._terms['joint_pos']._raw_actions,b._terms['joint_pos']._raw_actions)

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedRewardManagerTests(unittest.TestCase):
    def test_selected_episode_sums_and_logs_match_runtime(self):
        from types import SimpleNamespace
        from mjlab.managers.reward_manager import RewardManager
        from morphmimic.env.masked_reset import reset_reward_manager_masked
        def manager():
            return SimpleNamespace(_episode_sums={'one':torch.tensor([1.,2.,3.]),'two':torch.tensor([4.,5.,6.])},
                _class_term_cfgs=[],_env=SimpleNamespace(max_episode_length_s=10.))
        for mask in (torch.tensor([True,False,True]),torch.tensor([True,True,True])):
            with self.subTest(mask=mask.tolist()):
                a,b=manager(),manager()
                expected=RewardManager.reset(a,mask.nonzero().flatten())
                actual=reset_reward_manager_masked(b,mask)
                for key in expected:torch.testing.assert_close(expected[key],actual[key])
                for key in a._episode_sums:torch.testing.assert_close(a._episode_sums[key],b._episode_sums[key])

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedCommandStateTests(unittest.TestCase):
    def test_selected_command_counters_and_frames_reset_without_touching_others(self):
        from types import SimpleNamespace
        from morphmimic.env.masked_reset import reset_command_bookkeeping_masked
        class Library:
            clip_start=torch.tensor([0,100]);frame_clip=torch.tensor([0]*100+[1]*100);num_bins=2
            def frame_bin(self,frames): return (frames>=100).long()
            def clip_end(self,frames): return torch.where(frames<100,torch.tensor(99),torch.tensor(199))
            def remaining(self,frames): return self.clip_end(frames)-frames
        cfg=SimpleNamespace(sampling='start',failure_reset=0.,gamma=.9,r_max=1.,reset_norm='actual',
            ema_alpha=.1,softmax_temp=1.,mix=(.5,.4,.1),softmax_cap=None,resampling_time_range=(1e9,1e9))
        cmd=SimpleNamespace(num_envs=2,device='cpu',library=Library(),env_clip=torch.tensor([0,1]),cfg=cfg,
            G_b=torch.tensor([.2,.8]),past_softmax=torch.tensor([.4,.6]),start_bin=torch.tensor([0,1]),
            global_frame=torch.tensor([20,120]),consecutive_local=torch.tensor([0,0]),restart_frame=torch.tensor([-1,-1]),
            disc_return=torch.tensor([.7,.9]),gamma_pow=torch.tensor([.5,.5]),horizon_pow=torch.tensor([.1,.1]),
            episode_steps=torch.tensor([2,2]),command_counter=torch.tensor([3,4]),time_left=torch.tensor([10.,20.]),
            metrics={'error':torch.tensor([1.,2.]),'sampling_entropy':torch.tensor([.3,.4])},
            _env=SimpleNamespace(max_episode_length=100))
        frames=reset_command_bookkeeping_masked(cmd,torch.tensor([True,False]))
        torch.testing.assert_close(frames,torch.tensor([0,100]))
        torch.testing.assert_close(cmd.global_frame,torch.tensor([0,120]))
        torch.testing.assert_close(cmd.episode_steps,torch.tensor([0,2]))
        torch.testing.assert_close(cmd.command_counter,torch.tensor([1,4]))
        torch.testing.assert_close(cmd.disc_return,torch.tensor([0.,.9]))
        torch.testing.assert_close(cmd.gamma_pow,torch.tensor([1.,.5]))
        torch.testing.assert_close(cmd.metrics['error'],torch.tensor([0.,2.]))
        self.assertEqual(float(cmd.time_left[1]),20.)

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedReferenceCandidateTests(unittest.TestCase):
    def test_zero_noise_uses_requested_source_frames_and_native_mapping(self):
        from types import SimpleNamespace
        from morphmimic.constants import JOINT_NAMES
        from morphmimic.robots import Translator
        from morphmimic.env.masked_reset import reference_state_candidates
        q=torch.zeros(3,29);q[:,0]=torch.tensor([.1,.2,.3]);q[:,-1]=torch.tensor([-.1,-.2,-.3])
        root=torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
        quat=torch.tensor([[1.,0.,0.,0.]]).expand(3,-1)
        lib=SimpleNamespace(root_pos_w=root,root_quat_w=quat,root_lin_vel_w=root*2,root_ang_vel_w=root*3,
            joint_pos=q,joint_vel=q*2)
        robot=SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros(2,2)))
        cmd=SimpleNamespace(library=lib,device='cpu',num_envs=2,
            cfg=SimpleNamespace(joint_position_range=(0.,0.)),
            _env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.tensor([[10.,0.,0.],[20.,0.,0.]]))),
            spec=SimpleNamespace(translator=Translator.from_names((JOINT_NAMES[0],JOINT_NAMES[-1]),{},{},{})),robot=robot)
        values=reference_state_candidates(cmd,torch.tensor([0,2]),torch.zeros(6,2),torch.zeros(6,2))
        torch.testing.assert_close(values[0],torch.tensor([[11.,2.,3.],[27.,8.,9.]]))
        torch.testing.assert_close(values[1],quat[[0,2]])
        torch.testing.assert_close(values[2],(root*2)[[0,2]])
        torch.testing.assert_close(values[3],(root*3)[[0,2]])
        torch.testing.assert_close(values[4],torch.tensor([[.1,-.1],[.3,-.3]]))
        torch.testing.assert_close(values[5],torch.tensor([[.2,-.2],[.6,-.6]]))

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class ConditionalDelayTests(unittest.TestCase):
    def test_disabled_reset_tick_preserves_delay_ring_and_lag_state(self):
        from morphmimic.env.graph_delay import GraphDelayBuffer
        buf=GraphDelayBuffer(min_lag=0,max_lag=4,batch_size=2,device='cpu',update_period=13,per_env_phase=False)
        buf.append(torch.tensor([[1.],[2.]]));buf.compute()
        before=(buf._buffer._buffer.clone(),buf._buffer._num_pushes.clone(),buf._buffer._pointer.clone(),
                buf._current_lags.clone(),buf._step_count.clone())
        buf._enabled.fill_(False)
        buf.append(torch.tensor([[9.],[8.]]));buf.compute()
        after=(buf._buffer._buffer,buf._buffer._num_pushes,buf._buffer._pointer,buf._current_lags,buf._step_count)
        for a,b in zip(before,after):torch.testing.assert_close(a,b)

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedEntityTests(unittest.TestCase):
    def test_target_and_delay_reset_match_selected_runtime_semantics(self):
        from types import SimpleNamespace
        from mjlab.entity.data import EntityData
        from morphmimic.env.graph_delay import GraphDelayBuffer
        from morphmimic.env.masked_reset import reset_entity_masked
        fields=('joint_pos_target','joint_vel_target','joint_effort_target','tendon_len_target',
                'tendon_vel_target','tendon_effort_target','site_effort_target')
        def entity():
            data=SimpleNamespace(is_actuated=True,_resolve_env_ids=lambda ids:ids[:,None] if isinstance(ids,torch.Tensor) else ids)
            for name in fields:setattr(data,name,torch.ones(3,2))
            delay=GraphDelayBuffer(min_lag=0,max_lag=4,batch_size=3,device='cpu',update_period=13,per_env_phase=False)
            delay.append(torch.arange(6.).reshape(3,2));delay.compute()
            act=SimpleNamespace(_delay_buffer=delay)
            return SimpleNamespace(_data=data,actuators=[act])
        for mask in (torch.tensor([False,False,False]),torch.tensor([True,False,True]),torch.tensor([True,True,True])):
            with self.subTest(mask=mask.tolist()):
                torch.manual_seed(42);a=entity();torch.manual_seed(42);b=entity();ids=mask.nonzero().flatten()
                EntityData.clear_state(a._data,ids);a.actuators[0]._delay_buffer.reset(ids)
                reset_entity_masked(b,mask)
                for name in fields:torch.testing.assert_close(getattr(a._data,name),getattr(b._data,name))
                torch.testing.assert_close(a.actuators[0]._delay_buffer._buffer._buffer,b.actuators[0]._delay_buffer._buffer._buffer)

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_COLLECTOR_MIXED_SUBSET'), 'requires GPU physics fixture')
class MaskedEnvironmentTests(unittest.TestCase):
    def test_none_partial_all_match_eager_reference_reset(self):
        import os
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.env.masked_reset import MaskedEnvReset
        subset=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'];root=os.environ['BLINKMORPH_HUMAN_ROOT'];device='cuda:0'
        def cfg():
            value=morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=True,human_root=root)
            value.commands['motion'].allow_missing_human=True
            return value
        a=MorphMimicEnv(cfg(),device=device)
        b=MorphMimicEnv(cfg(),device=device,graph_stream=torch.cuda.Stream(device=device))
        try:
            reset=MaskedEnvReset(b)
            for bits in ([False,False],[True,False],[True,True]):
                a.reset();b.reset()
                action=torch.linspace(-.1,.1,29,device=device).repeat(2,1)
                a.step(action)
                from morphmimic.env.control_graph import PhysicsGraph
                control=PhysicsGraph(b);control.warmup(torch.zeros_like(action));b.reset();control.step(action)
                mask=torch.tensor(bits,device=device);ids=mask.nonzero().flatten()
                old_a={key:value.clone() for key,value in a.obs_buf.items()}
                if ids.numel(): expected,_=a.reset(env_ids=ids)
                else: expected=old_a
                actual=reset(mask)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(a.sim.data.qpos,b.sim.data.qpos,rtol=1e-5,atol=1e-5)
                torch.testing.assert_close(a.sim.data.qvel,b.sim.data.qvel,rtol=1e-4,atol=1e-4)
                for key in ('policy_clean','state','goal','goal_kp','context'):
                    torch.testing.assert_close(actual[key][mask],expected[key][mask],rtol=2e-4,atol=2e-4)
        finally:a.close();b.close()

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class CollectorOverflowBoundaryTests(unittest.TestCase):
    def test_refresh_sampler_is_noop_without_masked_reset(self):
        from morphmimic.train.collector import Collector
        collector=Collector.__new__(Collector);collector._masked_reset=None
        collector.refresh_reset_sampler()

    def test_graph_step_stops_before_replay_or_observation_publish(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from morphmimic.train.collector import Collector
        diag=SimpleNamespace(bits_tensor=lambda:torch.tensor(1),report=lambda:{'overflow_bits':1})
        env=SimpleNamespace(num_envs=1,device='cpu',physics_diagnostics=diag)
        collector=Collector.__new__(Collector);collector.env=env;collector.replay=MagicMock();collector._masked_reset=None
        collector.command=SimpleNamespace(global_frame=torch.zeros(1,dtype=torch.long))
        collector.obs={'goal':torch.zeros(1,1),'goal_kp':torch.zeros(1,1)};old=collector.obs
        collector.history=MagicMock();collector.history.flat.return_value=torch.zeros(1,1)
        collector.policy_fn=lambda *_:torch.zeros(1,29);collector.action_term=None
        collector.action_noise_std=0.;collector.noise_scale=torch.ones(1,1)
        collector._physics_graph=MagicMock();collector._physics_graph.step.return_value=({},torch.zeros(1),torch.zeros(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool),{})
        with self.assertRaisesRegex(RuntimeError,'overflow'):collector.step(1)
        collector.replay.add_step.assert_not_called();self.assertIs(collector.obs,old)

    def test_graph_reset_stops_before_observation_publish(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from morphmimic.train.collector import Collector
        diag=SimpleNamespace(bits_tensor=lambda:torch.tensor(1),report=lambda:{'overflow_bits':1})
        env=SimpleNamespace(physics_diagnostics=diag,reset=lambda:({},{}))
        collector=Collector.__new__(Collector);collector.env=env;collector.history=MagicMock();collector._masked_reset=None
        collector._physics_graph=MagicMock();collector._graph_warmed=True;collector.obs={'old':1}
        with self.assertRaisesRegex(RuntimeError,'overflow'):collector.reset_all()
        self.assertEqual(collector.obs,{'old':1});collector.history.reset.assert_not_called()

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_COLLECTOR_MIXED_SUBSET'), 'requires GPU physics fixture')
class MaskedEnvironmentCaptureTests(unittest.TestCase):
    def test_graph_replays_none_partial_all_masks(self):
        import os
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.env.masked_reset import MaskedResetGraph
        subset=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'];root=os.environ['BLINKMORPH_HUMAN_ROOT'];device='cuda:0'
        def cfg():
            value=morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=True,human_root=root)
            value.commands['motion'].allow_missing_human=True
            return value
        a=MorphMimicEnv(cfg(),device=device)
        b=MorphMimicEnv(cfg(),device=device,graph_stream=torch.cuda.Stream(device=device))
        try:
            a.reset();b.reset();reset=MaskedResetGraph(b)
            for bits in ([False,False],[True,False],[True,True]):
                mask=torch.tensor(bits,device=device);ids=mask.nonzero().flatten()
                if ids.numel():expected,_=a.reset(env_ids=ids)
                else:expected=a.obs_buf
                actual=reset(mask);torch.cuda.synchronize(device)
                torch.testing.assert_close(a.sim.data.qpos,b.sim.data.qpos,rtol=1e-5,atol=1e-5)
                torch.testing.assert_close(a.sim.data.qvel,b.sim.data.qvel,rtol=1e-4,atol=1e-4)
                for key in ('policy_clean','state','goal','goal_kp','context'):
                    torch.testing.assert_close(actual[key][mask],expected[key][mask],rtol=2e-4,atol=2e-4)
                ca=a.command_manager.get_term('motion');cb=b.command_manager.get_term('motion')
                for key in ('global_frame','G_b','past_softmax','episode_steps','command_counter'):
                    torch.testing.assert_close(getattr(ca,key),getattr(cb,key),rtol=2e-4,atol=2e-4)
        finally:a.close();b.close()

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_COLLECTOR_MIXED_SUBSET'), 'requires GPU physics fixture')
class MaskedTrainingCaptureTests(unittest.TestCase):
    def test_reset_graph_with_startup_dr_and_actuator_delay(self):
        import os
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.env.control_graph import PhysicsGraph
        from morphmimic.env.masked_reset import MaskedResetGraph
        from morphmimic.train.collector import Collector
        from morphmimic.train.replay import ReplayBuffer, ReplayCfg
        subset=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'];root=os.environ['BLINKMORPH_HUMAN_ROOT'];device='cuda:0'
        def cfg():
            value=morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=False,human_root=root)
            value.commands['motion'].allow_missing_human=True
            value.commands['motion'].joint_position_range=(0.,0.)
            value.observations['policy'].enable_corruption=False
            return value
        torch.manual_seed(29);a=MorphMimicEnv(cfg(),device=device)
        torch.manual_seed(29);b=MorphMimicEnv(cfg(),device=device,graph_stream=torch.cuda.Stream(device=device))
        def paired(left,right):
            torch.cuda.synchronize();cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state(device)
            x=left();torch.random.set_rng_state(cpu);torch.cuda.set_rng_state(gpu,device);y=right();torch.cuda.synchronize()
            return x,y
        def delays(env):
            result=[];seen=set()
            for actuator in env.scene['robot'].actuators:
                delay=actuator._delay_buffer
                if delay is not None and id(delay) not in seen:result.append(delay);seen.add(id(delay))
            return result
        try:
            collector=Collector(b,ReplayBuffer(ReplayCfg(2,16,device)),lambda h,g:torch.zeros(2,29,device=device),physics_graph=True)
            collector._warmup_graph();control=collector._physics_graph
            paired(a.reset,b.reset)
            reset=MaskedResetGraph(b);action=torch.linspace(-.1,.1,29,device=device).repeat(2,1)
            for bits in ([False,False],[True,False],[True,True],[False,True]):
                paired(lambda:a.step(action),lambda:control.step(action))
                mask=torch.tensor(bits,device=device);ids=mask.nonzero().flatten()
                if ids.numel(): a.reset(env_ids=ids)
                reset(mask);torch.cuda.synchronize(device)
                for da,db in zip(delays(a),delays(b),strict=True):
                    torch.testing.assert_close(da._buffer.buffer,db._buffer.buffer,rtol=1e-4,atol=1e-4)
                    torch.testing.assert_close(da._buffer._num_pushes,db._buffer._num_pushes)
                    torch.testing.assert_close(da._step_count,db._step_count)
                    da._current_lags.fill_(2);db._current_lags.fill_(2)
                torch.testing.assert_close(a.sim.data.qpos,b.sim.data.qpos,rtol=1e-4,atol=1e-4)
                torch.testing.assert_close(a.sim.data.qvel,b.sim.data.qvel,rtol=1e-4,atol=1e-4)
                ca=a.command_manager.get_term('motion');cb=b.command_manager.get_term('motion')
                for key in ('global_frame','G_b','past_softmax','episode_steps','command_counter'):
                    torch.testing.assert_close(getattr(ca,key),getattr(cb,key),rtol=2e-4,atol=2e-4)
            self.assertEqual(a.physics_diagnostics.report()['overflow_bits'],0)
            self.assertEqual(b.physics_diagnostics.report()['overflow_bits'],0)
        finally:a.close();b.close()
