"""Opt-in GPU comparison of actual collector semantics across a partial reset."""
import os
import unittest


def _forced_failure(env):
    return env.episode_length_buf == env.max_episode_length - 2


@unittest.skipUnless(os.environ.get('BLINKMORPH_COLLECTOR_MIXED_SUBSET'), 'requires real mixed GPU fixture')
class CollectorGraphParity(unittest.TestCase):
    def test_masked_sampler_refresh_uses_replaced_probe_focus(self):
        import torch
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.train.collector import Collector
        from morphmimic.train.replay import ReplayBuffer, ReplayCfg
        device='cuda:0';n=64
        cfg=morphmimic_env_cfg('q1',subset_yaml=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'],num_envs=n,play=True,eval=True,human_root=os.environ['BLINKMORPH_HUMAN_ROOT'])
        cfg.commands['motion'].allow_missing_human=True
        cfg.commands['motion'].sampling='return'
        env=MorphMimicEnv(cfg,device=device,graph_stream=torch.cuda.Stream(device=device))
        try:
            collector=Collector(env,ReplayBuffer(ReplayCfg(n,2,device)),lambda h,g:torch.zeros(n,29,device=device),physics_graph=True,masked_reset=True)
            env.reset()
            command=collector.command
            self.assertGreaterEqual(command.library.num_bins,2)
            mask=torch.ones(n,dtype=torch.bool,device=device)
            self.assertIsNone(command.probe_probabilities)
            collector._masked_reset(mask)
            for bin_id in (0,1):
                focus=torch.zeros(command.library.num_bins,device=device);focus[bin_id]=1
                command.set_probe_focus(focus,0.999999)
                collector.refresh_reset_sampler()
                self.assertIsNone(collector._masked_reset.graph)
                collector._masked_reset(mask)
                self.assertGreaterEqual(int((command.start_bin==bin_id).sum()),n-2)
        finally:env.close()

    def test_collector_matches_eager_through_partial_timeout_and_domain_randomization(self):
        import torch
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.train.collector import Collector
        from morphmimic.train.replay import ReplayBuffer, ReplayCfg

        device='cuda:0'
        subset=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET']
        human_root=os.environ['BLINKMORPH_HUMAN_ROOT']
        def config():
            cfg=morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=False,human_root=human_root)
            cfg.commands['motion'].allow_missing_human=True
            cfg.observations['policy'].enable_corruption=False
            return cfg
        torch.manual_seed(17)
        eager=MorphMimicEnv(config(),device=device)
        torch.manual_seed(17)
        captured=MorphMimicEnv(config(),device=device,graph_stream=torch.cuda.Stream(device=device))
        def policy(history,goals):
            return torch.linspace(-.1,.1,29,device=history.device).expand(history.shape[0],-1)
        replay_a=ReplayBuffer(ReplayCfg(num_envs=2,capacity=16,device=device))
        replay_b=ReplayBuffer(ReplayCfg(num_envs=2,capacity=16,device=device))
        a=Collector(eager,replay_a,policy,action_noise_std=0.)
        b=Collector(captured,replay_b,policy,action_noise_std=0.,physics_graph=True)
        def paired(f,g):
            torch.cuda.synchronize()
            cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state(device)
            left=f()
            torch.random.set_rng_state(cpu);torch.cuda.set_rng_state(gpu,device)
            right=g()
            torch.cuda.synchronize()
            return left,right
        def compare():
            for group in ('policy_clean','state','goal','goal_kp','context','policy'):
                torch.testing.assert_close(a.obs[group],b.obs[group],rtol=2e-4,atol=2e-4,msg=lambda message: f'{group}: {message}')
            torch.testing.assert_close(a.history.buf,b.history.buf,rtol=2e-4,atol=2e-4)
            for field in ('state','obs','action','reward','terminated','timeout','no_action','ctx','frame','ep_step','valid','ptr','size'):
                torch.testing.assert_close(getattr(replay_a,field),getattr(replay_b,field),rtol=2e-4,atol=2e-4)
            torch.testing.assert_close(a.ep_step,b.ep_step)
        try:
            b.reset_all()  # preflight creates graph resources before the synchronized resets
            paired(a.reset_all,b.reset_all)
            valid=a.command.library.human_valid_at(a.command.global_frame)
            self.assertTrue(bool(valid[0].any()))
            self.assertFalse(bool(valid[1].any()))
            self.assertIn('startup',eager.event_manager.available_modes)
            compare()
            paired(lambda:a.step(1),lambda:b.step(1));compare()
            for env in (eager,captured):
                env.episode_length_buf[0]=env.max_episode_length-1
            stats_a,stats_b=paired(lambda:a.step(1),lambda:b.step(1));compare()
            self.assertEqual(stats_a['episodes_finished'],1)
            self.assertEqual(stats_b['episodes_finished'],1)
            self.assertTrue(bool(replay_a.timeout[0].any()))
            self.assertTrue(bool(replay_a.no_action[0].any()))
            self.assertFalse(bool(replay_a.no_action[1].any()))
            paired(lambda:a.step(1),lambda:b.step(1));compare()
            self.assertEqual(eager.physics_diagnostics.report()['overflow_bits'],0)
            self.assertEqual(captured.physics_diagnostics.report()['overflow_bits'],0)
        finally:
            eager.close();captured.close()

    def test_graph_environment_can_be_first_warp_user(self):
        import subprocess
        import sys
        code='''
import os, torch
from morphmimic.env.env_cfg import morphmimic_env_cfg
from morphmimic.env.morphmimic_env import MorphMimicEnv
cfg=morphmimic_env_cfg('q1',subset_yaml=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'],num_envs=1,play=True,eval=True,human_root=os.environ['BLINKMORPH_HUMAN_ROOT'])
cfg.commands['motion'].allow_missing_human=True
env=MorphMimicEnv(cfg,device='cuda:0',graph_stream=torch.cuda.Stream(device='cuda:0'))
env.close()
'''
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stderr[-4000:])

    def test_full_batch_random_reset_changes_selected_values(self):
        import torch
        device='cuda:0'
        torch.manual_seed(9821)
        compact=torch.rand((1,6),device=device)
        torch.manual_seed(9821)
        full=torch.rand((64,6),device=device)
        self.assertFalse(torch.equal(compact[0],full[31]))

    def test_unused_native_graphs_can_be_released_without_changing_solver(self):
        import torch,warp as wp
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        cfg=morphmimic_env_cfg('q1',subset_yaml=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'],num_envs=2,play=True,eval=True,human_root=os.environ['BLINKMORPH_HUMAN_ROOT'])
        cfg.commands['motion'].allow_missing_human=True
        stream=torch.cuda.Stream(device='cuda:0')
        env=MorphMimicEnv(cfg,device='cuda:0',graph_stream=stream)
        try:
            self.assertIsNone(env.sim.step_graph)
            with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
                env.sim.use_cuda_graph=True;env.sim.create_graph()
            torch.cuda.synchronize()
            before=torch.cuda.mem_get_info()[0]
            with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
                env.sim.use_cuda_graph=False;env.sim.create_graph()
            torch.cuda.synchronize()
            after=torch.cuda.mem_get_info()[0]
            print('NATIVE_GRAPH_FREE_BYTES_BEFORE',before,'AFTER',after,'DELTA',after-before,flush=True)
            self.assertIsNone(env.sim.step_graph);self.assertIsNone(env.sim.forward_graph);self.assertIsNone(env.sim.reset_graph)
        finally:env.close()

    def test_warmup_preserves_first_sampler_reset_and_rng(self):
        import torch
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.train.collector import Collector
        from morphmimic.train.replay import ReplayBuffer,ReplayCfg
        subset=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'];root=os.environ['BLINKMORPH_HUMAN_ROOT'];device='cuda:0'
        def config():
            cfg=morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=True,human_root=root)
            cfg.commands['motion'].allow_missing_human=True
            return cfg
        a=MorphMimicEnv(config(),device=device)
        b=MorphMimicEnv(config(),device=device,graph_stream=torch.cuda.Stream(device=device))
        policy=lambda history,goals:torch.zeros((history.shape[0],29),device=device)
        ca=Collector(a,ReplayBuffer(ReplayCfg(2,16,device)),policy,action_noise_std=0.)
        cb=Collector(b,ReplayBuffer(ReplayCfg(2,16,device)),policy,action_noise_std=0.,physics_graph=True)
        try:
            torch.manual_seed(917)
            cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state(device)
            ca.reset_all()
            torch.random.set_rng_state(cpu);torch.cuda.set_rng_state(gpu,device)
            cb.reset_all();torch.cuda.synchronize()
            for field in ('G_b','past_softmax','global_frame','start_bin','disc_return','gamma_pow','episode_steps','time_left'):
                torch.testing.assert_close(getattr(ca.command,field),getattr(cb.command,field),msg=field)
            torch.testing.assert_close(ca.obs['policy_clean'],cb.obs['policy_clean'])
            self.assertEqual(a._sim_step_counter,b._sim_step_counter)
            self.assertEqual(a.common_step_counter,b.common_step_counter)
            self.assertGreater(b.physics_diagnostics.report()['checked_physics_steps'],0)
        finally:a.close();b.close()

    def test_reset_forward_rejects_replaced_simulator_array(self):
        import torch,warp as wp
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        cfg=morphmimic_env_cfg('q1',subset_yaml=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'],num_envs=1,play=True,eval=True,human_root=os.environ['BLINKMORPH_HUMAN_ROOT'])
        cfg.commands['motion'].allow_missing_human=True
        env=MorphMimicEnv(cfg,device='cuda:0',graph_stream=torch.cuda.Stream(device='cuda:0'))
        try:
            env.reset()
            original=env.sim.wp_data.qpos
            env.sim.wp_data.qpos=wp.clone(original)
            try:
                with self.assertRaisesRegex(RuntimeError,'arrays changed'):
                    env.reset()
            finally:
                env.sim.wp_data.qpos=original
        finally:env.close()

    def test_masked_collector_matches_eager_none_partial_and_all(self):
        import torch
        from mjlab.managers.termination_manager import TerminationTermCfg
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.train.collector import Collector
        from morphmimic.train.replay import ReplayBuffer,ReplayCfg
        subset=os.environ['BLINKMORPH_COLLECTOR_MIXED_SUBSET'];root=os.environ['BLINKMORPH_HUMAN_ROOT'];device='cuda:0'
        def cfg():
            c=morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=False,human_root=root)
            c.commands['motion'].allow_missing_human=True;c.commands['motion'].joint_position_range=(0.,0.)
            c.observations['policy'].enable_corruption=False
            c.terminations['forced_failure']=TerminationTermCfg(func=_forced_failure)
            return c
        torch.manual_seed(71);aenv=MorphMimicEnv(cfg(),device=device)
        torch.manual_seed(71);benv=MorphMimicEnv(cfg(),device=device,graph_stream=torch.cuda.Stream(device=device))
        def policy(history,goals): return torch.linspace(-.1,.1,29,device=device).expand(2,-1)
        a=Collector(aenv,ReplayBuffer(ReplayCfg(2,24,device)),policy,action_noise_std=0.)
        calls=[]
        b=Collector(benv,ReplayBuffer(ReplayCfg(2,24,device)),policy,action_noise_std=0.,physics_graph=True,
                    masked_reset=True,on_reset_masked=lambda mask:calls.append(mask.clone()))
        def paired(f,g):
            torch.cuda.synchronize();cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state(device)
            x=f();torch.random.set_rng_state(cpu);torch.cuda.set_rng_state(gpu,device);y=g();torch.cuda.synchronize();return x,y
        def compare():
            for field in ('state','obs','action','reward','terminated','timeout','no_action','ctx','frame','ep_step','valid','ptr','size'):
                torch.testing.assert_close(getattr(a.replay,field),getattr(b.replay,field),atol=2e-4,rtol=2e-4,msg=field)
            for key in ('policy_clean','goal','goal_kp','state','context'):
                torch.testing.assert_close(a.obs[key],b.obs[key],atol=2e-4,rtol=2e-4,msg=key)
            torch.testing.assert_close(a.history.buf,b.history.buf,atol=2e-4,rtol=2e-4)
        try:
            b._warmup_graph();b._graph_warmed=True
            for env in (aenv,benv):
                for act in env.scene['robot'].actuators:
                    if act._delay_buffer is not None:act._delay_buffer.min_lag=act._delay_buffer.max_lag=2
            paired(a.reset_all,b.reset_all);compare()
            for done in ((),(0,),(),(0,1)):
                for env in (aenv,benv):
                    for i in done:env.episode_length_buf[i]=env.max_episode_length-1
                sa,sb=paired(lambda:a.step(1),lambda:b.step(1));compare()
                self.assertEqual(sa['episodes_finished'],len(done));self.assertEqual(sb['episodes_finished'],len(done))
            for env in (aenv,benv):env.episode_length_buf[1]=env.max_episode_length-3
            sa,sb=paired(lambda:a.step(1),lambda:b.step(1));compare()
            self.assertEqual((sa['failures'],sb['failures']),(1,1))
            self.assertTrue(bool(a.replay.terminated[1].any()))
            self.assertEqual([mask.tolist() for mask in calls],[[True,True],[False,False],[True,False],[False,False],[True,True],[False,True]])
            self.assertEqual(aenv.physics_diagnostics.report()['overflow_bits'],0)
            self.assertEqual(benv.physics_diagnostics.report()['overflow_bits'],0)
        finally:aenv.close();benv.close()
