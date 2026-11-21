"""Native G1 reset parity, including its real builtin sensors and actuator delay."""
import os
import unittest
import torch


@unittest.skipUnless(os.environ.get('BLINKMORPH_G1_RESET_SUBSET'), 'requires real G1 GPU fixture')
class G1MaskedResetTests(unittest.TestCase):
    def test_none_partial_all_reset_graph_matches_eager_with_builtin_sensors_and_delay(self):
        from dataclasses import replace
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.env.masked_reset import MaskedEnvReset, MaskedResetGraph
        from morphmimic.train.collector import Collector
        from morphmimic.train.replay import ReplayBuffer, ReplayCfg
        device='cuda:0'
        def config():
            cfg=morphmimic_env_cfg('g1',subset_yaml=os.environ['BLINKMORPH_G1_RESET_SUBSET'],num_envs=2,
                play=True,eval=False,human_root=os.environ['BLINKMORPH_HUMAN_ROOT'])
            cfg.commands['motion'].allow_missing_human=True
            cfg.commands['motion'].joint_position_range=(0.,0.)
            cfg.observations['policy'].enable_corruption=False
            # Fixed nonzero lag isolates reset correctness from the documented
            # difference between full-world and selected-world random draws.
            art=cfg.scene.entities['robot'].articulation
            art.actuators=tuple(replace(a,delay_min_lag=2,delay_max_lag=2) for a in art.actuators)
            return cfg
        torch.manual_seed(29);a=MorphMimicEnv(config(),device=device)
        torch.manual_seed(29);b=MorphMimicEnv(config(),device=device,graph_stream=torch.cuda.Stream(device=device))
        def paired(left,right):
            torch.cuda.synchronize();cpu=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state(device)
            x=left();torch.random.set_rng_state(cpu);torch.cuda.set_rng_state(gpu,device);y=right();torch.cuda.synchronize()
            return x,y
        def delays(env):
            values={id(x._delay_buffer):x._delay_buffer for x in env.scene['robot'].actuators if x._delay_buffer is not None}
            return list(values.values())
        try:
            self.assertTrue(b.scene.sensors, 'G1 fixture must contain actual native builtin sensors')
            reset=MaskedResetGraph(b)  # Regression: the previous guard rejected G1 here.
            collector=Collector(b,ReplayBuffer(ReplayCfg(2,16,device)),lambda h,g:torch.zeros(2,29,device=device),physics_graph=True)
            collector._warmup_graph();control=collector._physics_graph
            paired(a.reset,b.reset)
            action=torch.linspace(-.1,.1,29,device=device).repeat(2,1)
            for bits in ([False,False],[True,False],[True,True],[False,True]):
                with self.subTest(mask=bits):
                    paired(lambda:a.step(action),lambda:control.step(action))
                    mask=torch.tensor(bits,device=device);ids=mask.nonzero().flatten()
                    expected=a.obs_buf
                    if ids.numel():expected,_=a.reset(env_ids=ids)
                    actual=reset(mask);torch.cuda.synchronize(device)
                    for field in ('qpos','qvel','sensordata'):
                        torch.testing.assert_close(getattr(a.sim.data,field),getattr(b.sim.data,field),rtol=2e-4,atol=2e-4,msg=field)
                    for name in a.scene.sensors:
                        torch.testing.assert_close(a.scene.sensors[name].data,b.scene.sensors[name].data,rtol=2e-4,atol=2e-4,msg=name)
                    for key in ('policy_clean','state','goal','goal_kp','context'):
                        torch.testing.assert_close(actual[key],expected[key],rtol=2e-4,atol=2e-4,msg=key)
                    ca=a.command_manager.get_term('motion');cb=b.command_manager.get_term('motion')
                    for key in ('global_frame','G_b','past_softmax','episode_steps','command_counter'):
                        torch.testing.assert_close(getattr(ca,key),getattr(cb,key),rtol=2e-4,atol=2e-4,msg=key)
                    self.assertTrue(delays(b))
                    for da,db in zip(delays(a),delays(b),strict=True):
                        torch.testing.assert_close(da._buffer.buffer,db._buffer.buffer,rtol=2e-4,atol=2e-4)
                        for field in ('_current_lags','_step_count','_phase_offsets'):
                            torch.testing.assert_close(getattr(da,field),getattr(db,field),msg=field)
                        torch.testing.assert_close(da._buffer._num_pushes,db._buffer._num_pushes)
            self.assertEqual(a.physics_diagnostics.report()['overflow_bits'],0)
            self.assertEqual(b.physics_diagnostics.report()['overflow_bits'],0)
            # Unrecognized sensors must remain a hard failure with their name.
            b.scene.sensors['unexpected_stateful']=object()
            try:
                with self.assertRaisesRegex(ValueError,'unexpected_stateful'):
                    MaskedEnvReset(b)
            finally:del b.scene.sensors['unexpected_stateful']
        finally:a.close();b.close()


@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class MaskedResetGuardTests(unittest.TestCase):
    def environment(self):
        from types import SimpleNamespace as NS
        return NS(tensor_step_stream=object(),cfg=NS(auto_reset=False),
            event_manager=NS(available_modes=[]),scene=NS(sensors={}),
            curriculum_manager=NS(active_terms=[]),metrics_manager=NS(active_terms=[]),
            recorder_manager=NS(active_terms=[]),reward_manager=NS(_class_term_cfgs=[]),
            termination_manager=NS(_class_term_cfgs=[]))

    def test_unknown_sensor_and_builtin_subclass_remain_rejected(self):
        from mjlab.sensor.builtin_sensor import BuiltinSensor
        from morphmimic.env.masked_reset import MaskedEnvReset
        class StatefulBuiltin(BuiltinSensor):
            def reset(self,env_ids=None):
                self.history=[]
        for sensor in (object(),StatefulBuiltin(name='extra')):
            env=self.environment();env.scene.sensors['extra_history']=sensor
            with self.subTest(sensor=type(sensor).__name__):
                with self.assertRaisesRegex(ValueError,'extra_history'):
                    MaskedEnvReset(env)

    def test_active_manager_rejection_identifies_category_and_term(self):
        from morphmimic.env.masked_reset import MaskedEnvReset
        for field,label in (('curriculum_manager','curriculum'),('metrics_manager','metrics'),('recorder_manager','recorder')):
            env=self.environment();getattr(env,field).active_terms=['unexpected_term']
            with self.subTest(manager=field):
                with self.assertRaisesRegex(ValueError,f'{label} terms.*unexpected_term'):
                    MaskedEnvReset(env)

    def test_stateful_reward_and_termination_guard_remains_strict(self):
        from morphmimic.env.masked_reset import MaskedEnvReset
        for field in ('reward_manager','termination_manager'):
            env=self.environment();getattr(env,field)._class_term_cfgs=[object()]
            with self.subTest(manager=field):
                with self.assertRaisesRegex(ValueError,'class-based reward or termination'):
                    MaskedEnvReset(env)
