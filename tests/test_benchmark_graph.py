import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

try:
    import morphmimic.eval.benchmark as benchmark
except ImportError:
    benchmark=None

@unittest.skipIf(benchmark is None,'project runtime required')
class BenchmarkGraphTests(unittest.TestCase):
    def test_warmup_is_once_and_does_not_consume_rng_or_counters(self):
        calls=[]
        class FakeGraph:
            def warmup(self,action):
                calls.append('warmup');torch.rand(2);env._sim_step_counter+=10;env.common_step_counter+=1
        names=('G_b','past_softmax','global_frame','start_bin','disc_return','gamma_pow','horizon_pow','episode_steps','time_left','command_counter','restart_frame','consecutive_local')
        command=SimpleNamespace(**{name:torch.zeros(4) for name in names},metrics={'x':torch.zeros(4)},_pending_forward=False)
        manager=SimpleNamespace(_terminated_buf=torch.zeros(4),_truncated_buf=torch.zeros(4),_term_dones={'x':torch.zeros(4)})
        env=SimpleNamespace(_eval_physics_graph=FakeGraph(),_eval_graph_warmed=False,device='cpu',num_envs=4,
                            _sim_step_counter=6,common_step_counter=8,
                            event_manager=SimpleNamespace(available_modes=[]),
                            command_manager=SimpleNamespace(get_term=lambda _:command),termination_manager=manager,
                            reward_manager=SimpleNamespace(_episode_sums={'x':torch.zeros(4)}))
        def reset():calls.append('reset');torch.rand(2)
        env.reset=reset
        torch.manual_seed(9);before=torch.random.get_rng_state().clone()
        benchmark.prepare_eval_graph(env);benchmark.prepare_eval_graph(env)
        self.assertEqual(calls,['reset','warmup'])
        self.assertEqual((env._sim_step_counter,env.common_step_counter),(6,8))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_graph_rejects_nonfinite_actions_and_overflow(self):
        class Diagnostics:
            def bits_tensor(self):return torch.tensor(0)
            def report(self):return {'overflow_bits':1}
        graph=SimpleNamespace(step=lambda action:('result',))
        env=SimpleNamespace(_eval_physics_graph=graph,physics_diagnostics=Diagnostics())
        with self.assertRaisesRegex(ValueError,'nonfinite'):benchmark.eval_graph_step(env,torch.tensor([float('nan')]))
        with patch.object(env.physics_diagnostics,'bits_tensor',return_value=torch.tensor(1)):
            with self.assertRaisesRegex(RuntimeError,'overflow'):benchmark.eval_graph_step(env,torch.ones(1))
        self.assertEqual(benchmark.eval_graph_step(env,torch.ones(1)),('result',))
