"""Source-body coverage and evaluation must exclude Q1 entry/return padding."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from morphmimic.constants import FPS, R_MAX
from morphmimic.data.motion_library import MotionLibrary


def write_motion(path, frames=12, robot='q1'):
    path.parent.mkdir(parents=True, exist_ok=True)
    q = np.zeros((frames, 1, 4), dtype=np.float32); q[..., 0] = 1
    v = np.zeros((frames, 1, 3), dtype=np.float32)
    np.savez(path, fps=FPS, robot=robot, joint_pos=np.zeros((frames, 29), dtype=np.float32),
             joint_vel=np.zeros((frames, 29), dtype=np.float32), body_names=np.array(['root']),
             body_pos_w=v, body_quat_w=q, body_lin_vel_w=v, body_ang_vel_w=v)


class SourceWindowTests(unittest.TestCase):
    def test_manifest_robot_clip_lookup_ceil_boundaries_and_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            a, b = root / 'q1/a.npz', root / 'g1/b.npz'
            write_motion(a); write_motion(b, frames=7, robot='g1')
            (root / 'manifest.json').write_text(json.dumps({'clips': [
                {'robot': 'g1', 'name': 'a', 'phases': {'entry': 0, 'dance': .22}},
                {'robot': 'q1', 'name': 'a', 'phases': {'entry': .021, 'dance': .151}},
                {'robot': 'g1', 'name': 'b'},
            ]}))
            lib = MotionLibrary([str(a), str(b)], body_names=('root',))
            torch.testing.assert_close(lib.source_start, torch.tensor([2, 0]))
            torch.testing.assert_close(lib.source_stop, torch.tensor([8, 7]))
            torch.testing.assert_close(lib.source_has_phases, torch.tensor([True, False]))
            frames = torch.tensor([0, 1, 2, 7, 8, 11, 12, 17, 18])
            torch.testing.assert_close(lib.source_mask(frames), torch.tensor([False, False, True, True, False, False, True, True, False]))

    def test_exact_decimal_phase_boundaries_do_not_gain_a_return_frame_from_float_roundoff(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); path = root / 'q1/a.npz'; write_motion(path, frames=600)
            (root / 'manifest.json').write_text(json.dumps({'clips': [{'robot': 'q1', 'name': 'a',
                'phases': {'entry': .14, 'dance': 9.8}}]}))
            lib = MotionLibrary([str(path)], body_names=('root',))
            self.assertEqual(int(lib.source_start[0]), 7)
            self.assertEqual(int(lib.source_stop[0]), 490)

    def test_usage_preserves_full_counts_but_source_ignores_padding_terminal_and_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); p = root / 'q1/a.npz'; write_motion(p)
            (root / 'manifest.json').write_text(json.dumps({'clips': [{'robot': 'q1', 'name': 'a', 'phases': {'entry': .04, 'dance': .16}}]}))
            lib = MotionLibrary([str(p)], body_names=('root',))
            lib.record_usage('world_model', torch.tensor([[0, 2, 3], [7, 8, 11]]), torch.tensor([[True, True, False], [True, True, True]]))
            torch.testing.assert_close(lib.usage['world_model'], torch.tensor([5]))
            torch.testing.assert_close(lib.source_usage['world_model'], torch.tensor([2]))
            torch.testing.assert_close(lib.source_usage['policy'], torch.tensor([0]))

    def test_unknown_or_missing_phase_metadata_uses_full_transition_window(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'q1/a.npz'; write_motion(p, frames=4)
            lib = MotionLibrary([str(p)], body_names=('root',))
            torch.testing.assert_close(lib.source_start, torch.tensor([0]))
            torch.testing.assert_close(lib.source_stop, torch.tensor([4]))
            torch.testing.assert_close(lib.source_mask(torch.arange(4)), torch.tensor([True, True, True, False]))

    def test_source_coverage_roundtrips_and_legacy_resume_starts_new_counter_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); path = root / 'q1/a.npz'; write_motion(path)
            lib = MotionLibrary([str(path)], body_names=('root',))
            lib.record_usage('policy', torch.tensor([0, 1, 11]))
            lib.source_usage_since_iteration = 120
            state = lib.source_coverage_state()
            checkpoint = root / 'coverage.pt'
            torch.save(state, checkpoint)
            lib.source_usage['policy'].zero_()
            lib.restore_source_coverage(torch.load(checkpoint, weights_only=False), 5000)
            torch.testing.assert_close(lib.source_usage['policy'], torch.tensor([2]))
            self.assertEqual(lib.source_usage_since_iteration, 120)
            with self.assertWarnsRegex(UserWarning, 'iteration 5000'):
                lib.restore_source_coverage(None, 5000)
            torch.testing.assert_close(lib.source_usage['policy'], torch.tensor([0]))
            torch.testing.assert_close(lib.usage['policy'], torch.tensor([3]))
            self.assertEqual(lib.source_usage_since_iteration, 5000)

    def test_benchmark_keeps_padding_separate_and_failure_before_body_scores_zero(self):
        from unittest.mock import patch
        from morphmimic.constants import GOAL_DIM, HUMAN_ENV_DIM, PROPRIO_DIM
        from morphmimic.env.commands import MultiMotionCommand
        from morphmimic.eval import benchmark
        lib = SimpleNamespace(has_human=False,num_clips=3, clip_names=['before', 'body', 'full'],
                              clip_len=torch.tensor([7, 7, 7]), clip_start=torch.tensor([0, 7, 14]),
                              source_has_phases=torch.tensor([True, True, False]),
                              source_start=torch.tensor([2, 2, 0]), source_stop=torch.tensor([5, 5, 7]),
                              frame_clip=torch.repeat_interleave(torch.arange(3), 7))
        command = MultiMotionCommand.__new__(MultiMotionCommand)
        command.library = lib
        command.cfg = SimpleNamespace(sampling='start')
        command.global_frame = torch.zeros(1, dtype=torch.long)
        command.env_clip = torch.zeros(1, dtype=torch.long)
        command.assign_clips = lambda ids, clips: command.env_clip.__setitem__(ids, clips)
        obs = {'policy': torch.zeros(1, PROPRIO_DIM), 'goal': torch.zeros(1, GOAL_DIM),
               'goal_kp': torch.zeros(1, HUMAN_ENV_DIM)}
        def reset(env_ids):
            command.global_frame[env_ids] = lib.clip_start[command.env_clip[env_ids]] + 1
            return obs, {}
        def step(action):
            clip = int(command.env_clip[0])
            frame = int(command.global_frame[0] - lib.clip_start[clip])
            reward = 100. if clip < 2 and frame not in (2, 3, 4) else R_MAX
            terminated = torch.tensor([clip == 0])
            truncated = torch.tensor([frame == 6])
            command.global_frame[0] += int(frame < 6)
            return obs, torch.tensor([reward]), terminated, truncated, {}
        env = SimpleNamespace(device='cpu', num_envs=1, step_dt=.02, reset=reset, step=step,
                              command_manager=SimpleNamespace(get_term=lambda _: command))
        trace = SimpleNamespace(record=lambda _: None, take=lambda ids, steps: [(0,)] * len(ids))
        with patch.object(benchmark, 'DriftTrace', return_value=trace), patch.object(benchmark, 'summarize_drift', return_value={k: 0. for k in benchmark.DRIFT_SUMMARY_COLUMNS}):
            result = benchmark.run_benchmark(env, lambda history, goals: torch.zeros(1, 29))
        self.assertEqual(result['source_total_steps'].tolist(), [3, 3, 6])
        self.assertEqual(result['source_steps'].tolist(), [0, 3, 6])
        np.testing.assert_allclose(result['source_norm_return'].tolist(), [0., 1., 1.], atol=1e-7)
        self.assertGreater(result['norm_return'][1], result['source_norm_return'][1])
        self.assertEqual(result['norm_return'][2], result['source_norm_return'][2])

    def test_summary_reports_body_return_and_completion_separately(self):
        import pandas as pd
        from morphmimic.eval.benchmark import DRIFT_SUMMARY_COLUMNS, summarize
        rows = pd.DataFrame({'name': ['a', 'b'], 'failed': [True, False],
                             'norm_return': [.9, .4], 'source_norm_return': [.1, .4],
                             'source_steps': [10, 20], 'source_total_steps': [100, 20],
                             **{k: [0., 0.] for k in DRIFT_SUMMARY_COLUMNS}})
        result = summarize(rows, pd.DataFrame({'name': ['a', 'b'], 'category': ['dance', 'walk']}))
        self.assertAlmostEqual(result['mean_norm_return'], .65)
        self.assertAlmostEqual(result['source']['mean_norm_return'], .25)
        self.assertEqual(result['source']['steps'], 30)
        self.assertEqual(result['source']['total_steps'], 120)
        self.assertEqual(result['source']['completed_clips'], 1)
        self.assertAlmostEqual(result['source']['mean_step_fraction'], .55)

    def test_source_reward_mask_cannot_count_high_padding_rewards(self):
        from morphmimic.eval.benchmark import source_reward_mask
        lib = SimpleNamespace(source_has_phases=torch.tensor([True, False]),
                              source_start=torch.tensor([2, 0]), source_stop=torch.tensor([4, 5]),
                              clip_start=torch.tensor([0, 6]), clip_len=torch.tensor([6, 6]),
                              frame_clip=torch.tensor([0] * 6 + [1] * 6))
        lib.source_mask = lambda frames: ((frames - lib.clip_start[lib.frame_clip[frames]] >= lib.source_start[lib.frame_clip[frames]])
                                         & (frames - lib.clip_start[lib.frame_clip[frames]] < lib.source_stop[lib.frame_clip[frames]]))
        frames = torch.tensor([1, 2, 3, 4, 5])
        rewards = torch.tensor([100., R_MAX, R_MAX / 2, 100., 100.])
        source = source_reward_mask(lib, frames)
        self.assertEqual(int(source.sum()), 2)
        self.assertAlmostEqual(float(rewards[source].sum() / (R_MAX * 2)), .75)
        # A robot without phase metadata keeps the existing full-clip benchmark score.
        self.assertTrue(source_reward_mask(lib, frames + 6).all())


if __name__ == '__main__':
    unittest.main()
