"""Native CPU playback contracts; no GPU needed for these tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import mujoco

from morphmimic.env.env_cfg import morphmimic_env_cfg
from morphmimic.model.rewards_torch import analytic_obs, goal_features, tracking_reward
from morphmimic.model.state import StateParts, join_state
from preview.cpu_backend import CpuPreviewEnv


CURRENT = Path(__file__).resolve().parents[1] / 'prepared/inplace-recovery-neutral-20261117-v2/manifest.json'
MANIFEST = Path(os.environ.get('BLINKMORPH_CPU_MANIFEST', str(CURRENT if CURRENT.is_file() else CURRENT.parent.parent/'allopt-20261112/manifest.json')))


@unittest.skipUnless(MANIFEST.is_file(), 'requires prepared native three-robot fixtures')
class NativeCpuPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.temp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_env(self, robot):
        row = next(r for r in self.manifest['clips'] if r['robot'] == robot and r['split'] == 'train' and r['dataset'] == 'seed')
        subset = Path(self.temp.name) / (robot + '.json')
        subset.write_text(json.dumps({'root': '/', 'clips': [row['output']]}))
        cfg = morphmimic_env_cfg(robot, subset_yaml=str(subset), num_envs=1, play=True, eval=True,
                                human_root=self.manifest['human_root'])
        cfg.commands['motion'].library_dtype = 'float32'
        env = CpuPreviewEnv(cfg, device='cpu')
        self.addCleanup(env.close)
        return env

    def test_native_reset_obs_and_sparse_mapping_all_robots(self):
        from mjlab.scene import Scene
        from mjlab.utils.lab_api.math import quat_apply_inverse
        for robot, joints, decimation in [('q1', 22, 10), ('bumi', 21, 10), ('g1', 29, 4)]:
            with self.subTest(robot=robot):
                env = self.make_env(robot)
                obs, _ = env.reset()
                model, data = env.sim.mj_model, env.native_data
                other = Scene(env.cfg.scene, 'cpu').compile()
                env.cfg.sim.mujoco.apply(other)
                for key in ('body_mass', 'body_inertia', 'geom_condim', 'geom_contype', 'geom_conaffinity',
                            'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'dof_armature'):
                    np.testing.assert_array_equal(getattr(model, key), getattr(other, key), err_msg=key)
                self.assertEqual(len(env.joint_qadr), joints)
                self.assertEqual(env.cfg.decimation, decimation)
                self.assertAlmostEqual(env.step_dt, .02)
                self.assertEqual(int(env.command.global_frame[0]), 1)
                lib = env.command.library
                expected_q = env.spec.translator.to_native(lib.joint_pos[:1].float(), env.default_joint_pos)
                joints = [model.joint("robot/" + name).id for name in env.spec.joint_names]
                ranges = model.jnt_range[joints]
                midpoint = ranges.mean(axis=1)
                half_width = (ranges[:, 1] - ranges[:, 0]) * env.cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor / 2
                expected_q = np.clip(expected_q[0].numpy(), midpoint - half_width, midpoint + half_width)
                np.testing.assert_allclose(data.qpos[env.joint_qadr], expected_q, atol=2e-7)
                np.testing.assert_allclose(data.qpos[:3], lib.root_pos_w[0], atol=2e-7)
                local_ang = quat_apply_inverse(lib.root_quat_w[:1].float(), lib.root_ang_vel_w[:1].float())
                np.testing.assert_allclose(data.qvel[3:6], local_ang[0], atol=2e-6)
                self.assertEqual(tuple(obs['policy'].shape), (1, 93))
                self.assertTrue(bool(torch.isfinite(obs['goal']).all()))
                self.assertTrue(lib.has_human)
                self.assertGreater(float(obs['goal_kp'].abs().sum()), 0)
                mask = torch.tensor(env.spec.translator.joint_mask)
                self.assertEqual(float(obs['policy'][0, :29][~mask].abs().sum()), 0)
                np.testing.assert_allclose(env.sim.data.qpos[0], data.qpos)

    def test_real_native_step_matches_independent_pd_substeps(self):
        for robot in ('q1', 'bumi', 'g1'):
            with self.subTest(robot=robot):
                env = self.make_env(robot)
                env.reset()
                oracle = mujoco.MjData(env.sim.mj_model)
                mujoco.mj_copyData(oracle, env.sim.mj_model, env.native_data)
                action = torch.linspace(-.12, .12, 29).reshape(1, 29)
                tr = env.spec.translator
                targets = env.default_joint_pos + tr.to_native_delta(action * env.action_scale)
                oracle.ctrl[:] = targets[0, env.actuator_native].numpy()
                for _ in range(env.cfg.decimation):
                    mujoco.mj_step(env.sim.mj_model, oracle)
                mujoco.mj_forward(env.sim.mj_model, oracle)
                before = env.command.global_frame.clone()
                obs, reward, terminated, truncated, _ = env.step(action)
                np.testing.assert_allclose(env.native_data.qpos, oracle.qpos, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(env.native_data.qvel, oracle.qvel, atol=1e-11, rtol=1e-11)
                self.assertEqual(int(env.command.global_frame[0]), int(before[0]) + 1)
                self.assertEqual(env.physics_diagnostics.report()['checked_physics_steps'], env.cfg.decimation)
                self.assertEqual(env.physics_diagnostics.report()['overflow_bits'], 0)
                expected, _ = tracking_reward(obs['state'], env.command.library.state(before), action * env.joint_mask,
                    torch.zeros_like(action), env.joint_limits, joint_mask=env.joint_mask)
                torch.testing.assert_close(reward, expected)
                self.assertEqual(terminated.dtype, torch.bool)
                self.assertEqual(truncated.dtype, torch.bool)

    def test_clip_end_and_failure_are_pre_advance_and_manual_reset(self):
        env = self.make_env('g1')
        env.reset()
        env.command.global_frame[:] = env.command.library.clip_len[0] - 1
        _, _, _, truncated, _ = env.step(torch.zeros(1, 29))
        self.assertTrue(bool(truncated[0]))
        self.assertEqual(int(env.command.global_frame[0]), int(env.command.library.clip_len[0]) - 1)
        env.reset()
        env.native_data.qpos[2] += 1.
        env.sim.forward()
        _, _, terminated, _, _ = env.step(torch.zeros(1, 29))
        self.assertTrue(bool(terminated[0]))
        self.assertTrue(bool(env.termination_manager.get_term('anchor_pos')[0]))

    def test_bad_action_fails_before_mutating_physics(self):
        env = self.make_env('q1')
        env.reset()
        before = env.native_data.qpos.copy()
        for action in (torch.full((1, 29), float('nan')), torch.zeros(2, 29), torch.zeros(1, 22)):
            with self.assertRaises(ValueError):
                env.step(action)
            np.testing.assert_array_equal(env.native_data.qpos, before)


if __name__ == '__main__':
    unittest.main()
