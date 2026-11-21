"""Sparse canonical slots and real Q1/Bumi/G1 asset contracts."""
import unittest

import torch

from morphmimic.constants import JOINT_NAMES
from morphmimic.robots import ROBOTS, Translator, get_robot


class TranslatorTests(unittest.TestCase):
  def test_sparse_roundtrip_and_missing_actions(self):
    native = (JOINT_NAMES[0], JOINT_NAMES[-1])
    tr = Translator.from_names(native, {}, {}, {})
    self.assertEqual(tr.perm[1], -1)
    self.assertEqual(sum(tr.joint_mask), 2)
    q = torch.tensor([[.3, -.2]])
    canon = tr.to_canonical(q)
    torch.testing.assert_close(canon[:, 1:-1], torch.zeros(1, 27))
    torch.testing.assert_close(tr.to_native(canon, torch.zeros(2)), q)
    actions = torch.randn(4, 29, requires_grad=True)
    masked = actions.detach().clone()
    masked[:, 1:-1] = 10000
    torch.testing.assert_close(tr.to_native_delta(actions), tr.to_native_delta(masked))
    tr.to_native_delta(actions).sum().backward()
    torch.testing.assert_close(actions.grad[:, 1:-1], torch.zeros(4, 27))
    limits = tr.limits_to_canonical(torch.tensor([[-1., 1.], [-2., 2.]]))
    torch.testing.assert_close(limits[1:-1], torch.zeros(27, 2))

  def test_sparse_sign_offset_and_validation(self):
    tr = Translator.from_names((JOINT_NAMES[0],), {}, {JOINT_NAMES[0]: -1}, {JOINT_NAMES[0]: .2})
    torch.testing.assert_close(tr.to_canonical(torch.tensor([.5]))[0], torch.tensor(-.3))
    torch.testing.assert_close(tr.to_native(tr.to_canonical(torch.tensor([.5])), torch.zeros(1)), torch.tensor([.5]))
    with self.assertRaises(ValueError):
      Translator.from_names((JOINT_NAMES[0],), {JOINT_NAMES[1]: JOINT_NAMES[0]}, {}, {})
    with self.assertRaises(ValueError):
      Translator.from_names((JOINT_NAMES[0],), {}, {JOINT_NAMES[0]: 0}, {})


class PolicyMaskTests(unittest.TestCase):
  def test_policy_output_and_missing_input_gradients(self):
    from morphmimic.model.policy import Policy, PolicyCfg
    policy = Policy(PolicyCfg(enc_hidden=(8,), dec_hidden=(8,), hist_len=1, latent=4))
    mask = torch.ones(29)
    mask[14] = 0
    policy.set_joint_mask(mask)
    hist = torch.randn(2, 93, requires_grad=True)
    goal = torch.randn(2, 670, requires_grad=True)
    action = policy(hist, goal)
    torch.testing.assert_close(action[:, 14], torch.zeros(2))
    action.sum().backward()
    torch.testing.assert_close(hist.grad[:, [14, 43, 78]], torch.zeros(2, 3))
    torch.testing.assert_close(goal.grad.view(2, 10, 67)[:, :, [14, 43]], torch.zeros(2, 10, 2))
    clone = Policy(policy.cfg)
    clone.load_state_dict(policy.state_dict())
    torch.testing.assert_close(clone(hist.detach(), goal.detach()), action.detach())


  def test_explicit_morphology_uses_its_own_mask(self):
    from morphmimic.model.policy import Policy, PolicyCfg
    for width in (154, 145):
      with self.subTest(width=width):
        policy = Policy(PolicyCfg(enc_hidden=(8,), dec_hidden=(8,), hist_len=1, latent=4, morph_dim=width))
        for layer in policy.modules():
          if isinstance(layer, torch.nn.Linear):
            torch.nn.init.constant_(layer.weight, .01)
            torch.nn.init.constant_(layer.bias, .1)
        q1_mask = torch.ones(29)
        q1_mask[14] = 0
        policy.set_joint_mask(q1_mask)
        policy.set_morphology(torch.cat([torch.zeros(width - 29), q1_mask]))
        physical_before = {name: value.clone() for name, value in policy.named_buffers()}
        hist = torch.zeros(2, 93, requires_grad=True)
        goal = torch.zeros(2, 670, requires_grad=True)
        g1_morph = torch.cat([torch.zeros(width - 29), torch.ones(29)])
        g1 = policy(hist, goal, g1_morph)
        self.assertTrue((g1[:, 14] > 0).all())
        g1.sum().backward()
        self.assertTrue((hist.grad[:, [14, 43, 78]] > 0).all())
        self.assertTrue((goal.grad.view(2, 10, 67)[:, :, [14, 43]] > 0).all())
        hist.grad.zero_()
        goal.grad.zero_()
        # The opposite override also removes a slot present in the default robot.
        bumi_morph = g1_morph.clone()
        bumi_morph[-29 + 13] = 0
        bumi = policy(hist, goal, bumi_morph)
        torch.testing.assert_close(bumi[:, 13], torch.zeros(2))
        batch = policy(hist, goal, torch.stack([g1_morph, bumi_morph]))
        torch.testing.assert_close(batch[0], g1[0])
        torch.testing.assert_close(batch[1], bumi[1])
        bumi.sum().backward()
        torch.testing.assert_close(hist.grad[:, [13, 42, 77]], torch.zeros(2, 3))
        torch.testing.assert_close(goal.grad.view(2, 10, 67)[:, :, [13, 42]], torch.zeros(2, 10, 2))
        torch.testing.assert_close(policy(hist, goal)[:, 14], torch.zeros(2))
        for name, before in physical_before.items():
          torch.testing.assert_close(dict(policy.named_buffers())[name], before)


@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires frozen mjlab runtime')
class AssetTests(unittest.TestCase):
  def test_bumi_entity_preserves_raw_body_collisions_and_foot_material(self):
    import re
    import mujoco
    import numpy as np
    from mjlab.entity import Entity
    from morphmimic.robots.bumi import get_robot_cfg

    cfg = get_robot_cfg()
    raw = cfg.spec_fn().compile()
    entity = Entity(cfg)
    model = entity.spec.compile()
    active = (raw.geom_contype != 0) | (raw.geom_conaffinity != 0)
    self.assertEqual(int(active.sum()), 13)
    np.testing.assert_array_equal(model.geom_contype, raw.geom_contype)
    np.testing.assert_array_equal(model.geom_conaffinity, raw.geom_conaffinity)
    for field in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_dataid'):
      np.testing.assert_array_equal(getattr(model, field), getattr(raw, field))
    feet = [i for i in range(model.ngeom) if re.fullmatch(r'(l|r)_foot_collision_.*', model.geom(i).name or '')]
    self.assertEqual(len(feet), 2)
    np.testing.assert_array_equal(model.geom_condim[feet], [3, 3])
    np.testing.assert_array_equal(model.geom_priority[feet], [1, 1])
    np.testing.assert_allclose(model.geom_friction[feet, 0], .6)
    other = np.setdiff1d(np.arange(model.ngeom), feet)
    for field in ('geom_condim', 'geom_priority', 'geom_friction'):
      np.testing.assert_array_equal(getattr(model, field)[other], getattr(raw, field)[other])
    self.assertTrue((model.geom_contype[~active] == 0).all())
    self.assertTrue((model.geom_conaffinity[~active] == 0).all())

    entity.spec.worldbody.add_geom(name='test_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0., 0., 1.), contype=1, conaffinity=1)
    model = entity.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('init_state').id)
    # A disposable horizontal pose tests contact filtering, not motion feasibility.
    data.qpos[:7] = [0., 0., 0., np.sqrt(.5), 0., np.sqrt(.5), 0.]
    mujoco.mj_forward(model, data)
    floor = model.geom('test_floor').id
    floor_contacts = [int(contact.geom2 if contact.geom1 == floor else contact.geom1)
        for contact in data.contact if floor in (contact.geom1, contact.geom2)]
    nonfeet = [i for i in floor_contacts if not re.fullmatch(r'(l|r)_foot_collision_.*', model.geom(i).name or '')]
    self.assertTrue(nonfeet, 'original non-foot collision meshes must contact the floor')
    self.assertTrue(all(model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0 for i in floor_contacts))

  def test_registry_assets_and_descriptor(self):
    import mujoco
    import numpy as np
    from morphmimic.robots.morphology import morphology_vector
    self.assertEqual(ROBOTS, ('q1', 'bumi', 'g1'))
    for name, width in [('q1', 22), ('bumi', 21), ('g1', 29)]:
      with self.subTest(robot=name):
        spec = get_robot(name)
        model = spec.entity_cfg_fn().spec_fn().compile()
        joints = tuple(model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE)
        self.assertEqual(joints, spec.joint_names)
        self.assertEqual(len(joints), width)
        self.assertEqual(sum(spec.translator.joint_mask), width)
        self.assertEqual(len(set(spec.tracked_bodies)), 14)
        self.assertTrue(all(model.body(b).id > 0 for b in spec.tracked_bodies))
        self.assertIn(spec.imu_body, spec.tracked_bodies)
        import re
        self.assertTrue(any(re.fullmatch(spec.foot_geom_regex, model.geom(i).name or '') for i in range(model.ngeom)))
        vec = morphology_vector(name)
        self.assertEqual(tuple(vec.shape), (154,))
        self.assertTrue(torch.isfinite(vec).all())
        torch.testing.assert_close(vec[-29:], torch.tensor(spec.translator.joint_mask, dtype=torch.float32))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if name == 'bumi':
          self.assertLess(spec.translator.sign[JOINT_NAMES.index('left_elbow_joint')], 0)
          self.assertGreater(np.linalg.norm(data.body(spec.tracked_bodies[10]).xpos - data.body(spec.tracked_bodies[9]).xpos), .05)


@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires frozen mjlab runtime')
class PhysicsMaskTests(unittest.TestCase):
  def test_mujoco_observation_and_reward(self):
    import mujoco
    import numpy as np
    from morphmimic.model.rewards_torch import analytic_obs, tracking_reward
    from morphmimic.model.state import StateParts, join_state
    for name in ROBOTS:
      spec = get_robot(name)
      model = spec.entity_cfg_fn().spec_fn().compile()
      data = mujoco.MjData(model)
      data.qvel[:] = np.linspace(-.1, .1, model.nv)
      mujoco.mj_forward(model, data)
      ids = [model.body(n).id for n in spec.tracked_bodies]
      linear, angular = [], []
      for body in ids:
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jp, jr, body)
        linear.append(jp @ data.qvel)
        angular.append(jr @ data.qvel)
      def tensor(x):
        return torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(0)
      native_joints = [model.joint(n).id for n in spec.joint_names]
      q = tensor(data.qpos[model.jnt_qposadr[native_joints]])
      qd = tensor(data.qvel[model.jnt_dofadr[native_joints]])
      state = join_state(StateParts(spec.translator.to_canonical(q), spec.translator.to_canonical_delta(qd), tensor(data.xpos[ids]), tensor(data.xquat[ids]), tensor(linear), tensor(angular)))
      mask = torch.tensor(spec.translator.joint_mask)
      obs = analytic_obs(state, torch.zeros(29), torch.zeros(29), torch.zeros(1, 29), spec.imu_slot, mask)
      rotation = data.xmat[model.body(spec.imu_body).id].reshape(3, 3)
      np.testing.assert_allclose(obs[0, 58:61], rotation.T @ angular[spec.imu_slot], atol=1e-6)
      np.testing.assert_allclose(obs[0, 61:64], rotation.T @ np.array([0., 0., -1.]), atol=1e-6)
      torch.testing.assert_close(obs[:, :29], spec.translator.to_canonical(q))
      limits = torch.tensor([[-10., 10.]] * 29)
      action = torch.zeros(1, 29)
      action[:, ~mask] = 1000
      total, terms = tracking_reward(state, state, action, torch.zeros_like(action), limits, joint_mask=mask)
      torch.testing.assert_close(total, torch.tensor([.1]), atol=1e-6, rtol=1e-5)
      torch.testing.assert_close(terms['action_rate'], torch.zeros(1))

  def test_world_model_ignores_missing_actions_and_joint_losses(self):
    from morphmimic.model.state import StateNormalizer, StateParts, join_state, split_state
    from morphmimic.model.world_model import WorldModel, WorldModelCfg
    mask = torch.ones(29)
    mask[14] = 0
    wm = WorldModel(WorldModelCfg(hidden=(8,)), StateNormalizer(), torch.tensor([[-10., 10.]] * 29), torch.zeros(29), joint_mask=mask)
    # Nonzero learned weights ensure this checks action masking, rather than zero initialization.
    torch.nn.init.normal_(wm.dynamics_mlp[-1].weight, std=.01)
    quat = torch.zeros(2, 14, 4)
    quat[..., 0] = 1
    state = join_state(StateParts(torch.zeros(2, 29), torch.zeros(2, 29), torch.zeros(2, 14, 3), quat, torch.zeros(2, 14, 3), torch.zeros(2, 14, 3)))
    action = torch.randn(2, 29, requires_grad=True)
    ctx = torch.zeros(2, 33)
    pred = wm.step(state, action, ctx)
    changed = action.detach().clone()
    changed[:, 14] = 10000
    torch.testing.assert_close(pred, wm.step(state, changed, ctx))
    torch.testing.assert_close(split_state(pred).joint_pos[:, 14], torch.zeros(2))
    torch.testing.assert_close(split_state(pred).joint_vel[:, 14], torch.zeros(2))
    pred.sum().backward()
    torch.testing.assert_close(action.grad[:, 14], torch.zeros(2))
    from morphmimic.train.replay import WindowBatch
    class Library:
      def state(self, frames):
        return state[:, None].expand(-1, frames.shape[1], -1)
    logged = torch.stack([state, state], dim=1)
    batch = WindowBatch(logged, action.detach()[:, None], torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.bool), ctx, torch.zeros(2, 2, dtype=torch.long))
    loss, info = wm.fit_loss(batch, Library())
    changed_logged = logged.clone()
    changed_logged[:, 1, 14] = 1000
    changed_logged[:, 1, 43] = -1000
    changed_loss, changed_info = wm.fit_loss(batch._replace(state=changed_logged), Library())
    torch.testing.assert_close(loss, changed_loss)
    torch.testing.assert_close(info['state_loss'], changed_info['state_loss'])
    noisy = wm.mask_obs(torch.ones(2, 3, 93))
    torch.testing.assert_close(noisy[..., [14, 43, 78]], torch.zeros(2, 3, 3))

  def test_sensor_corruption_has_zero_missing_slots(self):
    from morphmimic.env.env_cfg import morphmimic_env_cfg
    for name in ROBOTS:
      spec = get_robot(name)
      cfg = morphmimic_env_cfg(name, num_envs=1)
      self.assertAlmostEqual(cfg.sim.mujoco.timestep * cfg.decimation, .02)
      for key in ('joint_pos', 'joint_vel'):
        noise = cfg.observations['policy'].terms[key].noise
        actual = noise.apply(torch.zeros(3, 29))
        missing = ~torch.tensor(spec.translator.joint_mask)
        torch.testing.assert_close(actual[:, missing], torch.zeros(3, int(missing.sum())))


if __name__ == '__main__':
  unittest.main()
