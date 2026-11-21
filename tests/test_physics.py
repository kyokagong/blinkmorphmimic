"""Native overflow history must survive resets and be visible at CPU boundaries."""
import unittest
from unittest.mock import patch

import torch


class PhysicsDiagnosticsTests(unittest.TestCase):
  def diagnostics(self, source):
    from morphmimic.env.physics import PhysicsDiagnostics
    engine = {'epa_horizon': 64, 'overflow_flags': {'CONTACT': 1, 'CONSTRAINT': 2},
              'coverage': 'native overflow bits; EPA face exhaustion has no diagnostic bit'}
    with patch('morphmimic.env.physics.engine_metadata', return_value=engine):
      return PhysicsDiagnostics(source)

  def test_native_reset_does_not_erase_sticky_history(self):
    native = torch.zeros(3, dtype=torch.int32)
    diag = self.diagnostics(native)
    self.assertFalse(diag.report()['zero_overflow_observed'])
    native[0] = 1
    diag.sample()
    native.zero_()  # MuJoCo native reset clears only its own flags.
    diag.sample()
    native[1] = 2
    diag.sample()
    native.zero_()
    report = diag.report()
    self.assertEqual(int(diag.bits_tensor()), 3)
    self.assertEqual(report['overflow_bits'], 3)
    self.assertEqual(report['overflow_worlds'], 2)
    self.assertEqual(report['checked_physics_steps'], 3)
    self.assertEqual(report['checked_step_worlds'], 9)
    self.assertEqual(report['flagged_step_world_observations'], 2)
    self.assertEqual(report['overflow_flags'], ['CONTACT', 'CONSTRAINT'])
    self.assertFalse(report['zero_overflow_observed'])
    self.assertIn('no diagnostic bit', report['engine']['coverage'])

  def test_repeated_flags_count_observations_and_unknown_flags(self):
    native = torch.tensor([8, 0], dtype=torch.int32)
    diag = self.diagnostics(native)
    diag.sample()
    diag.sample()
    self.assertEqual(diag.report()['flagged_step_world_observations'], 2)
    self.assertEqual(diag.report()['overflow_flags'], ['UNKNOWN_0x8'])
    native.zero_()
    clean = self.diagnostics(native)
    clean.sample()
    self.assertTrue(clean.report()['zero_overflow_observed'])

  def test_forward_overflow_is_sticky_without_counting_a_physics_step(self):
    native = torch.tensor([2, 0], dtype=torch.int32)
    diag = self.diagnostics(native)
    diag.sample_forward()
    native.zero_()
    report = diag.report()
    self.assertEqual(report['overflow_bits'], 2)
    self.assertEqual(report['checked_physics_steps'], 0)
    self.assertEqual(report['flagged_step_world_observations'], 0)
    self.assertEqual(report['checked_forward_calls'], 1)
    self.assertEqual(report['flagged_forward_world_observations'], 1)

  def test_forward_direct_constraint_capacity_is_sticky_and_counts_world_once(self):
    from morphmimic.env.physics import PhysicsDiagnostics
    native=torch.tensor([0, 1, 0],dtype=torch.int32)
    rows=torch.tensor([251, 251, 250],dtype=torch.int32)
    engine={'overflow_flags':{'NEFC':1}, 'coverage':'native flags and direct forward NEFC'}
    with patch('morphmimic.env.physics.engine_metadata',return_value=engine):
      diag=PhysicsDiagnostics(native,constraint_rows=rows,constraint_capacity=250)
    diag.sample_forward()
    self.assertEqual(native.tolist(),[0,1,0])
    native.zero_();rows.zero_()
    diag.sample_forward()
    report=diag.report()
    self.assertEqual(report['overflow_bits'],1)
    self.assertEqual(report['overflow_worlds'],2)
    self.assertEqual(report['checked_forward_calls'],2)
    self.assertEqual(report['flagged_forward_world_observations'],2)
    self.assertEqual(report['checked_physics_steps'],0)
    self.assertTrue(report['engine']['direct_forward_nefc'])
    self.assertEqual(report['engine']['constraint_capacity'],250)

  def test_constraint_capacity_binding_rejects_invalid_shapes_and_types(self):
    from morphmimic.env.physics import PhysicsDiagnostics
    native=torch.zeros(2,dtype=torch.int32)
    engine={'overflow_flags':{'NEFC':1}}
    with patch('morphmimic.env.physics.engine_metadata',return_value=engine):
      for rows,cap in ((torch.zeros(3,dtype=torch.int32),250),
                       (torch.zeros(2),250), (torch.zeros(2,dtype=torch.int32),0),
                       (torch.zeros(2,dtype=torch.int32),True), (None,250)):
        with self.subTest(rows=rows,capacity=cap), self.assertRaises(ValueError):
          PhysicsDiagnostics(native,constraint_rows=rows,constraint_capacity=cap)

  def test_forward_mirrors_native_sparse_and_global_contact_boundaries(self):
    from morphmimic.env.physics import PhysicsDiagnostics
    flags=torch.zeros(5,dtype=torch.int32)
    rows=torch.tensor([0,1,2,3,4],dtype=torch.int32)
    adr=torch.tensor([[999,999,999],[5,0,0],[0,6,0],[0,0,6],[0,0,999]],dtype=torch.int32)
    nnz=torch.tensor([[999,999,999],[5,0,0],[0,5,0],[0,0,4],[0,0,999]],dtype=torch.int32)
    collision=torch.tensor([20],dtype=torch.int32);contacts=torch.tensor([20],dtype=torch.int32)
    engine={'overflow_flags':{'NEFC':1,'NJMAX_NNZ':2,'BROADPHASE':4,'NARROWPHASE':8}}
    with patch('morphmimic.env.physics.engine_metadata',return_value=engine):
      diag=PhysicsDiagnostics(flags,constraint_rows=rows,constraint_capacity=3,
        sparse_rows=(adr,nnz,10),contact_counts=(collision,contacts,20))
    diag.sample_forward()
    self.assertEqual(diag.overflow_bits.tolist(),[0,0,2,0,1])
    self.assertEqual(flags.tolist(),[0]*5)
    rows.zero_();collision.fill_(21);contacts.fill_(21)
    diag.sample_forward()
    self.assertEqual(diag.overflow_bits.tolist(),[12,12,14,12,13])
    self.assertEqual(diag.report()['flagged_forward_world_observations'],7)
    self.assertEqual(diag.report()['checked_physics_steps'],0)

  def test_forward_bindings_reject_invalid_sparse_and_contact_tensors(self):
    from morphmimic.env.physics import PhysicsDiagnostics
    flags=torch.zeros(2,dtype=torch.int32);rows=flags.clone();matrix=torch.zeros(2,3,dtype=torch.int32)
    engine={'overflow_flags':{'NEFC':1,'NJMAX_NNZ':2,'BROADPHASE':4,'NARROWPHASE':8}}
    with patch('morphmimic.env.physics.engine_metadata',return_value=engine):
      for kwargs in (dict(sparse_rows=(matrix,matrix,10)),
                     dict(constraint_rows=rows,constraint_capacity=3,sparse_rows=(matrix[:,:2],matrix,10)),
                     dict(contact_counts=(flags,flags,10)),
                     dict(contact_counts=(flags[:1],flags[:1],0))):
        with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
          PhysicsDiagnostics(flags,**kwargs)


@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires frozen runtime')
class PhysicsLogGuardTests(unittest.TestCase):
  diagnostics = PhysicsDiagnosticsTests.diagnostics
  def test_log_flush_stops_for_overflow_even_on_nonwriting_rank(self):
    from types import SimpleNamespace
    from morphmimic.train.trainer import Trainer, _Window
    diag = self.diagnostics(torch.tensor([1], dtype=torch.int32))
    diag.sample()
    trainer = Trainer.__new__(Trainer)
    trainer.env = SimpleNamespace(physics_diagnostics=diag)
    trainer.cfg = SimpleNamespace(probe_focus_dir=None)
    trainer.command = SimpleNamespace(G_b=torch.ones(1), metrics={'sampling_entropy': torch.ones(1)})
    trainer.wm = SimpleNamespace(hazard=None)
    trainer.multi, trainer.world_size, trainer.device = False, 1, 'cpu'
    trainer._writes, trainer.it = False, 7
    trainer._add_perf = lambda *args: None
    with self.assertRaisesRegex(RuntimeError, 'physics overflow.*iteration 7'):
      trainer._log(_Window())
    self.assertEqual(trainer.last_log['physics/overflow_bits'], 1)
    # A clean local rank must also stop when the pooled MAX came from another rank.
    trainer.multi, trainer.world_size = True, 2
    trainer.env.physics_diagnostics = self.diagnostics(torch.zeros(1, dtype=torch.int32))
    trainer._robot_rows = lambda out: []
    window = _Window()
    window._pooled = lambda device, size: {'physics/overflow_bits': 8.}
    with self.assertRaisesRegex(RuntimeError, 'physics overflow'):
      trainer._log(window)



@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_TEST_DIAG_CUDA'), 'optional CUDA graph check')
class PhysicsDiagnosticsGraphTests(unittest.TestCase):
  def test_forward_graph_replays_with_changing_native_counts(self):
    from morphmimic.env.physics import PhysicsDiagnostics
    device='cuda:0'
    flags=torch.zeros(2,dtype=torch.int32,device=device);rows=torch.zeros_like(flags)
    adr=torch.zeros(2,3,dtype=torch.int32,device=device);nnz=torch.zeros_like(adr)
    collision=torch.zeros(1,dtype=torch.int32,device=device);contacts=torch.zeros_like(collision)
    diag=PhysicsDiagnostics(flags,rows,3,(adr,nnz,10),(collision,contacts,20))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): diag.sample_forward()
    rows.copy_(torch.tensor([2,4],device=device,dtype=torch.int32))
    adr[0,1]=7;nnz[0,1]=4;collision.fill_(21)
    graph.replay();torch.cuda.synchronize()
    self.assertEqual(diag.overflow_bits.tolist(),[6,5])
    self.assertEqual(flags.tolist(),[0,0])
    self.assertEqual(diag.report()['flagged_forward_world_observations'],2)


@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_G1_RESET_SUBSET'), 'requires native G1 stress fixture')
class RealGlobalPoolTests(unittest.TestCase):
  def test_forward_pool_overflow_is_reported_before_native_step(self):
    import json, os, warp as wp, mujoco_warp as mjw
    from pathlib import Path
    from morphmimic.env.env_cfg import morphmimic_env_cfg
    from morphmimic.env.morphmimic_env import MorphMimicEnv
    from morphmimic.train.collector import Collector
    cfg=morphmimic_env_cfg('g1',subset_yaml=os.environ['BLINKMORPH_G1_RESET_SUBSET'],num_envs=16,
      play=True,eval=True,human_root=os.environ['BLINKMORPH_HUMAN_ROOT'])
    cfg.sim.nconmax=35;cfg.sim.njmax=1024
    env=MorphMimicEnv(cfg,device='cuda:0',graph_stream=torch.cuda.Stream(device='cuda:0'))
    try:
      pose=json.loads((Path(__file__).parent/'data/g1_constraint_stress.json').read_text())['qpos']
      stream=env.tensor_step_stream;stream.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
        env.sim.reset();env.sim.data.qpos.copy_(torch.tensor(pose,device='cuda:0').repeat(16,1))
        env.sim.data.qvel.zero_();mjw.forward(env.sim.wp_model,env.sim.wp_data)
        native=env.physics_diagnostics.source.clone();env.physics_diagnostics.sample_forward()
      torch.cuda.synchronize()
      report=env.physics_diagnostics.report();data=env.sim.wp_data
      self.assertFalse(bool(native.any()));self.assertTrue(torch.equal(native,env.physics_diagnostics.source))
      self.assertLessEqual(int(wp.to_torch(data.nefc).max()),data.njmax)
      self.assertGreater(int(wp.to_torch(data.ncollision)[0]),data.naconmax)
      self.assertGreater(int(wp.to_torch(data.nacon)[0]),data.naconmax)
      self.assertEqual(set(report['overflow_flags']),{'BROADPHASE','NARROWPHASE'})
      guard=Collector.__new__(Collector);guard.env=env
      with self.assertRaisesRegex(RuntimeError,'BROADPHASE'):guard._check_graph_overflow('real global pool')
    finally:env.close()


@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_PHYSICS_SUBSET'), 'optional real GPU env check')
class RealPhysicsDiagnosticsTests(unittest.TestCase):
  def test_substep_count_and_reset_history(self):
    import json
    import os
    from morphmimic.env.env_cfg import morphmimic_env_cfg
    from morphmimic.env.morphmimic_env import MorphMimicEnv
    cfg = morphmimic_env_cfg('q1', subset_yaml=os.environ['BLINKMORPH_PHYSICS_SUBSET'], num_envs=1, eval=True)
    env = MorphMimicEnv(cfg, device=os.environ.get('BLINKMORPH_PHYSICS_DEVICE', 'cuda:0'))
    try:
      env.reset()
      diag = env.physics_diagnostics
      env.step(torch.zeros(1, 29, device=env.device))
      first = diag.report()
      self.assertEqual(first['checked_physics_steps'], 10)
      self.assertEqual(first['overflow_bits'], 0)
      # Inject only into this disposable diagnostic view to verify sticky reset behavior.
      diag.source[0] = 1
      diag.sample()
      diag.source.zero_()
      env.reset()
      self.assertEqual(diag.report()['overflow_bits'], 1)
      self.assertEqual(diag.report()['checked_physics_steps'], 11)
      env.step(torch.zeros(1, 29, device=env.device))
      after = diag.report()
      self.assertEqual(after['checked_physics_steps'], 21)
      self.assertEqual(after['overflow_bits'], 1)
      print(json.dumps({'actual_first_control': first, 'after_synthetic_flag_and_reset': after}))
    finally:
      env.close()


if __name__ == '__main__':
  unittest.main()
