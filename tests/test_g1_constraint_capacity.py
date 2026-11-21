"""Actual native G1 constraint capacity, contact-pool stress and overflow guards."""
import json,os,unittest
from pathlib import Path
import torch


@unittest.skipUnless(os.environ.get('BLINKMORPH_G1_RESET_SUBSET'), 'requires native G1 GPU fixture')
class G1ConstraintCapacityTests(unittest.TestCase):
    def test_old_capacity_overflows_and_new_handles_all_worlds_fallen(self):
        import mujoco,numpy as np,warp as wp,mujoco_warp as mjw
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.train.collector import Collector
        fixture=json.loads((Path(__file__).parent/'data/g1_constraint_stress.json').read_text())
        results=[];models=[];n=16;device='cuda:0'
        for old in (True,False):
            cfg=morphmimic_env_cfg('g1',subset_yaml=os.environ['BLINKMORPH_G1_RESET_SUBSET'],num_envs=n,
                play=True,eval=True,human_root=os.environ['BLINKMORPH_HUMAN_ROOT'])
            cfg.commands['motion'].allow_missing_human=True
            if old:cfg.sim.nconmax,cfg.sim.njmax=35,250
            env=MorphMimicEnv(cfg,device=device,graph_stream=torch.cuda.Stream(device=device))
            try:
                m=env.sim.mj_model;data=mujoco.MjData(m);data.qpos[:]=fixture['qpos'];mujoco.mj_forward(m,data)
                self.assertEqual(int(data.ncon),fixture['cpu_ncon']);self.assertEqual(int(data.nefc),fixture['cpu_nefc'])
                self.assertEqual(int(m.opt.cone),int(mujoco.mjtCone.mjCONE_PYRAMIDAL))
                self.assertLessEqual(int(m.geom_condim.max()),3)
                self.assertEqual((m.neq,m.ntendon,int((m.dof_frictionloss>0).sum())),(0,0,0))
                models.append({name:getattr(m,name).copy() for name in ('geom_condim','geom_friction','jnt_range','actuator_gainprm','actuator_biasprm','dof_damping','dof_armature')})
                models[-1]['physics_options']=(float(m.opt.timestep),int(m.opt.solver),int(m.opt.iterations),int(m.opt.cone))
                stream=env.tensor_step_stream;stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
                    env.sim.reset()
                    q=torch.as_tensor(m.qpos0,device=device,dtype=torch.float32).repeat(n,1);q[:,2]=3
                    q[:1 if old else n]=torch.tensor(fixture['qpos'],device=device)
                    env.sim.data.qpos.copy_(q);env.sim.data.qvel.zero_()
                    mjw.forward(env.sim.wp_model,env.sim.wp_data)
                    native_flags=env.physics_diagnostics.source.clone()
                    env.physics_diagnostics.sample_forward()
                torch.cuda.synchronize(device)
                forward_stats=env.physics_diagnostics.report()
                self.assertEqual(int(native_flags.count_nonzero()),0)
                self.assertTrue(torch.equal(native_flags,env.physics_diagnostics.source))
                self.assertEqual('NEFC' in forward_stats['overflow_flags'],old)
                print('FORWARD_CAPACITY_RESULT',cfg.sim.njmax,forward_stats['overflow_flags'])
                with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
                    # NEFC is checked by native integration's next_time kernel.
                    mjw.step(env.sim.wp_model,env.sim.wp_data)
                    env.physics_diagnostics.sample()
                torch.cuda.synchronize(device)
                stats=env.physics_diagnostics.report();nefc=wp.to_torch(env.sim.wp_data.nefc).cpu();nacon=int(wp.to_torch(env.sim.wp_data.nacon)[0])
                guard=Collector.__new__(Collector);guard.env=env
                if old:
                    self.assertIn('NEFC',stats['overflow_flags'])
                    self.assertGreater(int(nefc[0]),250)
                    with self.assertRaisesRegex(RuntimeError,'NEFC'):guard._check_graph_overflow('capacity regression')
                else:
                    self.assertEqual(stats['overflow_bits'],0,stats)
                    guard._check_graph_overflow('capacity regression')
                    self.assertTrue(bool((nefc>250).all()))
                    self.assertGreater(nacon,35*n)
                    self.assertLessEqual(nacon,cfg.sim.nconmax*n)
                    self.assertTrue(bool(env.sim.data.qacc.isfinite().all()))
                    self.assertGreaterEqual(cfg.sim.njmax,4*cfg.sim.nconmax+int(m.jnt_limited.sum()))
                results.append((cfg.sim.nconmax,cfg.sim.njmax,int(nefc.max()),nacon,stats['overflow_flags']))
            finally:env.close()
        for key in models[0]:np.testing.assert_equal(models[0][key],models[1][key],err_msg=key)
        print('NATIVE_CAPACITY_RESULTS',results)
