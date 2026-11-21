import threading
import time
import unittest

import numpy as np
from preview.comparison import AsyncComparisonRenderer


class AsyncRendererTests(unittest.TestCase):
    def test_latest_snapshot_is_coherent_and_reset_discards_old_frame(self):
        started, release = threading.Event(), threading.Event()
        ownership, rendered = [], []
        class Renderer:
            def __init__(self): ownership.append(('init', threading.get_ident()))
            def frame_rgb(self, poses, reset):
                ownership.append(('render', threading.get_ident()))
                rendered.append((poses, reset))
                if len(rendered) == 1:
                    started.set(); release.wait(2)
                return np.array([poses['q1'][0], poses['bumi'][0], poses['g1'][0]])
            def close(self): ownership.append(('close', threading.get_ident()))
        worker = AsyncComparisonRenderer(Renderer)
        try:
            worker.submit({r: np.array([0]) for r in ('q1','bumi','g1')}, reset=True)
            self.assertTrue(started.wait(2))
            worker.submit({r: np.array([1]) for r in ('q1','bumi','g1')})
            worker.submit({r: np.array([2]) for r in ('q1','bumi','g1')}, reset=True)
            release.set()
            end = time.monotonic() + 2
            result = None
            while time.monotonic() < end:
                result = worker.latest()
                if result is not None: break
                time.sleep(.005)
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result[1], [2,2,2])
            self.assertEqual(len(rendered), 2)
            self.assertTrue(rendered[-1][1])
        finally:
            release.set(); worker.close()
        self.assertEqual(len({thread for _,thread in ownership}), 1)
        self.assertNotEqual(ownership[0][1], threading.get_ident())

    def test_wait_returns_rendered_frame_and_propagates_failure(self):
        class Renderer:
            def frame_rgb(self, poses, reset): return np.array([poses['q1'][0]])
            def close(self): pass
        worker = AsyncComparisonRenderer(Renderer)
        try:
            worker.submit({'q1': np.array([7])}, reset=True)
            np.testing.assert_array_equal(worker.wait_after(0)[1], [7])
        finally: worker.close()


class JointReleaseTests(unittest.TestCase):
    def test_joint_identity_binds_robot_iteration_training_manifest_and_mode(self):
        from preview.release import verify_human_identity
        ck = dict(iteration=10000, cfg=dict(robot='q1', num_envs=4096, optimization_schema=1, kp_interface=True,
            kp_fixed_horizon=1, policy_goal='motion', optimizer='muon'),
            optimized_training=dict(schema_version=1, iteration=10000, optimizer='muon', num_envs=4096,
                joint_motion_human=True, human_horizon=1, human_points='five', prepared_manifest_sha256='a'*64))
        verify_human_identity(ck, robot='q1', update=10000, data_sha256='b'*64, training_sha256='a'*64)
        for field,value in [('prepared_manifest_sha256','c'*64),('human_horizon',2),('num_envs',1)]:
            old=ck['optimized_training'][field];ck['optimized_training'][field]=value
            with self.assertRaises(ValueError):
                verify_human_identity(ck, robot='q1', update=10000, data_sha256='b'*64, training_sha256='a'*64)
            ck['optimized_training'][field]=old
        with self.assertRaises(ValueError):
            verify_human_identity(ck, robot='g1', update=10000, data_sha256='b'*64, training_sha256='a'*64)

    def test_joint_identity_accepts_only_matching_2048_or_4096_environments(self):
        from preview.release import verify_human_identity
        ck=dict(iteration=5000,cfg=dict(robot='q1',num_envs=2048,optimization_schema=1,kp_interface=True,
            kp_fixed_horizon=1,policy_goal='motion',optimizer='muon'),
            optimized_training=dict(schema_version=1,iteration=5000,optimizer='muon',num_envs=2048,
                joint_motion_human=True,human_horizon=1,human_points='five',prepared_manifest_sha256='a'*64))
        def verify():verify_human_identity(ck,robot='q1',update=5000,data_sha256='b'*64,training_sha256='a'*64)
        for n in (2048,4096):
            ck['cfg']['num_envs']=n;ck['optimized_training']['num_envs']=n;verify()
        for cfg,meta in ((2048,4096),(4096,2048),(1024,1024),(True,True),(None,2048)):
            ck['cfg']['num_envs']=cfg;ck['optimized_training']['num_envs']=meta
            with self.subTest(cfg=cfg,meta=meta),self.assertRaises(ValueError):verify()

    def test_2048_checkpoint_loads_in_both_joint_release_interfaces(self):
        import json,tempfile,torch
        from pathlib import Path
        from preview.server import load_release_catalogs,sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            (data/'manifest.json').write_text(json.dumps(dict(complete=True,human_data=True)))
            training=root/'training.json';training.write_text('{}');digest=sha256(training);checks={}
            for robot in ('q1','bumi','g1'):
                (data/robot).mkdir();path=root/robot/'ckpt_5000.pt';path.parent.mkdir()
                torch.save(dict(iteration=5000,cfg=dict(robot=robot,num_envs=2048,optimization_schema=1,kp_interface=True,
                    kp_fixed_horizon=1,policy_goal='motion',optimizer='muon'),optimized_training=dict(schema_version=1,
                    iteration=5000,optimizer='muon',num_envs=2048,joint_motion_human=True,human_horizon=1,
                    human_points='five',prepared_manifest_sha256=digest)),path)
                checks[robot]=dict(path=str(path),sha256=sha256(path))
            common=dict(data_root=str(data),checkpoints=checks,parent_iteration=5000,human_updates=5000,
                        training_manifest=dict(path=str(training),sha256=digest))
            rows=[dict(id='joint-motion-5000',label='motion',interface='motion',**common),
                  dict(id='joint-human-5000',label='human',interface='human',**common)]
            path=root/'releases.json';path.write_text(json.dumps(rows))
            self.assertEqual(set(load_release_catalogs(path)),{'joint-motion-5000','joint-human-5000'})

    def test_release_pins_separate_training_and_human_catalog_manifests(self):
        import json
        from pathlib import Path
        import tempfile
        import torch
        from preview.server import load_release_catalogs, sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; data.mkdir()
            (data/'manifest.json').write_text(json.dumps(dict(complete=True,human_data=True)))
            training=root/'training.json';training.write_text('{}'); digest=sha256(training)
            checks={}
            for robot in ('q1','bumi','g1'):
                (data/robot).mkdir()
                path=root/robot/'ckpt_10000.pt';path.parent.mkdir()
                torch.save(dict(iteration=10000,cfg=dict(robot=robot,num_envs=4096,optimization_schema=1,kp_interface=True,
                    kp_fixed_horizon=1,policy_goal='motion',optimizer='muon'),
                    optimized_training=dict(schema_version=1,iteration=10000,optimizer='muon',num_envs=4096,
                        joint_motion_human=True,human_horizon=1,human_points='five',prepared_manifest_sha256=digest)),path)
                checks[robot]=dict(path=str(path),sha256=sha256(path))
            release=dict(id='joint',label='Joint human',data_root=str(data),checkpoints=checks,interface='human',
                parent_iteration=10000,human_updates=10000,training_manifest=dict(path=str(training),sha256=digest))
            path=root/'releases.json';path.write_text(json.dumps([release]))
            catalog=load_release_catalogs(path)['joint']['catalog']
            self.assertEqual(catalog.training_manifest_sha256,digest)
            training.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'manifest'):
                load_release_catalogs(path)

@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class TensorStepGuards(unittest.TestCase):
    def test_rejects_auto_reset_events_recorders_and_expired_command(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import torch
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        env=SimpleNamespace(motion_command_name='motion')
        env.cfg=SimpleNamespace(auto_reset=False,observations={'policy':SimpleNamespace(nan_policy='disabled')})
        env.event_manager=SimpleNamespace(available_modes=[])
        env.sim=SimpleNamespace(nan_guard=SimpleNamespace(enabled=False))
        env.recorder_manager=SimpleNamespace(active_terms=[])
        env.metrics_manager=SimpleNamespace(active_terms=[])
        env.command_manager=MagicMock()
        env.command_manager.active_terms=['motion']
        command=env.command_manager.get_term.return_value
        command.time_left=torch.tensor([1.]);command._pending_forward=False
        env.step_dt=.02
        env._manual_reset_pending=torch.zeros(1,dtype=torch.bool)
        MorphMimicEnv.validate_tensor_step(env)
        for target,field,value in [(env.cfg,'auto_reset',True),(env.event_manager,'available_modes',['interval']),
                                   (env.recorder_manager,'active_terms',['recorder']),(env.metrics_manager,'active_terms',['metric']),(env.sim.nan_guard,'enabled',True)]:
            previous=getattr(target,field);setattr(target,field,value)
            with self.assertRaises(ValueError): MorphMimicEnv.validate_tensor_step(env)
            setattr(target,field,previous)
        env.cfg.observations['policy'].nan_policy='error'
        with self.assertRaises(ValueError): MorphMimicEnv.validate_tensor_step(env)
        env.cfg.observations['policy'].nan_policy='disabled'
        command.time_left.zero_()
        with self.assertRaises(ValueError): MorphMimicEnv.validate_tensor_step(env)

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_PHYSICS_SUBSET'), 'requires disposable GPU physics fixture')
class RealTensorStepTests(unittest.TestCase):
    def test_capture_matches_eager_after_reset(self):
        import os
        import torch
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from preview.control_graph import PhysicsGraph
        import warp as wp
        subset=os.environ['BLINKMORPH_PHYSICS_SUBSET']
        device=os.environ.get('BLINKMORPH_PHYSICS_DEVICE','cuda:0')
        robot=os.environ.get('BLINKMORPH_PHYSICS_ROBOT','q1')
        human_root=os.environ.get('BLINKMORPH_HUMAN_ROOT')
        eager=MorphMimicEnv(morphmimic_env_cfg(robot,subset_yaml=subset,num_envs=1,play=True,eval=True,human_root=human_root),device=device)
        stream=torch.cuda.Stream(device=device)
        captured=MorphMimicEnv(morphmimic_env_cfg(robot,subset_yaml=subset,num_envs=1,play=True,eval=True,human_root=human_root),device=device,graph_stream=stream)
        envs=[eager,captured]
        try:
            self.assertFalse(captured.sim.use_cuda_graph)
            self.assertIsNone(captured.sim.step_graph)
            self.assertIsNone(captured.sim.forward_graph)
            self.assertIsNone(captured.sim.reset_graph)
            captured.reset()
            PhysicsGraph(captured).warmup(torch.zeros((1,29),device=device))
            self.assertEqual(captured.physics_diagnostics.report()['overflow_bits'],0)
            graph=PhysicsGraph(captured)
            if human_root:
                self.assertTrue(captured.command_manager.get_term('motion').library.has_human)
            for run in range(2):
                graph.reset()
                eager.reset()
                captured.reset()
                torch.cuda.synchronize(device)
                for env in envs: env.validate_tensor_step()
                action=torch.linspace(-.1,.1,29,device=device).unsqueeze(0)
                expected=eager.step(action)
                actual=graph.step(action)
                for _ in range(2):
                    torch.testing.assert_close(captured.sim.data.qpos,eager.sim.data.qpos,atol=1e-5,rtol=1e-5)
                    torch.testing.assert_close(captured.sim.data.qvel,eager.sim.data.qvel,atol=1e-4,rtol=1e-4)
                    for key in ('policy','goal','goal_kp'):
                        torch.testing.assert_close(actual[0][key],expected[0][key],atol=1e-4,rtol=1e-4)
                    torch.testing.assert_close(actual[1],expected[1],atol=1e-4,rtol=1e-4)
                    self.assertTrue(torch.equal(actual[2],expected[2]))
                    self.assertTrue(torch.equal(actual[3],expected[3]))
                    expected=eager.step(action)
                    actual=graph.step(action)
                self.assertEqual(captured.physics_diagnostics.report()['overflow_bits'],0)
        finally:
            for env in envs: env.close()

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_PHYSICS_SUBSET'), 'requires disposable GPU physics fixture')
class PartialResetGraphTests(unittest.TestCase):
    def test_replay_after_one_environment_resets_without_recapture(self):
        import os
        import torch
        from morphmimic.env.env_cfg import morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.env.control_graph import PhysicsGraph
        subset=os.environ['BLINKMORPH_PHYSICS_SUBSET'];device='cuda:0'
        def config(): return morphmimic_env_cfg('q1',subset_yaml=subset,num_envs=2,play=True,eval=True)
        eager=MorphMimicEnv(config(),device=device)
        captured=MorphMimicEnv(config(),device=device,graph_stream=torch.cuda.Stream(device=device))
        try:
            eager.reset();captured.reset()
            control=PhysicsGraph(captured)
            action=torch.linspace(-.1,.1,29,device=device).repeat(2,1)
            control.warmup(torch.zeros_like(action));captured.reset()
            eager.step(action);control.step(action)
            ids=torch.tensor([0],device=device)
            eager.reset(env_ids=ids);captured.reset(env_ids=ids)
            expected=eager.step(action);actual=control.step(action)
            for key in ('policy','goal','goal_kp'):
                torch.testing.assert_close(actual[0][key],expected[0][key],atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(captured.sim.data.qpos,eager.sim.data.qpos,atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(captured.sim.data.qvel,eager.sim.data.qvel,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(actual[1],expected[1],atol=1e-4,rtol=1e-4)
            self.assertTrue(torch.equal(actual[2],expected[2]))
            self.assertTrue(torch.equal(actual[3],expected[3]))
        finally:
            eager.close();captured.close()


class PhysicsGraphTests(unittest.TestCase):
    def test_replay_revalidates_boundaries_and_reset_recaptures(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from preview.control_graph import PhysicsGraph
        class Env:
            device='cpu';cfg=SimpleNamespace(decimation=10)
            _sim_step_counter=0;common_step_counter=0
            def __init__(self):self.validations=0;self.actions=[]
            def validate_tensor_step(self):self.validations+=1
            def step_no_reset_tensor(self,action):
                self.actions.append(action);self._sim_step_counter+=10;self.common_step_counter+=1
                return action
        class Graph:
            def __init__(self):self.replays=0
            def replay(self):self.replays+=1
        env=Env()
        with patch('torch.cuda.CUDAGraph',Graph),patch('torch.cuda.graph',return_value=nullcontext()),patch('torch.cuda.device',return_value=nullcontext()):
            control=PhysicsGraph(env)
            control.step(torch.tensor([1.]));first=control.graph
            control.step(torch.tensor([2.]))
            self.assertEqual(first.replays,2);self.assertEqual(env.validations,2)
            self.assertEqual((env._sim_step_counter,env.common_step_counter),(20,2))
            self.assertEqual(float(env.actions[0][0]),2.)
            control.reset();control.step(torch.tensor([3.]))
            self.assertIsNot(control.graph,first)

    def test_warmup_checks_overflow_before_any_capture(self):
        from types import SimpleNamespace
        import torch
        from preview.control_graph import PhysicsGraph
        class Env:
            physics_diagnostics=SimpleNamespace(report=lambda: {'overflow_bits': 4})
            def validate_tensor_step(self): pass
            def step_no_reset_tensor(self,action): return action
        graph=PhysicsGraph(Env())
        with self.assertRaisesRegex(RuntimeError,'overflow'):
            graph.warmup(torch.zeros(1))
        self.assertIsNone(graph.graph)

    def test_actor_replays_with_new_inputs_and_keeps_output_storage(self):
        from contextlib import nullcontext
        from unittest.mock import patch
        import torch
        from preview.control_graph import ActorGraph
        class Graph:
            def __init__(self): self.replays=0
            def replay(self): self.replays+=1
        with patch('torch.cuda.CUDAGraph',Graph),patch('torch.cuda.graph',return_value=nullcontext()),patch('torch.cuda.device',return_value=nullcontext()),patch('torch.cuda.synchronize') as sync:
            actor=ActorGraph(lambda x: x, 'cpu')
            first=actor(torch.tensor([1.]))
            second=actor(torch.tensor([2.]))
            self.assertIs(first,second)
            self.assertEqual(actor.graph.replays,2)
            self.assertEqual(float(second[0]),2.)
            sync.assert_called_once()
            with self.assertRaises(ValueError): actor(torch.tensor([[1.]]))

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_PHYSICS_SUBSET'), 'requires disposable GPU fixture')
class RealActionGraphTests(unittest.TestCase):
    def test_graph_matches_masked_native_action_mapping(self):
        from types import SimpleNamespace
        import torch
        from morphmimic.env.actions import CanonicalJointPositionAction
        term=CanonicalJointPositionAction.__new__(CanonicalJointPositionAction)
        term.cfg=SimpleNamespace(clip=None)
        term._mask=torch.tensor([True,False,True,True],device='cuda')
        term._active=torch.tensor([0,2,3],device='cuda')
        term._perm=torch.tensor([2,0,1],device='cuda')
        term._sign=torch.tensor([1.,-1.,1.],device='cuda')
        term._scale_canon=torch.tensor([[2.,3.,4.]],device='cuda')
        term._raw_actions=torch.zeros((1,4),device='cuda')
        term.executed=torch.zeros((1,4),device='cuda')
        term._processed_actions=torch.zeros((1,3),device='cuda')
        term._offset=torch.tensor([[.1,.2,.3]],device='cuda')
        term.noise_std=0.
        action=torch.tensor([[.4,.5,.6,.7]],device='cuda')
        term.process_actions(action)
        expected=torch.tensor([[-1.7,3.,1.1]],device='cuda')
        torch.testing.assert_close(term._processed_actions,expected)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): term.process_actions(action)
        graph.replay()
        torch.testing.assert_close(term._processed_actions,expected)

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_PHYSICS_SUBSET'), 'requires disposable GPU fixture')
class RealTranslatorGraphTests(unittest.TestCase):
    def test_cached_joint_mapping_replays_and_preserves_gradients(self):
        import torch
        from morphmimic.constants import JOINT_NAMES
        from morphmimic.robots import Translator
        tr=Translator.from_names((JOINT_NAMES[0],JOINT_NAMES[-1]), {}, {}, {})
        q=torch.tensor([[.3,-.2]],device='cuda',requires_grad=True)
        expected=tr.to_canonical(q)
        expected.sum().backward()
        torch.testing.assert_close(q.grad,torch.ones_like(q))
        static=q.detach().clone()
        graph=torch.cuda.CUDAGraph()
        default=torch.zeros(2,device='cuda')
        tr.to_native(expected.detach(),default)
        with torch.cuda.graph(graph):
            out=tr.to_canonical(static)
            native=tr.to_native(out,default)
        graph.replay()
        torch.testing.assert_close(out,expected.detach())
        torch.testing.assert_close(native,static)
        static.copy_(torch.tensor([[.7,.4]],device='cuda'));graph.replay()
        torch.testing.assert_close(tr.to_native(out,torch.zeros(2,device='cuda')),static)


@unittest.skipUnless(__import__('importlib').util.find_spec('mjlab'), 'requires mjlab runtime')
class GraphRewardTests(unittest.TestCase):
    def test_cached_indices_match_original_on_rotated_selected_bodies(self):
        from types import SimpleNamespace
        import torch
        from mjlab.tasks.tracking.mdp import rewards
        from morphmimic.env.graph_rewards import prepare_graph_rewards
        names=['motion_relative_body_position_error_exp','motion_relative_body_orientation_error_exp',
               'motion_global_body_linear_velocity_error_exp','motion_global_body_angular_velocity_error_exp']
        torch.manual_seed(42)
        q=torch.randn(2,4,4); q=q/q.norm(dim=-1,keepdim=True)
        q2=torch.randn(2,4,4); q2=q2/q2.norm(dim=-1,keepdim=True)
        command=SimpleNamespace(cfg=SimpleNamespace(body_names=('a','b','c','untracked')),
            body_pos_relative_w=torch.randn(2,4,3),robot_body_pos_w=torch.randn(2,4,3),
            body_quat_relative_w=q,robot_body_quat_w=q2,
            body_lin_vel_w=torch.randn(2,4,3),robot_body_lin_vel_w=torch.randn(2,4,3),
            body_ang_vel_w=torch.randn(2,4,3),robot_body_ang_vel_w=torch.randn(2,4,3))
        env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:command))
        params=dict(command_name='motion',std=.37,body_names=('a','c'))
        expected=[getattr(rewards,name)(env,**params) for name in names]
        terms={name:SimpleNamespace(func=getattr(rewards,name),params=dict(params)) for name in names}
        cfg=SimpleNamespace(rewards=terms,commands={'motion':command.cfg})
        prepare_graph_rewards(cfg,'cpu')
        for name,want in zip(names,expected):
            torch.testing.assert_close(terms[name].func(env,**params),want)

class TranslatorGraphTests(unittest.TestCase):
    def test_constants_are_reused_per_device_dtype_and_preserve_gradients(self):
        import torch
        from morphmimic.constants import JOINT_NAMES
        from morphmimic.robots import Translator
        tr=Translator.from_names((JOINT_NAMES[0],JOINT_NAMES[-1]), {}, {}, {})
        q=torch.tensor([[.3,-.2]],requires_grad=True)
        first=tr._tensors(q); second=tr._tensors(q)
        self.assertTrue(all(a is b for a,b in zip(first,second)))
        canon=tr.to_canonical(q)
        torch.testing.assert_close(tr.to_native(canon,torch.zeros(2)),q)
        canon.sum().backward()
        torch.testing.assert_close(q.grad,torch.ones_like(q))
        alternate=tr._tensors(q.double())
        self.assertNotEqual(first[1].dtype,alternate[1].dtype)

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_PHYSICS_SUBSET'), 'requires disposable GPU fixture')
class RealImuGraphTests(unittest.TestCase):
    def test_capture_keeps_native_imu_selection_and_shape(self):
        from types import SimpleNamespace
        import torch
        from morphmimic.env.observations import _imu
        q=torch.arange(32.,device='cuda').reshape(2,4,4)
        w=torch.arange(24.,device='cuda').reshape(2,4,3)
        cmd=SimpleNamespace(body_indexes=torch.tensor([3,1,2],device='cuda'),
            spec=SimpleNamespace(imu_slot=1),robot=SimpleNamespace(data=SimpleNamespace(body_link_quat_w=q,body_link_ang_vel_w=w)))
        env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:cmd))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): quat,angular=_imu(env,'motion')
        graph.replay()
        torch.testing.assert_close(quat,q[:,1]);torch.testing.assert_close(angular,w[:,1])
        self.assertEqual(tuple(quat.shape),(2,4))

@unittest.skipUnless(__import__('os').environ.get('BLINKMORPH_PHYSICS_SUBSET'), 'requires disposable GPU fixture')
class RealActorGraphTests(unittest.TestCase):
    def test_first_capture_and_replay_match_eager_outputs(self):
        import torch
        from preview.control_graph import ActorGraph
        torch.manual_seed(77)
        model=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.Tanh(),torch.nn.Linear(16,4)).cuda().eval()
        actor=ActorGraph(model,'cuda:0')
        with torch.no_grad():
            for _ in range(3):
                x=torch.randn(3,8,device='cuda')
                torch.testing.assert_close(actor(x),model(x))


def test_comparison_keeps_simulating_held_targets_past_shortest_clip():
    from types import SimpleNamespace
    from preview.comparison import ComparisonSession
    session=ComparisonSession.__new__(ComparisonSession)
    session.failed=session.completed=False;session.errors={};session.executor=None;session.clip_frames=3
    session.policy=lambda sessions:{r:None for r in sessions}
    class Child:
        def __init__(self):self.steps=0;self.failed=self.completed=False;self.hold_frame=1
        def step(self,action):self.steps+=1
    session.sessions={r:Child() for r in ('q1','bumi','g1')}
    for _ in range(10):session.step()
    assert session.steps==10
    assert not session.completed
