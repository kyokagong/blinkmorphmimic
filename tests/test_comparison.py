"""Comparison control regressions; real-asset tests run in the frozen runtime."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeSession:
    def __init__(self, robot, path, checkpoint, device, *, render, source_body, shared_policy=None):
        assert render is False and source_body is True
        self.robot, self.device = robot, device
        self.frames = {"q1": 6, "bumi": 5, "g1": 7}[robot]
        self.received_actions = []
        self.closed = False
        self.failure_at = self.error_at = None
        self.barrier = None
        self.reset()

    def reset(self):
        self.steps = 0
        self.failed = self.completed = False

    def step(self, action=None):
        self.received_actions.append(action)
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        if self.steps == self.error_at:
            raise RuntimeError("native overflow fixture")
        self.steps += 1
        self.failed = self.steps == self.failure_at
        self.completed = self.steps == self.frames - 1

    def status(self):
        return dict(steps=self.steps, failed=self.failed, completed=self.completed,
                    clip_frames=self.frames, source_start_frame=150 if self.robot == "q1" else 0,
                    source_stop_frame=156 if self.robot == "q1" else self.frames,
                    source_frames=406 if self.robot == "q1" else self.frames)

    def close(self):
        self.closed = True


class FakeSharedPolicy:
    def __init__(self, checkpoints, device):
        self.calls = 0

    def __call__(self, sessions):
        self.calls += 1
        return dict(q1=11, bumi=22, g1=33)


class ComparisonTests(unittest.TestCase):
    def make_session(self, devices=("cpu",) * 3):
        from preview.comparison import ComparisonSession
        session = ComparisonSession({r: Path(r + '.npz') for r in ('q1', 'bumi', 'g1')},
                                    dict.fromkeys(('q1', 'bumi', 'g1'), object()), devices,
                                    session_factory=FakeSession, policy_factory=FakeSharedPolicy)
        self.addCleanup(session.close)
        return session

    def test_one_shared_forward_per_tick_dispatches_the_correct_robot_row(self):
        session = self.make_session()
        session.step()
        session.step()
        self.assertEqual(session.policy.calls, 2)
        self.assertEqual([s.received_actions for s in session.sessions.values()], [[11, 11], [22, 22], [33, 33]])

    def test_shared_inference_failure_does_not_advance_any_world(self):
        session = self.make_session()
        def broken(_):
            raise ValueError('nonfinite shared action')
        broken.calls = 0
        session.policy = broken
        session.step()
        self.assertTrue(session.failed)
        self.assertEqual([s.steps for s in session.sessions.values()], [0, 0, 0])
        self.assertIn('nonfinite', str(session.status()))

    def test_reset_restarts_the_actor_counter_with_the_world_counters(self):
        session=self.make_session()
        session.step();session.step();session.reset()
        self.assertEqual(session.policy.calls,0)
        session.step()
        self.assertEqual(session.status()['actor_forward_calls'],1)
        self.assertEqual(session.steps,1)

    def test_all_worlds_stop_at_shortest_body_and_reset_together(self):
        session = self.make_session()
        for _ in range(9):
            session.step()
        status = session.status()
        self.assertEqual(session.steps, 4)
        self.assertTrue(session.completed)
        self.assertFalse(session.failed)
        self.assertEqual({s['robot']: s['steps'] for s in status['participants']},
                         dict(q1=4, bumi=4, g1=4))
        self.assertEqual(status['clip_frames'], 5)
        self.assertEqual(status['participants'][0]['source_start_frame'], 150)
        session.reset()
        self.assertEqual(session.steps, 0)
        self.assertFalse(session.completed)
        self.assertEqual([s.steps for s in session.sessions.values()], [0, 0, 0])

    def test_failure_preserves_every_terminal_pose_until_explicit_reset(self):
        session = self.make_session()
        session.sessions['bumi'].failure_at = 2
        for _ in range(8):
            session.step()
        self.assertTrue(session.failed)
        self.assertEqual(session.steps, 2)
        self.assertEqual(session.status()['failed_robots'], ['bumi'])
        self.assertEqual([s.steps for s in session.sessions.values()], [2, 2, 2])
        session.reset()
        session.step()
        self.assertEqual([s.steps for s in session.sessions.values()], [1, 1, 1])

    def test_independent_devices_reach_a_step_barrier_before_next_control_step(self):
        session = self.make_session(('cuda:0', 'cuda:1', 'cuda:2'))
        barrier = threading.Barrier(3)
        for child in session.sessions.values():
            child.barrier = barrier
        session.step()
        session.step()
        self.assertFalse(session.failed)
        self.assertEqual([s.steps for s in session.sessions.values()], [2, 2, 2])

    def test_first_capture_and_recapture_are_serial_but_replay_is_parallel(self):
        session=self.make_session(('cuda:0','cuda:1','cuda:2'))
        lock=threading.Lock();active=0;peak=0
        for child in session.sessions.values():
            child.physics_graph=SimpleNamespace(graph=None)
            original=child.step
            def capture(action, child=child, original=original):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                time.sleep(.03)
                try:original(action);child.physics_graph.graph=object()
                finally:
                    with lock:active-=1
            child.step=capture
        session.step();self.assertFalse(session.failed);self.assertEqual(peak,1)
        peak=0;session.step();self.assertEqual(peak,3)
        session.reset()
        for child in session.sessions.values():child.physics_graph.graph=None
        peak=0;session.step();self.assertFalse(session.failed);self.assertEqual(peak,1)

    def test_step_exception_stops_group_and_exposes_failed_participant(self):
        session = self.make_session(('cuda:0', 'cuda:1', 'cuda:2'))
        session.sessions['g1'].error_at = 0
        session.step()
        session.step()
        self.assertTrue(session.failed)
        self.assertEqual(session.status()['failed_robots'], ['g1'])
        self.assertIn('native overflow', session.status()['participants'][2]['error'])
        self.assertEqual([s.steps for s in session.sessions.values()], [1, 1, 0])
        session.reset()
        self.assertFalse(session.failed)
        self.assertNotIn('error', session.status()['participants'][2])

    def test_partial_failed_step_never_submits_mixed_time_pose(self):
        session = self.make_session()
        session.sessions['g1'].error_at = 0
        session.step()
        session.submit_frame()
        self.assertIsNone(session.renderer)

    def test_partial_construction_closes_already_created_worlds(self):
        from preview.comparison import ComparisonSession
        built = []
        def factory(*args, **kwargs):
            if len(built) == 1:
                raise RuntimeError('fixture load failed')
            built.append(FakeSession(*args, **kwargs))
            return built[-1]
        with self.assertRaisesRegex(RuntimeError, 'load failed'):
            ComparisonSession(dict.fromkeys(('q1', 'bumi', 'g1'), Path('clip.npz')),
                              dict.fromkeys(('q1', 'bumi', 'g1'), object()), ('cpu',) * 3,
                              session_factory=factory, policy_factory=FakeSharedPolicy)
        self.assertTrue(built[0].closed)


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'requires frozen MuJoCo runtime and assets')
class SourceBodyTests(unittest.TestCase):
    def test_headless_policy_world_does_not_import_or_construct_an_individual_renderer(self):
        from preview.server import PolicySession
        environment = MagicMock(step_dt=.02, device='cpu')
        checkpoint = SimpleNamespace(path=Path('fixed.pt'), iteration=20, verify=lambda: None)
        meta = SimpleNamespace(robot='q1', iteration=20, hist_len=1,interface='motion')
        torch = MagicMock()
        torch.load.return_value = {'cfg': {}}
        modules = {
            'torch': torch,
            'mjlab.viewer': None,
            'mjlab.utils.torch': SimpleNamespace(configure_torch_backends=lambda **kwargs: None),
            'morphmimic.env.env_cfg': SimpleNamespace(COMMAND_NAME='motion', morphmimic_env_cfg=lambda *a, **kw: MagicMock()),
            'morphmimic.env.morphmimic_env': SimpleNamespace(MorphMimicEnv=lambda *a, **kw: environment),
            'morphmimic.eval.policy_loader': SimpleNamespace(load_policy=lambda *a, **kw: (MagicMock(), meta)),
            'morphmimic.train.obs_history': SimpleNamespace(ObsHistory=MagicMock()),
        }
        with patch.dict(sys.modules, modules), patch.object(PolicySession, 'reset'):
            session = PolicySession('q1', Path('/unused/dance.npz'), checkpoint, 'cpu', render=False)
            try:
                self.assertIsNone(session.renderer)
                with self.assertRaisesRegex(ValueError, 'no individual renderer'):
                    session.frame_rgb()
            finally:
                session.close()

    def test_body_crop_preserves_metadata_and_original_file(self):
        import numpy as np
        from preview.server import PolicySession
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'q1').mkdir()
            path = root / 'q1/dance.npz'
            q = np.arange(210 * 2, dtype=np.float32).reshape(210, 2)
            np.savez(path, fps=50, joint_pos=q, joint_vel=q + 1,
                     body_pos_w=np.arange(210 * 3).reshape(210, 1, 3),
                     joint_names=['native_a', 'native_b'], body_names=['pelvis'], robot='q1')
            row = dict(robot='q1', name='dance', source_id='source/dance', frames=210,
                       phases=dict(entry=3, dance=3.2, stand=4.2), original_sha256='provenance')
            (root / 'manifest.json').write_text(json.dumps(dict(clips=[row])))
            before = path.read_bytes()
            session = PolicySession.__new__(PolicySession)
            session.human_root = None
            session.tmp = tempfile.TemporaryDirectory(dir=folder)
            self.addCleanup(session.tmp.cleanup)
            crop = session._prepare_source_body('q1', path)
            with np.load(crop, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['joint_pos'], q[150:160])
                np.testing.assert_array_equal(z['joint_vel'], q[150:160] + 1)
                self.assertEqual(z['joint_names'].tolist(), ['native_a', 'native_b'])
                self.assertEqual(z['body_names'].tolist(), ['pelvis'])
                self.assertEqual(z['robot'].item(), 'q1')
                self.assertEqual(z['fps'].item(), 50)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(session.source_metadata, row)
            self.assertEqual(session.source_window['start_frame'], 150)
            self.assertEqual(session.source_window['stop_frame'], 160)
            self.assertEqual(session.source_window['offset_seconds'], 3)

    def test_human_sidecar_is_cropped_with_the_same_robot_window(self):
        import numpy as np
        from preview.server import PolicySession
        from morphmimic.data.human_convert import sidecar_path
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'q1').mkdir();path=root/'q1/dance.npz'
            np.savez(path,fps=50,joint_pos=np.zeros((210,2)))
            (root/'manifest.json').write_text(json.dumps(dict(clips=[dict(robot='q1',name='dance',phases=dict(entry=3,dance=3.2))])))
            human_root=root/'human50';side=sidecar_path(path,human_root);side.parent.mkdir(parents=True)
            positions=np.arange(210*24*3).reshape(210,24,3)
            np.savez(side,human_pos_w=positions,human_quat_w=np.zeros((210,24,4)),source_time_s=np.arange(210)/50,human_joint_valid=np.ones(24,dtype=bool),fps=50)
            before=side.read_bytes();session=PolicySession.__new__(PolicySession)
            session.human_root=str(human_root);session.tmp=tempfile.TemporaryDirectory(dir=folder)
            self.addCleanup(session.tmp.cleanup)
            cropped=session._prepare_source_body('q1',path)
            with np.load(sidecar_path(cropped,session.human_root)) as z:
                np.testing.assert_array_equal(z['human_pos_w'],positions[150:160])
                np.testing.assert_array_equal(z['source_time_s'],np.arange(210)[150:160]/50)
                assert z['human_joint_valid'].shape==(24,) and int(z['fps'])==50
            assert side.read_bytes()==before

    def test_invalid_body_window_refuses_empty_or_single_frame_clip(self):
        import numpy as np
        from preview.server import PolicySession
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'q1').mkdir()
            path = root / 'q1/dance.npz'
            np.savez(path, fps=50, joint_pos=np.zeros((200, 2)))
            (root / 'manifest.json').write_text(json.dumps(dict(clips=[dict(
                robot='q1', name='dance', frames=200, phases=dict(entry=3, dance=3.01))])))
            session = PolicySession.__new__(PolicySession)
            session.tmp = tempfile.TemporaryDirectory(dir=folder)
            self.addCleanup(session.tmp.cleanup)
            with self.assertRaisesRegex(ValueError, 'two frames'):
                session._prepare_source_body('q1', path)


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'requires frozen MuJoCo runtime and assets')
class RealSceneTests(unittest.TestCase):
    def test_namespaced_physics_world_maps_by_names_with_unrelated_joint_before_robot(self):
        import mujoco
        from morphmimic.robots import get_robot
        from preview.comparison import ComparisonRenderer
        sessions = {}
        joints = dict(q1='left_knee_joint', bumi='l_knee_pitch_joint', g1='left_knee_joint')
        values = dict(q1=.24, bumi=-.31, g1=.47)
        for robot in ('q1', 'bumi', 'g1'):
            world = mujoco.MjSpec.from_string('''<mujoco><worldbody><body name="unrelated">
              <joint name="unrelated"/><geom type="sphere" size=".01" mass=".1"/>
            </body></worldbody></mujoco>''')
            child = get_robot(robot).entity_cfg_fn().spec_fn()
            child.option.timestep = world.option.timestep
            child.njmax, child.nconmax = world.njmax, world.nconmax
            world.attach(child, prefix='robot/', frame=world.worldbody.add_frame())
            model = world.compile()
            qpos = model.qpos0.copy()
            qpos[0] = 9
            qpos[model.joint('robot/' + joints[robot]).qposadr[0]] = values[robot]
            sessions[robot] = SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(
                mj_model=model, data=SimpleNamespace(qpos=qpos[None]))))
        visual = ComparisonRenderer(sessions)
        self.addCleanup(visual.close)
        visual.reset({r: sessions[r].env.sim.data.qpos[0].copy() for r in sessions})
        visual.sync({r: sessions[r].env.sim.data.qpos[0].copy() for r in sessions})
        for robot in ('q1', 'bumi', 'g1'):
            self.assertAlmostEqual(visual.data.joint(robot + '/' + joints[robot]).qpos[0], values[robot])

    def test_asset_meshes_native_mapping_and_common_heading_keep_physical_sizes(self):
        import mujoco
        import numpy as np
        from morphmimic.robots import get_robot
        from preview.comparison import ComparisonRenderer

        sessions, sources = {}, {}
        for index, robot in enumerate(('q1', 'bumi', 'g1')):
            model = get_robot(robot).entity_cfg_fn().spec_fn().compile()
            data = mujoco.MjData(model)
            free = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
            addr = model.jnt_qposadr[free]
            data.qpos[addr:addr + 7] = [10 + index, -5, (.45, .7, .8)[index],
                                      np.sqrt(.5), 0, 0, np.sqrt(.5)]
            for joint in range(model.njnt):
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                    data.qpos[model.jnt_qposadr[joint]] = .03 * (joint + 1)
            mujoco.mj_forward(model, data)
            sources[robot] = model, data
            sessions[robot] = SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(
                mj_model=model, data=SimpleNamespace(qpos=data.qpos[None]))))
        visual = ComparisonRenderer(sessions)
        self.addCleanup(visual.close)
        visual.reset({r: sessions[r].env.sim.data.qpos[0].copy() for r in sessions})
        visual.sync({r: sessions[r].env.sim.data.qpos[0].copy() for r in sessions})
        mesh_offset = 0
        for index, (robot, (model, data)) in enumerate(sources.items()):
            prefix = robot + '/'
            for joint in range(model.njnt):
                name = model.joint(joint).name
                dest = visual.model.joint(prefix + name).qposadr[0]
                src = model.jnt_qposadr[joint]
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                    self.assertAlmostEqual(visual.data.qpos[dest], data.qpos[src])
                elif model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                    np.testing.assert_allclose(visual.data.qpos[dest:dest + 3],
                                               [0, (-2, 0, 2)[index], (.45, .7, .8)[index]], atol=1e-7)
                    np.testing.assert_allclose(visual.data.qpos[dest + 3:dest + 7], [1, 0, 0, 0], atol=1e-7)
            root = get_robot(robot).root_body
            root_src, root_dst = data.body(root).xpos, visual.data.body(prefix + root).xpos
            for body in get_robot(robot).tracked_bodies:
                delta = data.body(body).xpos - root_src
                expected = root_dst + [delta[1], -delta[0], delta[2]]
                np.testing.assert_allclose(visual.data.body(prefix + body).xpos, expected, atol=2e-6)
            for mesh in range(model.nmesh):
                dst_mesh = visual.model.mesh(prefix + model.mesh(mesh).name).id
                count = model.mesh_vertnum[mesh]
                source_vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + count]
                dest_vertices = visual.model.mesh_vert[visual.model.mesh_vertadr[dst_mesh]:
                                                      visual.model.mesh_vertadr[dst_mesh] + count]
                np.testing.assert_array_equal(dest_vertices, source_vertices)
            self.assertEqual(visual.model.mesh(prefix + model.mesh(0).name).id, mesh_offset)
            mesh_offset += model.nmesh
        # A native change after construction must reach the shared scene, not a reference pose.
        model, data = sources['q1']
        native = model.joint('left_knee_joint').qposadr[0]
        data.qpos[native] = .75
        visual.sync({r: sessions[r].env.sim.data.qpos[0].copy() for r in sessions})
        self.assertAlmostEqual(visual.data.joint('q1/left_knee_joint').qpos[0], .75)
        # Initial +X motion with heading +90deg becomes display -Y motion.
        data.qpos[0] += .5
        visual.sync({r: sessions[r].env.sim.data.qpos[0].copy() for r in sessions})
        self.assertAlmostEqual(visual.data.qpos[1], -2.5)

    def test_three_real_assets_produce_a_single_rgb_frame(self):
        import mujoco
        import numpy as np
        from morphmimic.robots import get_robot
        from preview.comparison import ComparisonRenderer
        sessions = {}
        for robot in ('q1', 'bumi', 'g1'):
            model = get_robot(robot).entity_cfg_fn().spec_fn().compile()
            sessions[robot] = SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(
                mj_model=model, data=SimpleNamespace(qpos=model.qpos0[None].copy()))))
        visual = ComparisonRenderer(sessions)
        self.addCleanup(visual.close)
        poses = {r: sessions[r].env.sim.data.qpos[0].copy() for r in sessions}
        rgb = visual.frame_rgb(poses, poses)
        self.assertEqual(rgb.shape, (720, 1280, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        visible_mesh_robots = set()
        for geom in visual.renderer.scene.geoms[:visual.renderer.scene.ngeom]:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                name = visual.model.body(visual.model.geom_bodyid[geom.objid]).name
                visible_mesh_robots.add(name.split('/')[0])
        self.assertEqual(visible_mesh_robots, {'q1', 'bumi', 'g1'})
        for column in range(3):
            self.assertGreater(float(rgb[200:650, column * 426:(column + 1) * 426].std()), 8)


if __name__ == '__main__':
    unittest.main()
