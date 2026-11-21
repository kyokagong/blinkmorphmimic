"""Data boundary checks; importing the converter must not initialize a simulator."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

_PATH = Path(__file__).parents[1] / 'src/morphmimic/data/local_prepare.py'


def converter():
    spec = importlib.util.spec_from_file_location('local_prepare', _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalDataTests(unittest.TestCase):
    def setUp(self):
        self.convert = converter()

    def test_native_quaternion_order_and_fifty_hz_interpolation(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'clip.csv'
            rows = np.zeros((31, 9))
            rows[:, 0] = np.arange(31) / 30
            rows[:, 5:7] = np.sqrt(.5)
            rows[:, 7] = rows[:, 0] * 2
            np.savetxt(p, rows, delimiter=',')
            pos, quat, joints = self.convert.read_native_csv(p, ('a', 'b'), 30)
            pos, quat, joints = self.convert.resample(pos, quat, joints, 30)
            self.assertEqual(pos.shape, (51, 3))
            np.testing.assert_allclose(pos[1], [.02, 0, 0])
            np.testing.assert_allclose(quat[1], [np.sqrt(.5), 0, 0, np.sqrt(.5)])
            np.testing.assert_allclose(joints[-1], [2, 0])

    def test_invalid_input_is_rejected_before_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'clip.csv'
            good = np.zeros((2, 9)); good[:, 6] = 1
            for label, rows, names, fps in [
                ('wrong width', good[:, :-1], ('a', 'b'), 30),
                ('one frame', good[:1], ('a', 'b'), 30),
                ('zero quaternion', good * 0, ('a', 'b'), 30),
                ('nonfinite', good * np.nan, ('a', 'b'), 30),
                ('duplicate names', good, ('a', 'a'), 30),
                ('unknown fps', good, ('a', 'b'), 0),
                ('fractional fps', good, ('a', 'b'), 29.97),
            ]:
                with self.subTest(label=label):
                    np.savetxt(p, rows, delimiter=',')
                    with self.assertRaises(ValueError):
                        self.convert.read_native_csv(p, names, fps)

    def test_sdk_interleaved_joints_are_reordered_to_native_names(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'bumi.csv'
            rows = np.array([[0, 0, 1, 0, 0, 0, 1, 11, 22, 33], [0, 0, 1, 0, 0, 0, 1, 44, 55, 66]])
            np.savetxt(p, rows, delimiter=',')
            _, _, joints = self.convert.read_native_csv(p, ('waist', 'left', 'right'), 30, ('left', 'right', 'waist'))
            np.testing.assert_equal(joints, [[33, 11, 22], [66, 44, 55]])

    def test_shortest_arc_slerp_handles_quaternion_sign_changes(self):
        q = np.array([[1., 0, 0, 0], [-1., 0, 0, 0]])
        pos, quat, joints = self.convert.resample(np.zeros((2, 3)), q, np.zeros((2, 1)), 30)
        self.assertTrue(np.isfinite(quat).all())
        np.testing.assert_allclose(quat, np.tile([1., 0, 0, 0], (2, 1)))

    def test_source_group_keeps_mirrors_retargets_and_subjects_together(self):
        group = self.convert.source_group
        self.assertEqual(group('seed', 'dance_seed_move_001__A306'), group('seed', 'dance_seed_move_001__A306_M'))
        self.assertEqual(group('seed', 'dance_seed_move_001__A306'), group('seed', 'dance_seed_move_001__A307'))
        self.assertEqual(group('lafan1', 'dance1_subject1'), group('lafan1', 'dance1_subject5'))
        for name in ['dance_seed_move_001__A306', 'dance_seed_move_001__A306_M']:
            self.assertEqual(self.convert.partition(group('seed', name)), self.convert.partition(group('seed', 'dance_seed_move_001__A307')))

    @unittest.skipUnless(Path('/data/kk/data/blinkmimic_q1/20261107/manifest.json').exists(), 'local dataset not mounted')
    def test_inventory_has_all_lafan_sources_with_explicit_missing_bumi_retargets(self):
        entries, missing, _ = self.convert.discover(Path('/data/kk/data'))
        for robot in ('q1', 'bumi', 'g1'):
            self.assertEqual(sum(e['robot'] == robot and e['dataset'] == 'lafan1' for e in entries), 40)
            self.assertEqual(sum(e['robot'] == robot and e['dataset'] == 'seed' for e in entries), 11006)
        self.assertEqual(len(missing), 32)
        bumi = [e for e in entries if e['robot'] == 'bumi' and e.get('retarget')]
        self.assertEqual(len(bumi), 32)
        self.assertTrue(all(e['fps'] == 30 for e in bumi))
        self.assertTrue(all(e['dataset'] in ('seed', 'lafan1') for e in entries))

    @unittest.skipUnless(importlib.util.find_spec('mujoco') and importlib.util.find_spec('scipy'), 'MuJoCo/SciPy runtime required')
    def test_export_has_native_joints_fourteen_bodies_and_world_velocities(self):
        import mujoco
        from types import SimpleNamespace
        children = ''.join(f'<body name="body{i}"><geom size=".01" mass=".1"/>'
                           + (f'<joint name="joint{i}" axis="0 0 1"/>' if i < 2 else '')
                           + '</body>' for i in range(13))
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="root"><freejoint/>'
                                              '<geom size=".01" mass="1"/>' + children + '</body></worldbody></mujoco>')
        names = ('root',) + tuple(f'body{i}' for i in range(13))
        robot = SimpleNamespace(joint_names=('joint0', 'joint1'), tracked_bodies=names, root_body='root')
        self.convert._robot_model = lambda _: (robot, model, [model.body(n).id for n in names])
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / 'source.csv', Path(root) / 'out.npz'
            rows = np.zeros((31, 9)); rows[:, 6] = 1
            rows[:, 0] = np.arange(31) / 30
            rows[:, 7] = rows[:, 0] * 2
            np.savetxt(source, rows, delimiter=',')
            stats = self.convert.convert_motion(source, output, 'fixture', 30)
            self.assertEqual(stats['frames'], 51)
            self.assertEqual(stats['source_duration_seconds'], 1)
            with np.load(output) as data:
                self.assertEqual(int(data['fps']), 50)
                self.assertEqual(data['body_pos_w'].shape, (51, 14, 3))
                self.assertEqual(data['joint_pos'].shape, (51, 2))
                np.testing.assert_allclose(data['joint_vel'][:, 0], 2, atol=1e-6)
                np.testing.assert_allclose(data['body_lin_vel_w'][:, 0], np.tile([1, 0, 0], (51, 1)), atol=1e-6)
                np.testing.assert_allclose(data['body_ang_vel_w'][:, 1], np.tile([0, 0, 2], (51, 1)), atol=1e-6)
                np.testing.assert_equal(data['joint_names'], ['joint0', 'joint1'])
                self.assertEqual(len(data['body_names']), 14)
                self.assertTrue(all(data[k].dtype == np.float32 for k in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w')))
            entry = {'input': str(source), 'robot': 'fixture', 'name': 'source', 'dataset': 'seed',
                     'group_id': 'seed/source', 'fps': 30}
            before = output.stat().st_mtime_ns
            adopted = self.convert._convert_job((entry, output, 'fixture-asset'))
            self.assertEqual(output.stat().st_mtime_ns, before)
            self.assertEqual(adopted['asset_sha256'], 'fixture-asset')
            self.assertEqual(adopted['frames'], 51)
            self.assertTrue(output.with_suffix('.receipt.json').exists())
            resumed = self.convert._convert_job((entry, output, 'fixture-asset'))
            self.assertTrue(resumed['resumed'])
            self.assertEqual(output.stat().st_mtime_ns, before)


    def test_resume_receipt_rejects_changed_source_asset_or_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'clip.npz'
            output.write_bytes(b'completed motion')
            receipt = {'input_sha256': 'source1', 'asset_sha256': 'asset1',
                       'output_sha256': self.convert.sha256(output), 'frames': 51}
            output.with_suffix('.receipt.json').write_text(__import__('json').dumps(receipt))
            self.assertEqual(self.convert._load_receipt(output, 'source1', 'asset1')['frames'], 51)
            for source, asset in [('source2', 'asset1'), ('source1', 'asset2')]:
                with self.assertRaises(ValueError):
                    self.convert._load_receipt(output, source, asset)
            output.write_bytes(b'truncated')
            with self.assertRaises(ValueError):
                self.convert._load_receipt(output, 'source1', 'asset1')

    @unittest.skipUnless(importlib.util.find_spec('mujoco') and Path('/data/kk/supermimic_vibed/tools/retarget_bumi.py').exists(), 'real retarget runtime required')
    def test_real_bumi_retarget_preflight_does_not_import_isaaclab(self):
        import sys
        fn = self.convert._retarget_function('/data/kk/supermimic_vibed')
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / 'source.csv', Path(root) / 'bumi.csv'
            rows = np.loadtxt('/data/kk/data/supermimic_g1/lafan1_g1/fallAndGetUp1_subject1.csv', delimiter=',', max_rows=2)
            np.savetxt(source, rows, delimiter=',')
            report = fn(source, output)
            result = np.loadtxt(output, delimiter=',', ndmin=2)
            self.assertEqual(result.shape, (2, 28))
            self.assertTrue(np.isfinite(result).all())
            self.assertEqual(report['source_frames'], 2)
            self.assertNotIn('isaaclab', sys.modules)
            self.assertEqual(fn.adaptation['import_replacements'], 1)

    @unittest.skipUnless(importlib.util.find_spec('yaml') and importlib.util.find_spec('pandas'), 'YAML/Pandas runtime required')
    def test_relative_subset_root_resolves_from_yaml_in_another_working_directory(self):
        from contextlib import chdir
        manifest_path = _PATH.with_name('manifest.py')
        module_spec = importlib.util.spec_from_file_location('local_manifest', manifest_path)
        manifest = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(manifest)
        with tempfile.TemporaryDirectory() as root:
            dataset = Path(root) / 'prepared/q1'
            dataset.mkdir(parents=True)
            elsewhere = Path(root) / 'other-cwd'
            elsewhere.mkdir()
            clip = dataset / 'clip.npz'
            np.savez(clip, value=np.array([42]))
            yaml = dataset / 'all.yaml'
            yaml.write_text('root: .\nclips: [clip.npz]\n')
            with chdir(elsewhere):
                loaded = manifest.load_subset(str(yaml))
                self.assertEqual(loaded, [str(clip)])
                with np.load(loaded[0]) as motion:
                    self.assertEqual(int(motion['value'][0]), 42)
                yaml.write_text(f'root: {dataset}\nclips: [clip.npz]\n')
                self.assertEqual(manifest.load_subset(str(yaml)), [str(clip)])

    def test_slerp_rotation_midpoint_and_120_hz_duration(self):
        angle = np.linspace(0, np.pi, 121) / 2
        q = np.column_stack([np.cos(angle), np.zeros((121, 2)), np.sin(angle)])
        _, quat, _ = self.convert.resample(np.zeros((121, 3)), q, np.zeros((121, 1)), 120)
        self.assertEqual(len(quat), 51)
        np.testing.assert_allclose(quat[25], [np.sqrt(.5), 0, 0, np.sqrt(.5)], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
