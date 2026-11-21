"""Flip data preserves ordinary skills and pairs real source-time human sidecars."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'prepared/importance-flips-20261116'


def api():
    spec = importlib.util.spec_from_file_location('flip_mix', ROOT/'scripts/prepare_flip_mix.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class FlatSoles:
    def sole_heights(self, positions, quaternions):
        return positions[:, :, 2]


class FlipMixTests(unittest.TestCase):
    def test_requires_airborne_sagittal_rotation_not_a_flip_label_or_cartwheel(self):
        m = api(); n = 151
        positions = np.zeros((n, 14, 3)); positions[40:100, [3, 6], 2] = .3
        for axis, expected in [('y', True), ('x', False), ('z', False)]:
            root = Rotation.from_euler(axis, np.linspace(0, 2*np.pi, n)[:, None]).as_quat()[:, [3, 0, 1, 2]]
            motion = dict(body_pos_w=positions, body_quat_w=np.repeat(root[:, None], 14, axis=1))
            metrics = m.flip_metrics(motion, FlatSoles())
            self.assertEqual(metrics['full_sagittal_rotation'], expected)
            self.assertAlmostEqual(metrics['flight_seconds'], 1.2)
        motion['body_pos_w'] = np.zeros_like(positions)
        self.assertFalse(m.flip_metrics(motion, FlatSoles())['full_sagittal_rotation'])
        self.assertEqual(m.longest_interval([False, True, True, False, True]), (1, 3))

    @unittest.skipUnless((OUTPUT/'q1/seed_flip_180_R_001__A304.npz').exists(), 'Historical append-only payloads retired; unified flip provenance is tested below')
    def test_generated_prefix_split_source_time_fk_floor_and_transition(self):
        m = api(); parent = ROOT/'prepared/neutral-recovery-20261116'
        old = json.loads((parent/'manifest.json').read_text()); new = json.loads((OUTPUT/'manifest.json').read_text())
        self.assertEqual(new['clips'][:len(old['clips'])], old['clips'])
        self.assertEqual(sum(r['dataset'] == 'neutral_recovery' for r in new['clips']), 121)
        added = new['clips'][len(old['clips']):]; self.assertEqual(len(added), 18)
        self.assertEqual({r['dataset'] for r in added}, {'seed_flips'})
        groups = {}
        for row in added:
            groups.setdefault(row['group_id'], set()).add(row['split'])
            self.assertTrue(row['flip_metrics']['full_sagittal_rotation'])
            self.assertFalse(row['validation']['dynamics_qualified'])
            self.assertEqual(row['human_sha256'], m.sha256(Path(row['human_sidecar'])))
            self.assertEqual(row['output_sha256'], m.sha256(Path(row['output'])))
            motion = m.load_npz(row['output']); side = m.load_npz(row['human_sidecar'])
            m.validate_native(motion, row['robot'])
            self.assertEqual(len(side['human_pos_w']), row['frames'])
            np.testing.assert_array_equal(side['source_time_s'], np.arange(row['frames'])/50)
            # Compare against independent raw SOMA conversion, never robot FK.
            original = m.load_npz(Path(row['derivation']['source_human']['path']).parent.parent/(row['derivation']['source_metadata']['filename']+'.human50.npz'))
            np.testing.assert_array_equal(side['human_pos_w'], original['human_pos_w'][:row['frames']])
            np.testing.assert_array_equal(side['human_quat_w'], original['human_quat_w'][:row['frames']])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        floor = json.loads((OUTPUT/'contact-floor.json').read_text()); old_floor = json.loads((parent/'contact-floor.json').read_text())
        self.assertEqual(floor['clips'][:len(old_floor['clips'])], old_floor['clips'])
        self.assertEqual(floor['prepared_manifest_sha256'], m.sha256(OUTPUT/'manifest.json'))
        for row in floor['clips'][len(old_floor['clips']):]:
            self.assertLessEqual(abs(row['offset_m']), .12)
            if not row['valid']: self.assertEqual(row['offset_m'], 0.)
        old_windows = json.loads((parent/'windows.json').read_text()); windows = json.loads((OUTPUT/'windows.json').read_text())
        for key in old_windows.keys()-{'training_manifest_sha256', 'subset_sha256'}:
            self.assertEqual(windows[key], old_windows[key])
        from morphmimic.train.dataset_transition import verify_append_transition
        for robot in m.ROBOTS:
            transition_path=OUTPUT/'transition.json'
            pinned_manifest=Path(json.loads(transition_path.read_text())['manifest']['path'])
            transition = verify_append_transition(transition_path, m.sha256(transition_path), pinned_manifest, robot)
            info = transition['robots'][robot]
            self.assertEqual(info['appended_clip_count'], 4)
            self.assertEqual(windows['subset_sha256'][robot], m.sha256(OUTPUT/robot/'train.yaml'))
        catalog = json.loads((OUTPUT/'flip-preview.json').read_text())
        self.assertEqual(catalog['training_manifest_sha256'], m.sha256(OUTPUT/'manifest.json'))
        self.assertEqual(len(catalog['clips']), 18)


    @unittest.skipUnless((ROOT/'prepared/inplace-recovery-neutral-20261117-v2/manifest.json').exists(), 'Unified corpus not generated')
    def test_unified_flips_keep_raw_human_motion_and_group_splits(self):
        m=api();root=ROOT/'prepared/inplace-recovery-neutral-20261117-v2'
        document=json.loads((root/'manifest.json').read_text())
        rows=[r for r in document['clips'] if r['dataset']=='seed_flips']
        self.assertEqual(len(rows),18);groups={}
        from morphmimic.data.neutral_bookends import source_frame_map,_sample,_sample_quat
        for row in rows:
            groups.setdefault(row['group_id'],set()).add(row['split'])
            motion=m.load_npz(row['output']);side=m.load_npz(row['human_sidecar'])
            m.validate_native(motion,row['robot'])
            self.assertEqual(m.sha256(Path(row['output'])),row['output_sha256'])
            self.assertEqual(m.sha256(Path(row['human_sidecar'])),row['human_sha256'])
            c=row['neutral_contract'];a,b=c['source_start_frame'],c['source_stop_frame']
            mapping=source_frame_map(c['original_source_frames'],c['source_ramp_frames'])
            source=row['derivation']['source_human']
            self.assertEqual(m.sha256(Path(source['path'])),source['sha256'])
            original=m.load_npz(Path(source['path']).parent.parent/(row['derivation']['source_metadata']['filename']+'.human50.npz'))
            n=c['original_source_frames'];p=original['human_pos_w'][:n];q=original['human_quat_w'][:n]
            turn=Rotation.from_rotvec([0,0,float(side['root_yaw_rotation_rad'])])
            p=turn.apply(p.reshape(-1,3)).reshape(p.shape)+side['root_translation_xyz']
            np.testing.assert_allclose(side['human_pos_w'][a:b],_sample(p,mapping),atol=2e-6,rtol=0)
            raw=Rotation.from_quat(q.reshape(-1,4)[:,[1,2,3,0]])
            rotated=(turn*raw).as_quat()[:,[3,0,1,2]].reshape(q.shape)
            expected=_sample_quat(rotated,mapping)
            np.testing.assert_allclose(abs((side['human_quat_w'][a:b]*expected).sum(-1)),1,atol=1e-6,rtol=0)
            np.testing.assert_allclose(side['source_time_s'][a:b],mapping/50,atol=1e-8,rtol=0)
            self.assertTrue(np.isnan(side['source_time_s'][:a]).all())
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))


if __name__ == '__main__':
    unittest.main()
