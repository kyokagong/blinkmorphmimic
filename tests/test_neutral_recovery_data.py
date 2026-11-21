"""Neutral augmentation keeps real recoveries and freezes the approved whole-body pose."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def api():
    path = ROOT / 'scripts/prepare_neutral_recovery.py'
    assert path.exists(), 'Neutral recovery implementation is missing'
    spec = importlib.util.spec_from_file_location('neutral_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NeutralRecoveryTests(unittest.TestCase):
    def test_real_native_prefix_fk_approved_hold_and_paired_human(self):
        m = api()
        review = json.loads((ROOT / 'runs/reports/neutral-standing-review-20261116/review-manifest.json').read_text())
        current = ROOT / 'prepared/inplace-recovery-neutral-20261117-v2'
        migrated = (current / 'manifest.json').exists()
        manifest = json.loads(((current if migrated else ROOT / 'prepared/allopt-20261112') / 'manifest.json').read_text())
        if migrated:
            review = json.loads((current / 'standing-review.json').read_text())
        windows = json.loads((ROOT / 'prepared/recovery-direct-20261114/windows.json').read_text())
        for robot in ('q1', 'bumi', 'g1'):
            w = next(w for w in windows['windows'] if w['robot'] == robot and w['category'] == 'seated')
            row = next(r for r in manifest['clips'] if r['robot'] == robot and r['name'] == w['clip'])
            target_row = next(r for r in manifest['clips'] if r['robot'] == robot and r['name'] == 'walk2_subject4')
            paths = (row['output'], row['human_sidecar'], target_row['output'], target_row['human_sidecar'])
            if migrated:
                from morphmimic.data.neutral_bookends import source_frame_map
                c = row['neutral_contract']
                mapping = source_frame_map(c['original_source_frames'], c['source_ramp_frames'])
                w = dict(w)
                for key in ('start_frame','stop_frame','stand_frame'):
                    w[key] = c['source_start_frame'] + int(np.searchsorted(mapping, w[key]))
                paths = (row['output'], row['human_sidecar'], review['robot_sources'][robot]['npz'], review['human_sources'][robot]['npz'])
            source, human, target, reference = [m.load_npz(p) for p in paths]
            out, side, new_window, provenance = m.derive_clip(source, human, target, reference, w, robot, review['prepared_frame_zero_based'])
            prefix = w['stand_frame'] - w['start_frame'] + 1
            np.testing.assert_array_equal(out['joint_pos'][:prefix], source['joint_pos'][w['start_frame']:w['stand_frame']+1])
            np.testing.assert_allclose(out['body_pos_w'][:prefix], source['body_pos_w'][w['start_frame']:w['stand_frame']+1], atol=3e-6)
            np.testing.assert_array_equal(side['human_pos_w'][:prefix], human['human_pos_w'][w['start_frame']:w['stand_frame']+1])
            stand, end = new_window['stand_frame'], new_window['hold_end_frame']
            self.assertEqual(end - stand, 150)
            self.assertLessEqual(end, 750)
            self.assertEqual(len(out['joint_pos']), end)
            self.assertGreater(stand, prefix)
            for key in ('joint_pos', 'body_pos_w', 'body_quat_w'):
                np.testing.assert_array_equal(out[key][stand:end], np.broadcast_to(out[key][stand], out[key][stand:end].shape))
            for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
                np.testing.assert_allclose(out[key][stand:end], 0., atol=1e-9)
            np.testing.assert_array_equal(out['joint_pos'][stand], target['joint_pos'][11446])
            self.assertAlmostEqual(float(out['body_pos_w'][stand, 0, 2]), float(target['body_pos_w'][11446, 0, 2]), places=6)
            # Rotation/translation cannot change human bone lengths; robot FK cannot satisfy this reference.
            delta = side['human_pos_w'][stand] - side['human_pos_w'][stand, 0]
            approved_delta = reference['human_pos_w'][11446] - reference['human_pos_w'][11446, 0]
            np.testing.assert_allclose(np.linalg.norm(delta, axis=1), np.linalg.norm(approved_delta, axis=1), atol=2e-6)
            np.testing.assert_array_equal(side['human_quat_w'][stand:end], np.broadcast_to(side['human_quat_w'][stand], side['human_quat_w'][stand:end].shape))
            if robot == 'q1':
                invalid = dict(target, joint_pos=target['joint_pos'].copy())
                invalid['joint_pos'][11446, 0] = np.nan
                with self.assertRaises(ValueError):
                    m.derive_clip(source, human, invalid, reference, w, robot, 11446)
            self.assertTrue(np.all(out['body_pos_w'][stand, [10, 13], 2] < out['body_pos_w'][stand, [8, 11], 2]))
            self.assertEqual(provenance['original_prefix_frames'], prefix)
            self.assertEqual(provenance['source_start_frame'], w['start_frame'])

    def test_reject_invalid_window(self):
        m = api()
        with self.assertRaises(ValueError):
            m.validate_window(dict(start_frame=2, stop_frame=1, stand_frame=3), 10)
        with self.assertRaises(ValueError):
            m.validate_window(dict(start_frame=0, stop_frame=2, stand_frame=12), 10)

    @unittest.skipUnless((ROOT / 'prepared/neutral-recovery-20261116/q1/fallAndGetUp1_subject1__neutral_lying_000.npz').exists(), 'Historical append-only payloads retired; current full-rewrite inventory is covered by test_neutral_prepare')
    def test_generated_inventory_is_append_only_and_floor_rows_keep_source_offsets(self):
        m = api(); original_root = ROOT / 'prepared/allopt-20261112'; new_root = ROOT / 'prepared/neutral-recovery-20261116'
        original = json.loads((original_root / 'manifest.json').read_text()); new = json.loads((new_root / 'manifest.json').read_text())
        self.assertEqual(new['clips'][:len(original['clips'])], original['clips'])
        old_floor = json.loads((original_root / 'contact-floor.json').read_text()); floor = json.loads((new_root / 'contact-floor.json').read_text())
        self.assertEqual(floor['clips'][:len(old_floor['clips'])], old_floor['clips'])
        self.assertEqual(floor['prepared_manifest_sha256'], m.sha256(new_root / 'manifest.json'))
        lookup = {(r['robot'], r['name']): r for r in old_floor['clips']}
        added = new['clips'][len(original['clips']):]
        self.assertEqual(len(added), 121)
        windows = json.loads((new_root / 'windows.json').read_text())
        self.assertEqual(windows['max_recovery_seconds'], 15.)
        by_name = {(w['robot'], w['clip']): w for w in windows['windows']}
        for row, entry in zip(added, floor['clips'][len(old_floor['clips']):]):
            self.assertEqual(entry['offset_m'], lookup[row['robot'], row['derivation']['source_clip']]['offset_m'])
            self.assertEqual(entry['sha256'], m.sha256(row['output']))
            self.assertEqual(row['human_sha256'], m.sha256(row['human_sidecar']))
            out, side = m.load_npz(row['output']), m.load_npz(row['human_sidecar'])
            source = m.load_npz(row['derivation']['source_npz']['path'])
            source_human = m.load_npz(row['derivation']['source_human']['path'])
            first = row['derivation']['source_start_frame']; prefix = row['derivation']['original_prefix_frames']
            np.testing.assert_array_equal(out['joint_pos'][:prefix], source['joint_pos'][first:first+prefix])
            np.testing.assert_allclose(out['body_pos_w'][:prefix], source['body_pos_w'][first:first+prefix], atol=3e-6)
            np.testing.assert_array_equal(side['human_pos_w'][:prefix], source_human['human_pos_w'][first:first+prefix])
            w = by_name[row['robot'], row['name']]
            self.assertEqual(w['hold_end_frame'] - w['stand_frame'], 150)
            self.assertEqual(len(out['joint_pos']), w['hold_end_frame'])
            self.assertLessEqual(w['hold_end_frame'], 750)
            for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
                np.testing.assert_allclose(out[key][w['stand_frame']:], 0., atol=1e-9)
        for robot in ('q1', 'bumi', 'g1'):
            old = json.loads((original_root / robot / 'train.yaml').read_text())['clips']; new = json.loads((new_root / robot / 'train.yaml').read_text())['clips']
            self.assertEqual(new[:len(old)], old)
            self.assertEqual(len(new)-len(old), sum(r['robot'] == robot for r in added))


if __name__ == '__main__':
    unittest.main()
