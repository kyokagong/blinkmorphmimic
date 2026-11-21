"""Recovery labels must follow FK geometry and a real, bounded future stand."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def api():
    path = Path(__file__).resolve().parents[1] / 'scripts/prepare_recovery.py'
    assert path.is_file(), 'Recovery preparation implementation is missing'
    spec = importlib.util.spec_from_file_location('prepare_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def motion(robot='g1', count=400):
    pos = np.zeros((count, 14, 3), dtype=np.float64)
    pos[:, 0, 2] = 1
    pos[:, [2, 5], 2] = .5
    pos[:, [8, 11], 2] = 1.4
    quat = np.zeros((count, 14, 4)); quat[..., 0] = 1
    joints = np.full((count, 2), .1)
    pos[50:100, 0, 2] = .15
    quat[50:100, 0] = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]
    pos[100:150, 0, 2] = .4
    pos[100:150, [2, 5], 2] = .35
    joints[100:150] = 1.5
    pos[150:200, 0, 2] = .7
    pos[150:200, [2, 5], 2] = .3
    joints[150:200] = 1
    names = ['pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
             'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link',
             'torso_link', 'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
             'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link']
    return dict(fps=np.array(50), robot=np.array(robot), root_body=np.array('pelvis'),
                body_names=np.array(names), joint_names=np.array(['left_knee_joint', 'right_knee_joint']),
                body_pos_w=pos, body_quat_w=quat, joint_pos=joints,
                body_lin_vel_w=np.zeros_like(pos), body_ang_vel_w=np.zeros_like(pos))


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'prepared/inplace-recovery-neutral-20261117-v2'


def real_source(robot, name):
    current = (CURRENT / 'manifest.json').is_file()
    path = (CURRENT if current else ROOT / 'prepared/human-v1') / robot / (name+'.npz')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    contract = json.loads(path.with_suffix('.receipt.json').read_text())['neutral_contract'] if current else None
    return data, contract


def source_index(contract, index):
    if contract is None: return index
    from morphmimic.data.neutral_bookends import source_frame_map
    mapping = source_frame_map(contract['original_source_frames'], contract['source_ramp_frames'])
    return contract['source_start_frame'] + int(np.searchsorted(mapping, index, side='left'))


class RecoveryDataTests(unittest.TestCase):
    def test_geometry_separates_lying_seated_half_squat_and_rejects_high_crouch(self):
        m = api(); data = motion()
        labels, standing, _ = m.classify_frames(data, {'standing_leg_height_m': 1., 'ankle_floor_m': 0.})
        self.assertEqual(labels[[25, 75, 125, 175, 225]].tolist(), ['', 'lying', 'seated', 'half_squat', ''])
        self.assertEqual(standing[[25, 75, 125, 175, 225]].tolist(), [True, False, False, False, True])
        data['body_pos_w'][175, 0, 2] = .86
        data['body_pos_w'][125, 5, 2] = .1
        data['body_pos_w'][75, 0, 2] = .8
        labels, _, _ = m.classify_frames(data, {'standing_leg_height_m': 1., 'ankle_floor_m': 0.})
        self.assertEqual(labels[[75, 125, 175]].tolist(), ['', '', ''])

    def test_half_squat_rejects_airborne_feet_and_knees_below_ankles(self):
        m = api(); data = motion()
        calibration = {'standing_leg_height_m': 1., 'ankle_floor_m': 0.}
        data['body_pos_w'][175, [3, 6], 2] = .25
        data['body_pos_w'][175, [2, 5], 2] = .4
        self.assertEqual(m.classify_frames(data, calibration)[0][175], '')
        data['body_pos_w'][175, [3, 6], 2] = 0.
        data['body_pos_w'][175, [2, 5], 2] = -.1
        self.assertEqual(m.classify_frames(data, calibration)[0][175], '')

    def test_real_airborne_g1_frame801_is_rejected_and_grounded_squats_remain(self):
        m = api()
        calibration = {'q1': {'standing_leg_height_m': .3782202005, 'ankle_floor_m': .0463561714},
                       'bumi': {'standing_leg_height_m': .4277965426, 'ankle_floor_m': .0562233254},
                       'g1': {'standing_leg_height_m': .7465422750, 'ankle_floor_m': .0408330671}}
        bad, contract = real_source('g1', 'fallAndGetUp2_subject3')
        bad_calibration = dict(calibration['g1'])
        if contract: bad_calibration['ankle_floor_m'] += contract['root_translation_xyz'][2]
        self.assertEqual(m.classify_frames(bad, bad_calibration)[0][source_index(contract, 801)], '')
        for robot in ('q1', 'bumi', 'g1'):
            with self.subTest(robot=robot):
                good, contract = real_source(robot, 'fallAndGetUp1_subject4')
                good_calibration = dict(calibration[robot])
                if contract: good_calibration['ankle_floor_m'] += contract['root_translation_xyz'][2]
                self.assertEqual(m.classify_frames(good, good_calibration)[0][source_index(contract, 7930)], 'half_squat')

    def test_direct_windows_reject_real_descending7930_and_retain_ascending8298(self):
        m = api()
        manifest = (CURRENT if (CURRENT/'manifest.json').is_file() else ROOT/'prepared/allopt-20261112')/'manifest.json'
        result = m.build_dataset(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())
        for robot in ('q1', 'bumi', 'g1'):
            with self.subTest(robot=robot):
                _, descending = real_source(robot, 'fallAndGetUp1_subject4')
                _, ascending = real_source(robot, 'fallAndGetUp1_subject5')
                down, up = source_index(descending, 7930), source_index(ascending, 8298)
                self.assertFalse(any(w['robot']==robot and w['clip']=='fallAndGetUp1_subject4' and
                                     w['start_frame']<=down<w['stop_frame'] and w['category']=='half_squat'
                                     for w in result['windows']))
                self.assertTrue(any(w['robot']==robot and w['clip']=='fallAndGetUp1_subject5' and
                                    w['start_frame']<=up<w['stop_frame'] and w['category']=='half_squat'
                                    for w in result['windows']))
        height = result['calibration']['q1']['standing_leg_height_m']
        bad, contract = real_source('q1', 'fallAndGetUp1_subject4')
        a, b = source_index(contract, 7930), source_index(contract, 8202)
        self.assertGreater(float(bad['body_pos_w'][a,0,2] - bad['body_pos_w'][a:b,0,2].min()), .15*height)
        good, contract = real_source('q1', 'fallAndGetUp1_subject5')
        a, b = source_index(contract, 8298), source_index(contract, 8345)
        self.assertLessEqual(float(good['body_pos_w'][a,0,2] - good['body_pos_w'][a:b,0,2].min()), .15*height)

    def test_calibration_preserves_each_robots_physical_scale(self):
        m = api(); a = motion(); b = motion('q1')
        b['body_pos_w'] *= .4
        self.assertAlmostEqual(m.calibrate([a])['standing_leg_height_m'], 1.)
        self.assertAlmostEqual(m.calibrate([b])['standing_leg_height_m'], .4)
        self.assertEqual(m.classify_frames(b, m.calibrate([b]))[0][175], 'half_squat')

    def test_future_stand_requires_full_hold_and_clips_to_ten_seconds(self):
        m = api()
        labels = np.full(700, '', dtype='<U10'); labels[0:200] = 'lying'; labels[300:350] = 'seated'
        standing = np.zeros(700, dtype=bool); standing[600:650] = True
        windows = m.recovery_ranges(labels, standing, np.ones(len(labels)))
        self.assertEqual(windows, [dict(category='lying', start_frame=150, stop_frame=200, stand_frame=600, hold_end_frame=650),
                                   dict(category='seated', start_frame=300, stop_frame=350, stand_frame=600, hold_end_frame=650)])
        self.assertTrue(all(w['hold_end_frame'] - w['start_frame'] <= 500 for w in windows))
        labels[100:200] = ''
        self.assertEqual(m.recovery_ranges(labels, standing, np.ones(len(labels))), [windows[1]])
        standing[625] = False
        self.assertEqual(m.recovery_ranges(labels, standing, np.ones(len(labels))), [])

    def test_direct_recovery_allows_small_dips_and_trims_each_reset_frame(self):
        m = api(); labels = np.full(120, '', dtype='<U10'); labels[30:50] = 'half_squat'
        standing = np.zeros(120, dtype=bool); standing[70:] = True
        heights = np.ones(120); heights[30:50] = np.linspace(.8,.6,20); heights[50:70] = .59
        ranges = m.recovery_ranges(labels,standing,heights)
        self.assertEqual(ranges, [dict(category='half_squat',start_frame=36,stop_frame=50,stand_frame=70,hold_end_frame=120)])
        heights[50] = .3
        self.assertEqual(m.recovery_ranges(labels,standing,heights), [])
        heights[50] = .59; labels[50] = 'lying'
        self.assertEqual(m.recovery_ranges(labels,standing,heights), [])
        with self.assertRaises(ValueError):m.recovery_ranges(labels,standing,heights[:-1])
        heights[0] = np.nan
        with self.assertRaises(ValueError):m.recovery_ranges(labels,standing,heights)

    def test_boundary_hold_at_clip_end_and_short_reset_ranges(self):
        m = api(); labels = np.full(80, '', dtype='<U10'); labels[20:30] = 'half_squat'
        stand = np.zeros(80, dtype=bool); stand[30:] = True
        self.assertEqual(m.recovery_ranges(labels, stand, np.ones(len(labels))), [dict(category='half_squat', start_frame=20, stop_frame=30, stand_frame=30, hold_end_frame=80)])
        self.assertEqual(m.recovery_ranges(labels[:-1], stand[:-1], np.ones(len(labels)-1)), [])
        stand[:] = False; stand[:20] = True
        self.assertEqual(m.recovery_ranges(labels, stand, np.ones(len(labels))), [])

    def test_invalid_fps_quaternion_and_nonfinite_values_are_rejected(self):
        m = api()
        for key, mutate in [('fps', lambda d: d.update(fps=np.array(30))),
                            ('quaternion', lambda d: d['body_quat_w'].__setitem__((0, 0), [0, 0, 0, 0])),
                            ('nonfinite', lambda d: d['body_pos_w'].__setitem__((0, 0, 0), np.nan))]:
            with self.subTest(key=key):
                data = motion(); mutate(data)
                with self.assertRaises(ValueError):
                    m.classify_frames(data, {'standing_leg_height_m': 1., 'ankle_floor_m': 0.})

    def test_build_pins_manifest_subsets_and_selected_source_bytes(self):
        m = api()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = []
            for robot in ('q1', 'bumi', 'g1'):
                (root / robot).mkdir(); path = root / robot / 'fallAndGetUp1_subject1.npz'
                np.savez(path, **motion(robot))
                rows.append(dict(robot=robot, name=path.stem, dataset='lafan1', split='train',
                                 source_id='lafan1/fallAndGetUp1_subject1', output=str(path), output_sha256=m.sha256(path)))
                (root / robot / 'train.yaml').write_text(json.dumps({'root': '/', 'clips': [str(path)]}))
            manifest = root / 'manifest.json'; manifest.write_text(json.dumps({'complete': True, 'clips': rows}))
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            result = m.build_dataset(manifest, digest)
            self.assertEqual(len(result['windows']), 9)
            self.assertEqual(result['training_manifest_sha256'], digest)
            self.assertEqual(result['subset_sha256']['q1'], hashlib.sha256((root/'q1/train.yaml').read_bytes()).hexdigest())
            self.assertEqual(result['windows'][0]['source_id'], 'lafan1/fallAndGetUp1_subject1')
            with self.assertRaisesRegex(ValueError, 'manifest'):
                m.build_dataset(manifest, '0'*64)
            with (root/'q1/fallAndGetUp1_subject1.npz').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                m.build_dataset(manifest, digest)


if __name__ == '__main__':
    unittest.main()
