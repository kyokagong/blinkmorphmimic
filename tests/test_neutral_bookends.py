"""Bookends preserve native action samples and meet a stationary neutral boundary."""
import importlib
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def review_document():
    path = ROOT/'prepared/inplace-recovery-neutral-20261117-v2/standing-review.json'
    if not path.is_file():
        path = ROOT/'runs/reports/neutral-standing-review-20261116/review-manifest.json'
    return json.loads(path.read_text())


def api():
    assert (ROOT/'src/morphmimic/data/neutral_bookends.py').is_file(), 'neutral bookend implementation is missing'
    return importlib.import_module('morphmimic.data.neutral_bookends')


def fixture(robot='q1', speed=10.):
    import mujoco
    from morphmimic.data.local_prepare import _robot_model
    review = review_document()
    with np.load(review['robot_sources'][robot]['npz']) as z:
        target = {k: v.copy() for k, v in z.items()}
    spec, model, ids = _robot_model(robot)
    frame = review['prepared_frame_zero_based']
    for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
        target[key] = target[key][frame:frame+1]
    n = 12; t = np.arange(n)/50
    source = {k: v.copy() for k, v in target.items()}
    joints = np.repeat(target['joint_pos'].astype(float), n, axis=0)
    joint = next(i for i, name in enumerate(spec.joint_names) if 'yaw' in name and ('shoulder' in name or 'arm' in name))
    joints[:, joint] += speed*t
    root = np.tile(target['body_pos_w'][0, 0].astype(float), (n, 1))
    root[:, 0] = 2.+.5*t; root[:, 1] = 3.+.2*t
    quat = Rotation.from_euler('z', (.6+.3*t)[:, None]).as_quat(scalar_first=True)
    data = mujoco.MjData(model); p = []; q = []
    for position, orientation, angles in zip(root, quat, joints):
        data.qpos[:] = np.r_[position, orientation, angles]
        mujoco.mj_kinematics(model, data)
        p.append(data.xpos[ids].copy()); q.append(data.xquat[ids].copy())
    source.update(joint_pos=joints, body_pos_w=np.array(p), body_quat_w=np.array(q),
                  joint_vel=np.zeros_like(joints), body_lin_vel_w=np.zeros_like(p), body_ang_vel_w=np.zeros_like(p))
    return source, target, joint


class NeutralBookendTests(unittest.TestCase):
    def test_all_robots_reach_approved_stationary_pose_and_preserve_native_core(self):
        m = api()
        for robot in ('q1', 'bumi', 'g1'):
            with self.subTest(robot=robot):
                source, target, joint = fixture(robot, speed=1.)
                original = {k: v.copy() for k, v in source.items()}
                out, c = m.neutralize_motion(source, target, robot, target_frame=0, source_floor_offset=.03)
                a, b = c['source_start_frame'], c['source_stop_frame']
                mapping = m.source_frame_map(len(source['joint_pos']), c['source_ramp_frames'])
                integer = mapping == np.floor(mapping)
                np.testing.assert_array_equal(out['joint_pos'][a:b][integer], source['joint_pos'][mapping[integer].astype(int)])
                np.testing.assert_allclose(out['joint_pos'][[0, -1]], np.repeat(target['joint_pos'], 2, axis=0), atol=1e-7)
                np.testing.assert_allclose(out['body_pos_w'][0, 0, :2], 0., atol=1e-7)
                np.testing.assert_allclose(out['body_pos_w'][[0, -1], 0, 2], target['body_pos_w'][0, 0, 2], atol=1e-6)
                np.testing.assert_allclose(out['body_pos_w'][-1, 0, :2], out['body_pos_w'][b-1, 0, :2], atol=1e-7)
                turn = Rotation.from_rotvec([0., 0., c['root_yaw_rotation_rad']])
                want = turn.apply(source['body_pos_w'].reshape(-1, 3)).reshape(source['body_pos_w'].shape) + c['root_translation_xyz']
                np.testing.assert_allclose(out['body_pos_w'][a:b][integer], want[mapping[integer].astype(int)], atol=2e-6)
                for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
                    np.testing.assert_allclose(out[key][:c['entry_hold_end_frame']], 0., atol=1e-8)
                    np.testing.assert_allclose(out[key][c['exit_hold_start_frame']:], 0., atol=1e-8)
                self.assertGreaterEqual(c['entry_frames'], 50); self.assertGreaterEqual(c['exit_frames'], 50)
                self.assertEqual(c['frames'], len(out['joint_pos']))
                self.assertFalse(c['dynamics_qualified'])
                for key in original:
                    np.testing.assert_array_equal(source[key], original[key])

    def test_human_neutral_uses_robot_forward_basis_instead_of_smpl_lateral_x(self):
        m=api();source,target,_=fixture(speed=1.)
        _,c=m.neutralize_motion(source,target,'q1',target_frame=0,in_place=True)
        review=review_document()
        with np.load(review['human_sources']['q1']['npz']) as f:
            reference={k:f[k].copy() for k in f.files}
        human={k:v.copy() for k,v in reference.items() if k!='source_time_s'}
        for k in ('human_pos_w','human_quat_w'):human[k]=np.repeat(reference[k][11446:11447],len(source['joint_pos']),axis=0)
        side=m.neutralize_human(human,reference,c)
        lateral=side['human_pos_w'][0,1]-side['human_pos_w'][0,2]
        self.assertGreater(lateral[1],.1)
        self.assertLess(abs(lateral[0]),1e-5)

    def test_in_place_recovery_ends_at_the_identical_full_neutral_pose(self):
        m=api();source,target,_=fixture(speed=1.)
        out,c=m.neutralize_motion(source,target,'q1',target_frame=0,in_place=True)
        np.testing.assert_allclose(out['body_pos_w'][0],out['body_pos_w'][-1],atol=1e-6,rtol=0)
        np.testing.assert_allclose(out['body_quat_w'][0],out['body_quat_w'][-1],atol=1e-6,rtol=0)
        self.assertTrue(c['recovery_in_place'])
        self.assertAlmostEqual(c['exit_neutral_yaw_rad'],0.)
        self.assertTrue(m.validate_boundary(out,target,'q1',c,target_frame=0)['valid'])

    def test_authorized_slow_clock_reduces_boundary_speed_and_keeps_middle_ten_rad_per_second(self):
        m = api(); source, target, joint = fixture()
        out, c = m.neutralize_motion(source, target, 'q1', target_frame=0)
        a, b = c['source_start_frame'], c['source_stop_frame']
        self.assertLess(abs(out['joint_vel'][a, joint]), 1.)
        self.assertLess(abs(out['joint_vel'][b-1, joint]), 1.)
        middle = a+c['source_ramp_frames']*2+2
        self.assertAlmostEqual(out['joint_vel'][middle, joint], 10., delta=.01)
        self.assertLess(abs(out['joint_vel'][a-1, joint]-out['joint_vel'][a, joint]), 1.)

    def test_poses_are_native_fk_and_all_derivatives_match_assembled_motion(self):
        import mujoco
        from morphmimic.data.local_prepare import _robot_model
        from morphmimic.data.seed_convert import finite_diff_velocities
        m = api(); source, target, _ = fixture()
        out, c = m.neutralize_motion(source, target, 'q1', target_frame=0)
        _, model, ids = _robot_model('q1'); data = mujoco.MjData(model)
        for i in range(len(out['joint_pos'])):
            data.qpos[:] = np.r_[out['body_pos_w'][i, 0], out['body_quat_w'][i, 0], out['joint_pos'][i]]
            mujoco.mj_kinematics(model, data)
            np.testing.assert_allclose(out['body_pos_w'][i], data.xpos[ids], atol=2e-6)
            np.testing.assert_allclose(abs((out['body_quat_w'][i]*data.xquat[ids]).sum(axis=-1)), 1., atol=1e-6)
        for key, wanted in zip(('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'), finite_diff_velocities(
                out['joint_pos'].astype(float), out['body_pos_w'].astype(float), out['body_quat_w'].astype(float), 50)):
            np.testing.assert_allclose(out[key], wanted, atol=2e-4, rtol=1e-4)

    def test_rejects_nonfinite_inputs_and_bad_native_order(self):
        m = api(); source, target, _ = fixture()
        broken = dict(source, joint_pos=source['joint_pos'].copy()); broken['joint_pos'][0, 0] = np.nan
        with self.assertRaises(ValueError): m.neutralize_motion(broken, target, 'q1', target_frame=0)
        with self.assertRaises(ValueError): m.neutralize_motion(source, target, 'q1', target_frame=0, source_floor_offset=float('nan'))
        with self.assertRaises(ValueError): m.neutralize_motion(dict(source, joint_names=source['joint_names'][::-1]), target, 'q1', target_frame=0)

    def test_source_clock_is_monotone_covers_endpoints_and_keeps_middle_original_rate(self):
        m = api(); mapping = m.source_frame_map(101, 25)
        self.assertEqual(len(mapping), 151)
        self.assertEqual(mapping[0], 0.); self.assertEqual(mapping[-1], 100.)
        self.assertTrue((np.diff(mapping) > 0).all())
        self.assertTrue((np.diff(mapping) <= 1.).all())
        np.testing.assert_array_equal(mapping[50:-50], np.arange(25, 76))
        self.assertLess(mapping[1]-mapping[0], .002)

    def test_real_q1_flip_synthetic_bridge_does_not_add_joint_limit_excursion(self):
        from morphmimic.data.local_prepare import _robot_model
        m = api()
        manifest_path = ROOT/'prepared/importance-flips-20261116/manifest.json'
        if not manifest_path.is_file(): manifest_path = ROOT/'prepared/inplace-recovery-neutral-20261117-v2/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        row = next(r for r in manifest['clips'] if r['robot']=='q1' and r['dataset']=='seed_flips')
        if Path(row['output']).is_file():
            with np.load(row['output']) as z: source = {k:v for k,v in z.items()}
        else:
            # Regenerate the original fast edge, rather than testing an already softened clip twice.
            import hashlib,tempfile
            from morphmimic.data.local_prepare import convert_motion
            from morphmimic.data.seed_convert import finite_diff_velocities
            retarget=row['derivation']['retarget'];raw=Path(retarget['outputs']['q1']['path'])
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),retarget['outputs']['q1']['sha256'])
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'original-flip.npz'
                convert_motion(raw,path,'q1',30,source_joint_names=retarget['q1_joint_names'])
                with np.load(path) as z:source={k:v for k,v in z.items()}
            n=row.get('neutral_contract',{}).get('original_source_frames',row['frames'])
            for key in ('joint_pos','body_pos_w','body_quat_w'):source[key]=source[key][:n]
            velocity=finite_diff_velocities(*(source[key].astype(float) for key in ('joint_pos','body_pos_w','body_quat_w')),50)
            source.update(zip(('joint_vel','body_lin_vel_w','body_ang_vel_w'),velocity))
            self.assertGreater(float(abs(source['joint_vel'][0]).max()),5.)
        review = review_document()
        with np.load(review['robot_sources']['q1']['npz']) as z: target = {k:v for k,v in z.items()}
        out, c = m.neutralize_motion(source, target, 'q1')
        self.assertLessEqual(c['quality']['synthetic_joint_limit_error_rad'], 1e-6)
        self.assertTrue(c['entry']['derivative_limits_met']); self.assertTrue(c['exit']['derivative_limits_met'])
        self.assertTrue(m.validate_boundary(out,target,'q1',c)['valid'])

    def test_boundary_validator_rejects_fk_consistent_joint_excursion_inside_synthetic_bridge(self):
        import mujoco
        from morphmimic.data.local_prepare import _robot_model
        from morphmimic.data.seed_convert import finite_diff_velocities
        m = api(); source, target, joint = fixture(speed=1.)
        out, c = m.neutralize_motion(source, target, 'q1', target_frame=0)
        i = c['entry_hold_end_frame']+c['entry_frames']//2
        out['joint_pos'][i, joint] += .5
        _, model, ids = _robot_model('q1'); data = mujoco.MjData(model)
        data.qpos[:] = np.r_[out['body_pos_w'][i, 0], out['body_quat_w'][i, 0], out['joint_pos'][i]]
        mujoco.mj_kinematics(model, data)
        out['body_pos_w'][i], out['body_quat_w'][i] = data.xpos[ids], data.xquat[ids]
        out.update(zip(('joint_vel','body_lin_vel_w','body_ang_vel_w'), finite_diff_velocities(
            out['joint_pos'],out['body_pos_w'],out['body_quat_w'],50)))
        with self.assertRaises(ValueError): m.validate_boundary(out,target,'q1',c,target_frame=0)

    def test_boundary_validator_rejects_changed_neutral_and_nonzero_hold_velocity(self):
        m = api(); source, target, _ = fixture(speed=1.)
        out, c = m.neutralize_motion(source, target, 'q1', target_frame=0)
        self.assertTrue(m.validate_boundary(out, target, 'q1', c, target_frame=0)['valid'])
        broken = dict(out, joint_pos=out['joint_pos'].copy()); broken['joint_pos'][0, 0] += .1
        with self.assertRaises(ValueError): m.validate_boundary(broken, target, 'q1', c, target_frame=0)
        broken = dict(out, joint_vel=out['joint_vel'].copy()); broken['joint_vel'][0, 0] = .1
        with self.assertRaises(ValueError): m.validate_boundary(broken, target, 'q1', c, target_frame=0)

    def test_human_core_is_authentic_transformed_data_with_explicit_synthetic_timing(self):
        from morphmimic.constants import HUMAN_JOINTS
        m = api(); source, target, _ = fixture(speed=1.)
        out, c = m.neutralize_motion(source, target, 'q1', target_frame=0)
        n = len(source['joint_pos'])
        p = np.zeros((n, 24, 3)); p[:, :, 0] = np.arange(24)*.05; p[:, :, 2] = 1.
        p[:, :, 0] += np.arange(n)[:, None]*.02
        q = np.zeros((n, 24, 4)); q[:, :, 0] = 1.
        human = dict(human_pos_w=p, human_quat_w=q, human_joint_names=np.array(HUMAN_JOINTS),
                     human_joint_valid=np.ones(24, dtype=bool), fps=np.array(50), source_time_s=5.+np.arange(n)/50)
        reference = dict(human, human_pos_w=p[:2].copy(), human_quat_w=q[:2].copy())
        side = m.neutralize_human(human, reference, c, target_frame=0)
        a,b = c['source_start_frame'], c['source_stop_frame']
        mapping = m.source_frame_map(n, c['source_ramp_frames'])
        integer = mapping == np.floor(mapping)
        turn = Rotation.from_rotvec([0., 0., float(side['root_yaw_rotation_rad'])])
        wanted = turn.apply(p.reshape(-1,3)).reshape(p.shape) + side['root_translation_xyz']
        np.testing.assert_allclose(side['human_pos_w'][a:b][integer], wanted[mapping[integer].astype(int)], atol=2e-6)
        np.testing.assert_allclose(side['source_time_s'][a:b], 5.+mapping/50, atol=1e-10)
        self.assertTrue(np.isnan(side['source_time_s'][:a]).all())
        self.assertTrue(np.isnan(side['source_time_s'][b:]).all())
        np.testing.assert_array_equal(side['source_frame_index'][a:b], mapping)
        np.testing.assert_array_equal(side['source_frame_index'][:a], -1)
        np.testing.assert_allclose(np.diff(side['time_s']), .02, atol=1e-10)
        self.assertEqual(len(side['human_pos_w']), len(out['joint_pos']))
        np.testing.assert_allclose(side['human_pos_w'][:c['entry_hold_end_frame']], np.repeat(side['human_pos_w'][:1], c['entry_hold_end_frame'], axis=0), atol=1e-7)
        self.assertIn('synthetic', str(side['derivation']))
        np.testing.assert_allclose(side['human_pos_w'][0, 0, :2], 0., atol=1e-8)


if __name__ == '__main__': unittest.main()
