"""Append source-timed real recoveries ending in the reviewed native neutral pose."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from morphmimic.data.local_prepare import _robot_model
from morphmimic.data.seed_convert import finite_diff_velocities

ROBOTS = ('q1', 'bumi', 'g1')
FPS = 50
BLEND_FRAMES = 60
HOLD_FRAMES = 150
REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=12)
def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_window(window, frames):
    if not 0 <= window['start_frame'] < window['stop_frame'] <= window['stand_frame'] < frames:
        raise ValueError('Recovery reset interval must precede its original stand inside the source')


def _rotation(q):
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]])


def _wxyz(rotation):
    q = rotation.as_quat()[..., [3, 0, 1, 2]]
    return np.where(q[..., :1] < 0, -q, q)


def _aligned_target(pos, quat, original_pos, original_quat):
    """Only world yaw and root xy change; approved native height is retained."""
    yaw = _rotation(original_quat).as_euler('xyz')[2] - _rotation(quat[0]).as_euler('xyz')[2]
    rotation = Rotation.from_rotvec([0., 0., yaw])
    aligned = rotation.apply(pos - pos[0]) + pos[0]
    aligned[:, :2] += original_pos[:2] - aligned[0, :2]
    return aligned, _wxyz(rotation * _rotation(quat)), float(yaw)


def _blend_quat(original, target, weight):
    shape = original.shape
    start = _rotation(original.reshape(-1, 4))
    finish = _rotation(np.broadcast_to(target, shape).reshape(-1, 4))
    delta = (finish * start.inv()).as_rotvec().reshape(*shape[:-1], 3)
    delta *= weight.reshape((-1,) + (1,) * (delta.ndim - 1))
    return _wxyz(Rotation.from_rotvec(delta.reshape(-1, 3)) * start).reshape(shape)


def derive_clip(source, human, target, reference, window, robot, target_frame):
    """Keep every source frame through standing, then ease to neutral and freeze."""
    import mujoco
    validate_window(window, len(source['joint_pos']))
    for data, keys in ((source, ('joint_pos', 'body_pos_w', 'body_quat_w')),
                       (human, ('human_pos_w', 'human_quat_w')),
                       (target, ('joint_pos', 'body_pos_w', 'body_quat_w')),
                       (reference, ('human_pos_w', 'human_quat_w'))):
        if int(data['fps']) != FPS or any(not np.isfinite(data[k]).all() for k in keys):
            raise ValueError('Native robot and real human inputs must be finite at 50 Hz')
    spec, model, body_ids = _robot_model(robot)
    if tuple(source['joint_names']) != spec.joint_names or tuple(target['joint_names']) != spec.joint_names:
        raise ValueError('Native joint order differs from model')
    if body_ids[0] != model.jnt_bodyid[0]:
        raise ValueError('Tracked root must be the native free-joint body')
    first, recovered = window['start_frame'], window['stand_frame']
    prefix = recovered - first + 1
    stand = prefix + BLEND_FRAMES
    end = stand + HOLD_FRAMES
    if end > 15 * FPS:
        raise ValueError('Unchanged recovery plus neutral blend/hold exceeds 15 seconds')
    indices = np.r_[np.arange(first, recovered + 1),
                    np.minimum(recovered + np.arange(1, BLEND_FRAMES + 1), len(source['joint_pos']) - 1),
                    np.full(HOLD_FRAMES, recovered)]
    t = np.arange(1, BLEND_FRAMES + 1) / BLEND_FRAMES
    weight = np.r_[np.zeros(prefix), t ** 3 * (10 - 15 * t + 6 * t ** 2), np.ones(HOLD_FRAMES)]
    target_pos, target_quat, yaw = _aligned_target(target['body_pos_w'][target_frame], target['body_quat_w'][target_frame],
                                                   source['body_pos_w'][recovered, 0], source['body_quat_w'][recovered, 0])
    root_pos = source['body_pos_w'][indices, 0].astype(float)
    root_pos += weight[:, None] * (target_pos[0] - root_pos)
    root_quat = _blend_quat(source['body_quat_w'][indices, 0], target_quat[0], weight)
    joints = source['joint_pos'][indices].astype(float)
    joints += weight[:, None] * (target['joint_pos'][target_frame] - joints)
    # Duplicate the exact target before the hold so central differences are zero throughout it.
    joints[stand-1:] = target['joint_pos'][target_frame]
    root_pos[stand-1:] = target_pos[0]
    root_quat[stand-1:] = target_quat[0]
    data = mujoco.MjData(model)
    pos = np.empty((end, 14, 3)); quat = np.empty((end, 14, 4))
    for i in range(end):
        data.qpos[:] = np.r_[root_pos[i], root_quat[i], joints[i]]
        mujoco.mj_kinematics(model, data)
        pos[i], quat[i] = data.xpos[body_ids], data.xquat[body_ids]
    quat = np.where(quat[..., :1] < 0, -quat, quat)
    joint_vel, linear, angular = finite_diff_velocities(joints, pos, quat, FPS)
    out = dict(fps=np.array(FPS), joint_pos=joints, joint_vel=joint_vel, body_pos_w=pos, body_quat_w=quat,
               body_lin_vel_w=linear, body_ang_vel_w=angular, body_names=np.array(spec.tracked_bodies),
               joint_names=np.array(spec.joint_names), robot=np.array(robot), root_body=np.array(spec.root_body))
    out = {k: a.astype(np.float32) if a.dtype.kind == 'f' else a for k, a in out.items()}
    hp, hq, human_yaw = _aligned_target(reference['human_pos_w'][target_frame], reference['human_quat_w'][target_frame],
                                      human['human_pos_w'][recovered, 0], human['human_quat_w'][recovered, 0])
    side = {k: v.copy() for k, v in human.items() if k not in ('human_pos_w', 'human_quat_w', 'source_time_s')}
    positions = human['human_pos_w'][indices].astype(float)
    positions += weight[:, None, None] * (hp - positions)
    orientations = _blend_quat(human['human_quat_w'][indices], hq, weight)
    positions[:prefix] = human['human_pos_w'][first:recovered+1]
    orientations[:prefix] = human['human_quat_w'][first:recovered+1]
    positions[stand-1:] = hp; orientations[stand-1:] = hq
    side.update(human_pos_w=positions.astype(np.float32), human_quat_w=orientations.astype(np.float32),
                human_joint_valid=np.asarray(human['human_joint_valid']) & np.asarray(reference['human_joint_valid']))
    updated = dict(robot=robot, category=window['category'], start_frame=0,
                   stop_frame=window['stop_frame'] - first, stand_frame=stand, hold_end_frame=end)
    provenance = dict(source_clip=window['clip'], source_start_frame=first, source_stand_frame=recovered,
                      original_prefix_frames=prefix, source_frame_interval=[first, recovered+1],
                      blend_frames=BLEND_FRAMES, hold_frames=HOLD_FRAMES, robot_alignment_yaw_rad=yaw,
                      human_alignment_yaw_rad=human_yaw, target_frame=target_frame,
                      derivation='Unchanged native recovery through original stand; quintic whole-body native-pose blend; frozen hold. Real paired human poses blended independently; no robot FK human labels.')
    return out, side, updated, provenance


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _identity(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def prepare(manifest_path, windows_path, review_path, floor_path, output):
    manifest_path, windows_path, review_path, floor_path, output = map(Path, (manifest_path, windows_path, review_path, floor_path, output))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text()); windows = json.loads(windows_path.read_text())
    review = json.loads(review_path.read_text()); floor = json.loads(floor_path.read_text())
    old_sha = sha256(manifest_path)
    if manifest.get('complete') is not True or windows['training_manifest_sha256'] != old_sha or floor['prepared_manifest_sha256'] != old_sha:
        raise ValueError('Original manifest, recovery windows and floor must share one complete identity')
    if review['prepared_frame_zero_based'] != 11446 or review['prepared_fps'] != FPS:
        raise ValueError('Approved review target identity differs')
    if sha256(review['source_bvh']) != review['source_bvh_sha256']:
        raise ValueError('Approved real human BVH identity differs')
    rows = {(r['robot'], r['name']): r for r in manifest['clips']}
    subsets = {r: json.loads((manifest_path.parent/r/'train.yaml').read_text()) for r in ROBOTS}
    for robot in ROBOTS:
        if sha256(manifest_path.parent/robot/'train.yaml') != windows['subset_sha256'][robot]:
            raise ValueError('Original training subset identity differs')
    for row in manifest['clips']:
        for key, digest in [('output', 'output_sha256')] + ([('human_sidecar', 'human_sha256')] if row.get('human_available') else []):
            if sha256(row[key]) != row[digest]:
                raise ValueError('Original prepared content identity differs: ' + row[key])
    output.mkdir(parents=True)
    human_root = output/'human50'
    for robot in ROBOTS:
        (output/robot).mkdir(); (human_root/robot).mkdir(parents=True)
    for row in manifest['clips']:
        if row.get('human_available'):
            (human_root/row['robot']/(row['name']+'.npz')).symlink_to(row['human_sidecar'])
    added, new_windows, added_floor = [], [], []
    floor_rows = {(r['robot'], r['name']): r for r in floor['clips']}
    approved = dict(review_manifest=_identity(review_path), frame=11446, fps=FPS, robots={})
    for robot in ROBOTS:
        import mujoco
        _, model, _ = _robot_model(robot)
        buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=buffer)
        asset_sha = hashlib.sha256(buffer.tobytes()).hexdigest()
        target_row = rows[robot, 'walk2_subject4']; approved_robot = review['robot_sources'][robot]
        if target_row['output'] != approved_robot['npz'] or sha256(target_row['output']) != approved_robot['sha256']:
            raise ValueError('Approved native target identity differs')
        if asset_sha != target_row['asset_sha256']:
            raise ValueError('Native model fingerprint differs from approved source asset')
        approved['robots'][robot] = dict(compiled_model_sha256=asset_sha, native=_identity(target_row['output']), human=_identity(target_row['human_sidecar']))
    for number, window in enumerate(windows['windows']):
        robot = window['robot']; source = rows[robot, window['clip']]; target = rows[robot, 'walk2_subject4']
        if source['split'] != 'train' or source['output'] not in subsets[robot]['clips']:
            raise ValueError('Recovery source is not in original training inventory')
        motion, side, updated, provenance = derive_clip(load_npz(source['output']), load_npz(source['human_sidecar']),
            load_npz(target['output']), load_npz(target['human_sidecar']), window, robot, 11446)
        name = f"{window['clip']}__neutral_{window['category']}_{number:03d}"
        motion_path = output/robot/(name+'.npz'); human_path = human_root/robot/(name+'.npz')
        np.savez(motion_path, **motion); np.savez(human_path, **side)
        provenance.update(source_npz=_identity(source['output']), source_human=_identity(source['human_sidecar']),
                          approved_target=approved['robots'][robot], review_manifest=approved['review_manifest'])
        row = dict(name=name, dataset='neutral_recovery', source_id='derived/'+source['source_id']+'/'+name,
                   group_id=source['group_id'], robot=robot, split='train', fps=FPS,
                   frames=len(motion['joint_pos']), duration_seconds=(len(motion['joint_pos'])-1)/FPS,
                   output=str(motion_path), output_sha256=sha256(motion_path), human_sidecar=str(human_path),
                   human_sha256=sha256(human_path), human_available=True, alignment_eligible=True,
                   human_joint_valid=side['human_joint_valid'].tolist(), asset_sha256=source['asset_sha256'],
                   derived=True, derivation=provenance)
        added.append(row); new_windows.append(dict(updated, clip=name, source_id=row['source_id']))
        original_floor = floor_rows[robot, window['clip']]
        added_floor.append(dict(original_floor, dataset='neutral_recovery', name=name, path=str(motion_path),
                                sha256=row['output_sha256'], method='inherited-original-source-offset-v1',
                                reason='Original source floor offset preserved for derived native recovery',
                                source_clip=window['clip']))
    new_manifest = dict(manifest, clips=manifest['clips']+added, human_root=str(human_root),
                        derivation=dict(parent_manifest=_identity(manifest_path), recovery_windows=_identity(windows_path),
                                        approved_target=approved, original_clip_count=len(manifest['clips']), appended_clip_count=len(added)))
    _json(output/'manifest.json', new_manifest)
    new_sha = sha256(output/'manifest.json')
    _json(output/'contact-floor.json', dict(floor, clips=floor['clips']+added_floor, prepared_manifest_sha256=new_sha))
    transition = dict(schema_version=1, parent_manifest=_identity(manifest_path), manifest=_identity(output/'manifest.json'),
                      parent_contact_floor=_identity(floor_path), contact_floor=_identity(output/'contact-floor.json'),
                      approved_target=approved, robots={})
    subset_hashes = {}
    for robot in ROBOTS:
        appended = [r for r in added if r['robot'] == robot]
        _json(output/robot/'train.yaml', dict(subsets[robot], clips=subsets[robot]['clips']+[r['output'] for r in appended]))
        (output/robot/'val.yaml').write_bytes((manifest_path.parent/robot/'val.yaml').read_bytes())
        old_rows = [rows[robot, Path(path).stem] for path in subsets[robot]['clips']]
        transition['robots'][robot] = dict(parent_subset=_identity(manifest_path.parent/robot/'train.yaml'), subset=_identity(output/robot/'train.yaml'),
            original_clips=[r['name'] for r in old_rows], appended_clips=[r['name'] for r in appended],
            original_clip_frames=[r['frames'] for r in old_rows], appended_clip_frames=[r['frames'] for r in appended],
            original_frames=sum(r['frames'] for r in old_rows), appended_frames=sum(r['frames'] for r in appended),
            original_clip_count=len(old_rows), appended_clip_count=len(appended))
        subset_hashes[robot] = sha256(output/robot/'train.yaml')
    _json(output/'transition.json', transition)
    preview_poses = {category: {} for category in ('lying', 'seated', 'half_squat')}
    for category in preview_poses:
        for robot in ROBOTS:
            choices = [w for w in new_windows if w['robot'] == robot and w['category'] == category]
            selected = max(choices, key=lambda w: w['stop_frame'] - w['start_frame'])
            preview_poses[category][robot] = dict(clip=selected['clip'], start_frame=(selected['start_frame']+selected['stop_frame'])//2, hold_frame=selected['stand_frame'])
    _json(output/'windows.json', dict(schema_version=1, fps=FPS, training_manifest_sha256=new_sha,
        subset_sha256=subset_hashes, frame_interval='start inclusive, stop exclusive', max_recovery_seconds=15.,
        stand_hold_seconds=3., calibration=windows['calibration'], windows=new_windows, preview_poses=preview_poses,
        evaluation_cases=[dict(w, start_frame=(w['start_frame']+w['stop_frame'])//2,
                              stop_frame=(w['start_frame']+w['stop_frame'])//2+1,
                              case_id=f"{w['robot']}/{w['category']}/{w['clip']}") for w in new_windows],
        approved_target=approved, derivation='Real original reset labels only; unchanged source timing; 1.2s neutral blend then3s hold'))
    if sha256(manifest_path) != old_sha:
        raise ValueError('Original manifest changed during preparation')
    return transition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPO/'prepared/allopt-20261112/manifest.json')
    parser.add_argument('--windows', type=Path, default=REPO/'prepared/recovery-direct-20261114/windows.json')
    parser.add_argument('--review', type=Path, default=REPO/'runs/reports/neutral-standing-review-20261116/review-manifest.json')
    parser.add_argument('--contact-floor', type=Path, default=REPO/'prepared/allopt-20261112/contact-floor.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.manifest.resolve(), args.windows.resolve(), args.review.resolve(), args.contact_floor.resolve(), args.output.resolve())
    print(json.dumps(result['robots'], indent=2))


if __name__ == '__main__':
    main()
