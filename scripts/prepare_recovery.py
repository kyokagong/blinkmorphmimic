"""Extract geometry-labelled recovery reset ranges from unchanged 50 Hz training clips."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROBOTS = ('q1', 'bumi', 'g1')
CATEGORIES = ('lying', 'seated', 'half_squat')
FPS = 50
TRAINING_MANIFEST_SHA256 = '088d90efbe44ab8af6bfc977bcd7a74e49138c36a2c3c7cf416ff4bd96af08a0'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_motion(data):
    pos, quat, joints = (data[k] for k in ('body_pos_w', 'body_quat_w', 'joint_pos'))
    frames = len(pos)
    if int(data['fps']) != FPS or frames < 2:
        raise ValueError('Recovery motions require at least two frames at 50 Hz')
    if pos.shape != (frames, 14, 3) or quat.shape != (frames, 14, 4) or joints.ndim != 2 or len(joints) != frames:
        raise ValueError('Invalid recovery FK/joint shapes')
    names = list(data['body_names'])
    if (len(names) != 14 or names[0] != str(data['root_body']) or
        not all('knee' in names[i] for i in (2, 5)) or
        not all('ankle' in names[i] for i in (3, 6))):
        raise ValueError('Recovery requires root/knee/ankle tracked-body order')
    for key in ('body_pos_w', 'body_quat_w', 'joint_pos', 'body_lin_vel_w', 'body_ang_vel_w'):
        if not np.isfinite(data[key]).all():
            raise ValueError('Nonfinite recovery motion: ' + key)
    if any(data[k].shape != pos.shape for k in ('body_lin_vel_w', 'body_ang_vel_w')):
        raise ValueError('Invalid recovery velocity shapes')
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1., atol=1e-3, rtol=0):
        raise ValueError('Recovery quaternions must be unit wxyz')
    knee_indices = [i for i, name in enumerate(data['joint_names']) if 'knee' in name]
    if len(knee_indices) != 2 or len(data['joint_names']) != joints.shape[1]:
        raise ValueError('Recovery requires both named knee joints')
    return knee_indices


def calibrate(motions):
    """Measure each robot's leg height and ankle origin floor from upright FK."""
    heights, floors = [], []
    for data in motions:
        knees = validate_motion(data)
        q = data['body_quat_w'][:, 0]
        up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        straight = (up > .95) & (np.max(np.abs(data['joint_pos'][:, knees]), axis=1) < .45)
        pos = data['body_pos_w'][straight]
        if len(pos):
            ankle = np.min(pos[:, [3, 6], 2], axis=1)
            floors.append(ankle)
            heights.append(pos[:, 0, 2] - ankle)
    if not heights:
        raise ValueError('No upright straight-leg frames for robot calibration')
    height = float(np.median(np.concatenate(heights)))
    if height <= 0:
        raise ValueError('Nonpositive standing leg height')
    return dict(standing_leg_height_m=height, ankle_floor_m=float(np.median(np.concatenate(floors))))


def classify_frames(data, calibration):
    knees = validate_motion(data)
    height, floor = (float(calibration[k]) for k in ('standing_leg_height_m', 'ankle_floor_m'))
    if not np.isfinite([height, floor]).all() or height <= 0:
        raise ValueError('Invalid recovery height calibration')
    pos, q = data['body_pos_w'], data['body_quat_w'][:, 0]
    h = (pos[:, 0, 2] - floor) / height
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    knee = np.abs(data['joint_pos'][:, knees])
    # Both thighs must be nearly horizontal; averaging can mistake kneeling for sitting.
    thigh_drop = np.max(pos[:, 0, None, 2] - pos[:, [2, 5], 2], axis=1) / height
    labels = np.full(len(pos), '', dtype='<U10')
    labels[(up < .55) & (h < .50)] = 'lying'
    labels[(up >= .55) & (h < .56) & (thigh_drop < .18)] = 'seated'
    grounded = np.max(np.abs(pos[:, [3, 6], 2] - floor), axis=1) < .15 * height
    knees_above_feet = np.min(pos[:, [2, 5], 2] - pos[:, [3, 6], 2], axis=1) > .05 * height
    labels[(up > .65) & (h >= .56) & (h < .80) & (knee.min(axis=1) > .35) & grounded & knees_above_feet] = 'half_squat'
    velocity = data['body_lin_vel_w'][:, 0]
    angular = data['body_ang_vel_w'][:, 0]
    standing = ((up > .92) & (h > .90) & (knee.max(axis=1) < .95) &
                (np.abs(velocity[:, 2]) < .25 * height) &
                (np.linalg.norm(velocity[:, :2], axis=1) < .60 * height) &
                (np.linalg.norm(angular, axis=1) < 1.25))
    return labels, standing, dict(height_ratio=h, root_up_cos=up, thigh_drop_ratio=thigh_drop)


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def recovery_ranges(labels, standing, height_ratios):
    """Half-open reset intervals, each completing a continuous 1 s stand within 10 s."""
    labels, standing, height_ratios = np.asarray(labels), np.asarray(standing), np.asarray(height_ratios)
    if height_ratios.shape != labels.shape or not np.isfinite(height_ratios).all():
        raise ValueError('Recovery height ratios must be matching finite vectors')
    if labels.ndim != 1 or standing.shape != labels.shape or standing.dtype != np.bool_:
        raise ValueError('Recovery labels and standing mask must be matching vectors')
    if not np.isin(labels, ('',) + CATEGORIES).all():
        raise ValueError('Unknown recovery category')
    starts = np.array([a for a, b in _runs(standing) if b - a >= FPS], dtype=int)
    windows = []
    for category in CATEGORIES:
        for first, stop in _runs(labels == category):
            future = starts[starts >= stop]
            if not len(future):
                continue
            stand = int(future[0]); first = max(int(first), stand + FPS - 10 * FPS)
            if stop - first < 10:
                continue
            valid = np.ones(stop - first, dtype=bool)
            if category != 'lying':
                future_min = np.minimum.accumulate(height_ratios[first:stand+1][::-1])[::-1][:stop-first]
                future_lying = np.maximum.accumulate((labels[first:stand+1] == 'lying')[::-1])[::-1][:stop-first]
                valid &= (future_min >= height_ratios[first:stop] - .15 - 1e-6) & ~future_lying
            for left, right in _runs(valid):
                if right - left >= 10:
                    windows.append(dict(category=category, start_frame=first + int(left), stop_frame=first + int(right),
                                        stand_frame=stand, hold_end_frame=stand + FPS))
    return windows


def build_dataset(manifest_path, expected_manifest_sha256=TRAINING_MANIFEST_SHA256):
    manifest_path = Path(manifest_path)
    manifest_sha = sha256(manifest_path)
    if manifest_sha != expected_manifest_sha256:
        raise ValueError('Training manifest SHA differs from expected identity')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('complete') is not True:
        raise ValueError('Training manifest is incomplete')
    subset_sha, subset_paths = {}, {}
    for robot in ROBOTS:
        path = manifest_path.parent / robot / 'train.yaml'
        subset_sha[robot] = sha256(path)
        subset = json.loads(path.read_text())
        subset_paths[robot] = {str((Path(subset['root']) / p).resolve()) for p in subset['clips']}
    rows = [row for row in manifest['clips'] if row['robot'] in ROBOTS and
            row['split'] == 'train' and row['dataset'] == 'lafan1' and row['name'].startswith('fallAndGetUp')]
    rows.sort(key=lambda row: (ROBOTS.index(row['robot']), row['name']))
    data_by_robot = {r: [] for r in ROBOTS}
    sources = []
    for row in rows:
        path = Path(row['output'])
        if str(path.resolve()) not in subset_paths[row['robot']] or path.stem != row['name']:
            raise ValueError('Recovery source is absent from matching training subset: ' + str(path))
        if sha256(path) != row['output_sha256']:
            raise ValueError('Recovery source SHA differs: ' + str(path))
        if row.get('human_sidecar') and sha256(row['human_sidecar']) != row['human_sha256']:
            raise ValueError('Recovery human SHA differs: ' + row['human_sidecar'])
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        if str(data['robot']) != row['robot']:
            raise ValueError('Recovery robot identity mismatch')
        validate_motion(data)
        if row.get('frames', len(data['joint_pos'])) != len(data['joint_pos']):
            raise ValueError('Recovery source frame count differs from manifest')
        data_by_robot[row['robot']].append((row, data))
        sources.append({k: row[k] for k in ('robot', 'name', 'source_id', 'output', 'output_sha256')})
    calibration = {robot: calibrate([d for _, d in motions]) for robot, motions in data_by_robot.items()}
    windows = []
    for robot, motions in data_by_robot.items():
        for row, data in motions:
            labels, standing, diagnostics = classify_frames(data, calibration[robot])
            for window in recovery_ranges(labels, standing, diagnostics['height_ratio']):
                first, stop = window['start_frame'], window['stop_frame']
                future_min = np.minimum.accumulate(diagnostics['height_ratio'][first:window['stand_frame']+1][::-1])[::-1][:stop-first]
                window.update(robot=robot, clip=row['name'], source_id=row['source_id'],
                              max_pelvis_drop_before_stand_ratio=float((diagnostics['height_ratio'][first:stop] - future_min).max()),
                              min_height_ratio=float(diagnostics['height_ratio'][first:stop].min()),
                              max_height_ratio=float(diagnostics['height_ratio'][first:stop].max()),
                              min_root_up_cos=float(diagnostics['root_up_cos'][first:stop].min()),
                              max_root_up_cos=float(diagnostics['root_up_cos'][first:stop].max()))
                windows.append(window)
    cases = []
    for robot in ROBOTS:
        for category in CATEGORIES:
            candidates = [w for w in windows if w['robot'] == robot and w['category'] == category]
            if not candidates:
                raise ValueError(f'No truthful {robot}/{category} recovery ranges')
            # Regression reset starts are from the existing training inventory, not unseen motions.
            window = max(candidates, key=lambda w: w['stop_frame'] - w['start_frame'])
            case = dict(window)
            case['start_frame'] = (window['start_frame'] + window['stop_frame']) // 2
            case['stop_frame'] = case['start_frame'] + 1
            case['case_id'] = f'{robot}/{category}/representative'
            cases.append(case)
        failure = [w for w in windows if w['robot'] == robot and w['clip'] == 'fallAndGetUp2_subject2' and
                   w['start_frame'] <= 650 < w['stop_frame']]
        if failure:
            cases.append(dict(failure[0], start_frame=650, stop_frame=651, case_id=f'{robot}/known_failure/frame650'))
    if sha256(manifest_path) != manifest_sha or any(sha256(manifest_path.parent / r / 'train.yaml') != subset_sha[r] for r in ROBOTS):
        raise ValueError('Training inventory changed while preparing recovery windows')
    return dict(schema_version=1, fps=FPS, training_manifest_sha256=manifest_sha, subset_sha256=subset_sha,
                frame_interval='start inclusive, stop exclusive', max_recovery_seconds=10., stand_hold_seconds=1.,
                calibration=calibration, selected_sources=sources,
                direct_recovery=dict(max_non_lying_pelvis_drop_leg_ratio=.15, allow_lying_before_stand_for_non_lying_resets=False),
                evaluation_inventory='Regression reset starts from unchanged training clips; not unseen motions',
                thresholds=dict(lying='root_up_cos < .55; floor-relative pelvis height < .50 leg height',
                                seated='root_up_cos >= .55; height < .56; BOTH pelvis-to-knee vertical drops < .18 leg height',
                                half_squat='root_up_cos > .65; .56 <= height < .80; BOTH knee flexions > .35 rad; BOTH ankles within .15 leg height of calibrated floor; BOTH knees > .05 leg height above ankles',
                                standing='root_up_cos > .92; height > .90; knee flexions < .95 rad; vertical speed < .25 leg height/s; horizontal speed < .60 leg height/s; angular speed < 1.25 rad/s'),
                windows=windows, evaluation_cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'prepared/allopt-20261112/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_dataset(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), counts=dict(Counter(w['robot'] + '/' + w['category'] for w in result['windows'])),
                          evaluation_cases=len(result['evaluation_cases'])), indent=2))


if __name__ == '__main__':
    main()
