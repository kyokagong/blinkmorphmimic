"""Append six authentic BONES-SEED flip takes without changing the existing mix."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np
from scipy.spatial.transform import Rotation
from morphmimic.data.local_prepare import _robot_model, _retarget_function, convert_motion, sha256
from morphmimic.data.seed_convert import finite_diff_velocities, parse_seed_csv
from morphmimic.data.human_convert import convert_human_clip
from morphmimic.data.contact_calibration import FootContactGeometry, calibrate_reference_floor

REPO = Path(__file__).resolve().parents[1]
ROBOTS = ('q1', 'bumi', 'g1')
FPS = 50
# Hold out entire upstream takes, including all actor variants and mirrors.
TAKES = {'flip_180_R_001__A304': 'train', 'flip_180_R_004__A415': 'train',
         'flip_180_R_006__A304': 'val', 'flip_360_009__A416': 'train',
         'flip_360_001__A304': 'train', 'flip_360_011__A416': 'val'}


def identity(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(Path(path)))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def longest_interval(mask, containing=None):
    changes = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    spans = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    if containing is not None:
        spans = [span for span in spans if span[0] <= containing < span[1]]
    return max(spans, key=lambda x: x[1]-x[0], default=(0, 0))


def flip_metrics(motion, geometry, floor_offset=0.):
    """Body-local pitch integral distinguishes somersaults from yaw/cartwheels."""
    q = motion['body_quat_w'][:, 0]
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    delta = (rotation[:-1].inv() * rotation[1:]).as_rotvec()
    integral = delta.sum(axis=0)
    sole = geometry.sole_heights(motion['body_pos_w'][:, [3, 6]], motion['body_quat_w'][:, [3, 6]])
    up = rotation.as_matrix()[:, 2, 2]
    start, stop = longest_interval((sole - floor_offset).min(axis=1) > .08, int(up.argmin()) if up.min() < -.85 else None)
    pitch = float(integral[1])
    # A twisting somersault has noncommuting rotations: local pitch need not integrate to exactly 2π.
    return dict(pitch_rad=pitch, local_rotation_integral_rad=integral.tolist(), minimum_up=float(up.min()),
                flight_seconds=float((stop-start)/FPS), flight_window=[int(start), int(stop)],
                minimum_sole_m=float(sole.min()), maximum_sole_m=float(sole.max()),
                maximum_root_step_m=float(np.linalg.norm(np.diff(motion['body_pos_w'][:, 0], axis=0), axis=1).max()),
                maximum_root_rotation_step_rad=float(np.linalg.norm(delta, axis=1).max()),
                full_sagittal_rotation=bool(abs(pitch) > 4.8 and abs(pitch) > 1.3*max(abs(integral[0]), abs(integral[2]))
                                            and up.min() < -.85 and (stop-start)/FPS >= .12))


def validate_native(motion, robot):
    """Reject corrupt references; report physical limits separately from source inclusion."""
    import mujoco
    spec, model, ids = _robot_model(robot)
    n = len(motion['joint_pos'])
    if int(motion['fps']) != FPS or tuple(motion['joint_names']) != spec.joint_names or tuple(motion['body_names']) != spec.tracked_bodies:
        raise ValueError('Native fps/joint/body order differs')
    for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
        if len(motion[key]) != n or not np.isfinite(motion[key]).all():
            raise ValueError('Nonfinite/inconsistent native reference')
    if not np.allclose(np.linalg.norm(motion['body_quat_w'], axis=-1), 1, atol=1e-5):
        raise ValueError('Nonunit native quaternion')
    data = mujoco.MjData(model); minimum_contact = 0.; collision_frames = 0
    for i in range(n):
        data.qpos[:] = np.r_[motion['body_pos_w'][i, 0], motion['body_quat_w'][i, 0], motion['joint_pos'][i]]
        mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data); mujoco.mj_collision(model, data)
        distance = min((c.dist for c in data.contact), default=0.)
        minimum_contact = min(minimum_contact, float(distance)); collision_frames += distance < -.002
        if not np.allclose(data.xpos[ids], motion['body_pos_w'][i], atol=2e-6, rtol=0):
            raise ValueError('Reference is not native FK')
        if not np.allclose(np.abs(np.sum(data.xquat[ids]*motion['body_quat_w'][i], axis=-1)), 1, atol=1e-5):
            raise ValueError('Reference rotation differs from native FK')
    for key, wanted in zip(('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'), finite_diff_velocities(
            motion['joint_pos'].astype(float), motion['body_pos_w'].astype(float), motion['body_quat_w'].astype(float), FPS)):
        if not np.allclose(motion[key], wanted, atol=2e-4, rtol=1e-4):
            raise ValueError('Reference derivatives differ from native poses')
    q = motion['joint_pos']; limits = model.jnt_range[1:]
    return dict(maximum_joint_limit_error_rad=float(np.maximum(limits[:, 0]-q, q-limits[:, 1]).max(initial=0)),
                maximum_joint_speed_rad_s=float(np.abs(motion['joint_vel']).max()),
                minimum_self_contact_distance_m=minimum_contact, self_collision_frames_over_2mm=int(collision_frames),
                dynamics_qualified=False, qualification='Kinematics/provenance checked; torque, landing stability and real hardware are unqualified.')


def model_identity(robot):
    import mujoco
    model = _robot_model(robot)[1]
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def retarget_q1(source, output, name, repo):
    """Reuse the existing bounded IK, retaining difficult source motion before repair."""
    path = repo/'tools/retarget_q1.py'; original = path.read_text()
    replacements = {
        'from supermimic.robot import G1_JOINT_NAMES': 'from morphmimic.constants import JOINT_NAMES as G1_JOINT_NAMES',
        'result=ik.ground_motion(ik.repair_interpolation(original))': 'result=original',
        "if min_ground<-.001 or min_contact<-.001 or speed>1+1e-5:\n            raise ValueError(f'Final audit ground={min_ground}, self={min_contact}, speed_ratio={speed}')": '# Kinematic source retention does not claim physical qualification.',
        "record['status']=='passed'": "record['status']=='kinematic_only'",
        "status='passed',frames=len(result)": "status='kinematic_only',frames=len(result)",
    }
    adapted = original
    for before, after in replacements.items():
        if adapted.count(before) != 1:
            raise ValueError('Upstream Q1 adapter no longer matches: '+before)
        adapted = adapted.replace(before, after)
    module = ModuleType('flip_q1_retarget'); module.__file__ = str(path)
    sys.path.insert(0, str(repo)); exec(compile(adapted, str(path), 'exec'), module.__dict__)
    output.mkdir(parents=True, exist_ok=True); (output/'reports').mkdir(exist_ok=True)
    _, record = module.retarget((str(source), str(output), name, dict(sha256=sha256(source/(name+'.csv')))))
    if record['status'] != 'kinematic_only':
        raise ValueError('Q1 source retarget failed: '+str(record))
    record['adaptation'] = dict(source=identity(path), adapted_source_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
        replacements=replacements, reason='Retain complete hard flip motion without flat-ground repairs; report collision/speed limits without excluding the source.')
    return record


def build_sources(cache, retarget_repo):
    """Reuse raw 120 Hz data and source-timed 30 Hz retarget helpers, without alias weights."""
    from morphmimic.data.seed_convert import resample
    from supermimic import bumi
    records = {}
    for folder in ('native30', 'q130', 'bumi30'):
        (cache/folder).mkdir(exist_ok=True)
    for name in TAKES:
        receipt = cache/(name+'.retarget.json')
        native = cache/'native'/(name+'.csv'); bvh = cache/'human'/(name+'.bvh')
        if not native.is_file():
            raise FileNotFoundError('Extract the selected native CSV first: '+name)
        inputs = dict(native=identity(native), helper_q1=identity(retarget_repo/'tools/retarget_q1.py'),
                      helper_bumi=identity(retarget_repo/'tools/retarget_bumi.py'), models={r:model_identity(r) for r in ROBOTS},
                      dependencies={n:identity(retarget_repo/'supermimic'/n) for n in ('q1_ik.py', 'bumi_ik.py', 'retarget_geometry.py')})
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record['inputs'] != inputs or any(sha256(Path(r['path'])) != r['sha256'] for r in record['outputs'].values()):
                raise ValueError('Cached retarget source/helper/output changed')
            records[name] = record; continue
        pos, quat, joints = resample(*parse_seed_csv(str(native)), 120, 30)
        g1 = cache/'native30'/(name+'.csv')
        np.savetxt(g1, np.c_[pos, quat[:, [1, 2, 3, 0]], joints], delimiter=',', fmt='%.9f')
        q1 = retarget_q1(cache/'native30', cache/'q130', name, retarget_repo)
        bpath = cache/'bumi30'/(name+'.csv')
        bstats = _retarget_function(str(retarget_repo))(g1, bpath)
        bstats['adaptation'] = _retarget_function(str(retarget_repo)).adaptation
        record = dict(inputs=inputs, outputs=dict(q1=identity(cache/'q130'/(name+'.csv')), bumi=identity(bpath)),
                      q1=q1, bumi=bstats, native30=identity(g1), retarget_fps=30,
                      q1_joint_names=list(_robot_model('q1')[0].joint_names), bumi_joint_names=bumi.JOINT_NAMES)
        write_json(receipt, record); records[name] = record
        print('retargeted', name, flush=True)
    return records


def prepare(parent, output, cache, retarget_repo):
    parent, output, cache, retarget_repo = (Path(x).resolve() for x in (parent, output, cache, retarget_repo))
    if output.exists():
        raise FileExistsError(output)
    manifest_path = parent/'manifest.json'; old_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text()); floor = json.loads((parent/'contact-floor.json').read_text())
    windows = json.loads((parent/'windows.json').read_text()); previous_transition = json.loads((parent/'transition.json').read_text())
    if not manifest.get('complete') or floor['prepared_manifest_sha256'] != old_sha or windows['training_manifest_sha256'] != old_sha:
        raise ValueError('Parent manifest/floor/windows are not one complete dataset')
    subsets = {(r, s): json.loads((parent/r/(s+'.yaml')).read_text()) for r in ROBOTS for s in ('train', 'val')}
    for robot in ROBOTS:
        if windows['subset_sha256'][robot] != sha256(parent/robot/'train.yaml'):
            raise ValueError('Parent training subset differs from getup windows')
    assets = {r: model_identity(r) for r in ROBOTS}
    if assets != {r: previous_transition['approved_target']['robots'][r]['compiled_model_sha256'] for r in ROBOTS}:
        raise ValueError('Current native models differ from the approved ordinary target')
    selection = json.loads((cache/'selection.json').read_text())
    if {r['filename'] for r in selection} != set(TAKES) or any(r['is_mirror'] for r in selection):
        raise ValueError('Expected six authentic unmirrored selected takes')
    retargets = build_sources(cache, retarget_repo)
    output.mkdir(parents=True)
    human_root = output/'human50'
    for robot in ROBOTS:
        (output/robot).mkdir(); (human_root/robot).mkdir(parents=True)
    for row in manifest['clips']:
        if row.get('human_available'):
            (human_root/row['robot']/(row['name']+'.npz')).symlink_to(row['human_sidecar'])
    added, floors, previews = [], [], []
    by_name = {r['filename']: r for r in selection}
    for name, split in TAKES.items():
        clip = 'seed_flip_'+name.removeprefix('flip_')
        meta = by_name[name]; original = cache/'native'/(name+'.csv'); bvh = cache/'human'/(name+'.bvh')
        human = cache/(name+'.human50.npz')
        convert_human_clip(str(bvh), str(human)); side = load_npz(human)
        side['human_joint_valid'] = np.ones(24, dtype=bool)
        side['source_time_s'] = np.arange(len(side['human_pos_w']))/FPS
        motions = {}
        for robot in ROBOTS:
            path = output/robot/(clip+'.npz')
            if robot == 'g1':
                convert_motion(original, path, robot, 120, seed=True)
            else:
                source = cache/(robot+'30')/(name+'.csv')
                source_names = retargets[name][robot+'_joint_names']
                convert_motion(source, path, robot, 30, source_joint_names=source_names)
            motions[robot] = load_npz(path)
        frames = min(len(side['human_pos_w']), *(len(x['joint_pos']) for x in motions.values()))
        for key in ('human_pos_w', 'human_quat_w', 'source_time_s'):
            side[key] = side[key][:frames]
        source_id = 'seed_flips/'+name
        group = 'seed_flips/'+re.sub(r'__A\d+$', '', name)
        for robot, motion in motions.items():
            for key in ('joint_pos', 'body_pos_w', 'body_quat_w'):
                motion[key] = motion[key][:frames]
            velocity = finite_diff_velocities(motion['joint_pos'].astype(float), motion['body_pos_w'].astype(float), motion['body_quat_w'].astype(float), FPS)
            motion.update(zip(('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'), (x.astype(np.float32) for x in velocity)))
            path = output/robot/(clip+'.npz'); side_path = human_root/robot/(clip+'.npz')
            np.savez(path, **motion); np.savez(side_path, **side)
            spec, model, _ = _robot_model(robot); geometry = FootContactGeometry.from_mujoco(model, spec)
            calibrated = calibrate_reference_floor(motion['body_pos_w'][:, [3, 6]], motion['body_quat_w'][:, [3, 6]],
                motion['body_lin_vel_w'][:, [3, 6]], motion['body_ang_vel_w'][:, [3, 6]], geometry, max_offset=.12)
            metrics = flip_metrics(motion, geometry, calibrated.offset_m)
            if not metrics['full_sagittal_rotation'] or metrics['maximum_root_step_m'] > .25 or metrics['maximum_root_rotation_step_rad'] > .9:
                raise ValueError('Not a complete continuous airborne sagittal flip: '+robot+'/'+name+str(metrics))
            validation = validate_native(motion, robot)
            row = dict(name=clip, dataset='seed_flips', source_id=source_id, group_id=group, robot=robot, split=split,
                fps=FPS, frames=frames, duration_seconds=(frames-1)/FPS, output=str(path), output_sha256=sha256(path),
                human_sidecar=str(side_path), human_sha256=sha256(side_path), human_available=True, alignment_eligible=True,
                human_joint_valid=[True]*24, asset_sha256=assets[robot], original_source=str(original), original_sha256=sha256(original),
                source_fps=120, source_frames=meta['move_duration_frames'], source_body_window_original=[0, frames],
                pair_alignment=dict(robot_start_frame=0, robot_stop_frame=frames, human_tail_frames=len(load_npz(human)['human_pos_w'])-frames),
                phases={}, source_duration_seconds=(frames-1)/FPS, flip_metrics=metrics, validation=validation,
                derivation=dict(source_native=identity(original), source_human=identity(bvh), source_metadata=meta,
                    source_revision='2f59b2077b9da34dd4e43618e705c7cb962c9a66', retarget=retargets[name] if robot != 'g1' else None,
                    pairing='Shared source time t=k/50; only common tail trimmed; no retiming, mirrors, sampling aliases or synthetic human labels.'))
            added.append(row)
            floors.append(dict(path=str(path), sha256=row['output_sha256'], robot=robot, name=clip, dataset='seed_flips', split=split, **asdict(calibrated)))
            previews.append(dict(robot=robot, clip=clip, source_id=source_id, split=split, source_fps=120, source_window=[0, frames],
                fps=FPS, frames=frames, pitch_rad=metrics['pitch_rad'], flight_seconds=metrics['flight_seconds'],
                source=identity(original), human_source=identity(bvh)))
            print('prepared', robot, name, split, frames, round(metrics['pitch_rad'], 3), round(metrics['flight_seconds'], 3), flush=True)
    candidates = json.loads((Path('/data/kk/data/supermimic_g1/seed_flips')/'flip_candidates.json').read_text())
    exclusions = [dict(dataset='seed_flips', name=r['filename'], reason='mirror of upstream take' if r['is_mirror'] else
        'Outside the six-source representative front/back flip subset; not rejected for dynamics or difficulty.')
        for r in candidates if r['filename'] not in TAKES]
    new_manifest = dict(manifest, clips=manifest['clips']+added, human_root=str(human_root),
        excluded=manifest.get('excluded', [])+exclusions,
        derivation=dict(parent_manifest=identity(manifest_path), approved_target=previous_transition['approved_target'],
            original_clip_count=len(manifest['clips']), appended_clip_count=len(added), selected_sources=identity(cache/'selection.json'),
            split_policy='Whole upstream take group held out across every actor, mirror and robot.', dynamics_qualified=False))
    write_json(output/'manifest.json', new_manifest); new_sha = sha256(output/'manifest.json')
    all_floors = floor['clips']+floors
    summary = {r:dict(valid=sum(x['robot'] == r and x['valid'] for x in all_floors),
                      unknown=sum(x['robot'] == r and not x['valid'] for x in all_floors)) for r in ROBOTS}
    write_json(output/'contact-floor.json', dict(floor, clips=all_floors, summary=summary, prepared_manifest_sha256=new_sha))
    transition = dict(schema_version=1, parent_manifest=identity(manifest_path), manifest=identity(output/'manifest.json'),
        parent_contact_floor=identity(parent/'contact-floor.json'), contact_floor=identity(output/'contact-floor.json'),
        approved_target=previous_transition['approved_target'], robots={})
    subset_hashes = {}
    for robot in ROBOTS:
        for split in ('train', 'val'):
            rows = [r for r in added if r['robot'] == robot and r['split'] == split]
            write_json(output/robot/(split+'.yaml'), dict(subsets[robot, split], clips=subsets[robot, split]['clips']+[r['output'] for r in rows]))
        old_paths = subsets[robot, 'train']['clips']; lookup = {r['output']: r for r in manifest['clips'] if r['robot'] == robot}
        old_rows = [lookup[str((Path(subsets[robot, 'train']['root'])/p).resolve())] for p in old_paths]
        new_rows = [r for r in added if r['robot'] == robot and r['split'] == 'train']
        transition['robots'][robot] = dict(parent_subset=identity(parent/robot/'train.yaml'), subset=identity(output/robot/'train.yaml'),
            original_clips=[r['name'] for r in old_rows], appended_clips=[r['name'] for r in new_rows],
            original_clip_frames=[r['frames'] for r in old_rows], appended_clip_frames=[r['frames'] for r in new_rows],
            original_frames=sum(r['frames'] for r in old_rows), appended_frames=sum(r['frames'] for r in new_rows),
            original_clip_count=len(old_rows), appended_clip_count=len(new_rows))
        subset_hashes[robot] = sha256(output/robot/'train.yaml')
    write_json(output/'transition.json', transition)
    write_json(output/'windows.json', dict(windows, training_manifest_sha256=new_sha, subset_sha256=subset_hashes))
    write_json(output/'flip-preview.json', dict(schema_version=1, training_manifest_sha256=new_sha, subset_sha256=subset_hashes,
        fps=FPS, clips=previews, dynamics_qualified=False))
    if sha256(manifest_path) != old_sha:
        raise ValueError('Parent manifest changed during preparation')
    return transition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=REPO/'prepared/neutral-recovery-20261116')
    p.add_argument('--output', type=Path, default=REPO/'prepared/importance-flips-20261116')
    p.add_argument('--source-cache', type=Path, default=Path('/data/kk/data/blinkmorph_flip_sources_20261116'))
    p.add_argument('--retarget-repo', type=Path, default=Path('/data/kk/supermimic_vibed'))
    a = p.parse_args(); result = prepare(a.parent, a.output, a.source_cache, a.retarget_repo)
    print(json.dumps({r: {k: v for k, v in info.items() if k.startswith('appended')} for r, info in result['robots'].items()}, indent=2))


if __name__ == '__main__':
    main()
