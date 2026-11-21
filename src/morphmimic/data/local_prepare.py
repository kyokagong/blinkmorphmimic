"""Local BONES-SEED/LAFAN1 conversion to the official native-joint motion schema."""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

FPS = 50


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _validate(pos, quat, joints, fps):
    if fps not in (30, 50, 120):
        raise ValueError(f'unknown fps: {fps!r}; expected 30, 50 or 120')
    n = len(pos)
    if n < 2 or pos.shape != (n, 3) or quat.shape != (n, 4) or joints.ndim != 2 or len(joints) != n:
        raise ValueError('motion requires at least two frames with matching position/quaternion/joint shapes')
    if not all(np.isfinite(a).all() for a in (pos, quat, joints)):
        raise ValueError('motion contains nonfinite values')
    norms = np.linalg.norm(quat, axis=1)
    if not np.allclose(norms, 1, atol=1e-3, rtol=0):
        raise ValueError('root quaternions must be nonzero unit quaternions')


def read_native_csv(path, joint_names, fps, source_joint_names=None):
    """Read metre/radian native CSV: xyz, xyzw quaternion, native joint columns."""
    source_joint_names = tuple(joint_names if source_joint_names is None else source_joint_names)
    if len(set(joint_names)) != len(joint_names) or len(set(source_joint_names)) != len(source_joint_names):
        raise ValueError('duplicate joint names')
    if set(source_joint_names) != set(joint_names):
        raise ValueError('source joint names do not match the robot')
    rows = np.loadtxt(path, delimiter=',', ndmin=2)
    if rows.shape[1] != 7 + len(source_joint_names):
        raise ValueError(f'{path}: wrong column count {rows.shape[1]}')
    pos, quat = rows[:, :3], rows[:, [6, 3, 4, 5]]
    joints = rows[:, [7 + source_joint_names.index(n) for n in joint_names]]
    _validate(pos, quat, joints, fps)
    return pos, quat / np.linalg.norm(quat, axis=1, keepdims=True), joints


def resample(pos, quat, joints, in_fps):
    """Keep every 50 Hz time step within the full input time span; shortest-arc SLERP."""
    _validate(pos, quat, joints, in_fps)
    count = int(np.floor((len(pos) - 1) * FPS / in_fps + 1e-9)) + 1
    if count < 2:
        raise ValueError('motion is shorter than one 50 Hz transition')
    x = np.minimum(np.arange(count) * in_fps / FPS, len(pos) - 1)
    lo = np.minimum(x.astype(int), len(pos) - 2)
    hi = lo + 1
    w = (x - lo)[:, None]
    a, b = quat[lo], quat[hi]
    dot = (a * b).sum(axis=1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    # Near-identical rotations use normalized lerp, avoiding division by zero.
    sine = np.sin(angle)
    safe = np.where(sine < 1e-8, 1, sine)
    q = np.where(sine < 1e-8, (1 - w) * a + w * b,
                 np.sin((1 - w) * angle) / safe * a + np.sin(w * angle) / safe * b)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return pos[lo] * (1 - w) + pos[hi] * w, q, joints[lo] * (1 - w) + joints[hi] * w


def source_group(dataset: str, name: str) -> str:
    """One split for every mirror, retarget, performer/subject of a source sequence."""
    name = name.removesuffix('_M')
    name = re.sub(r'__A\d+$', '', name) if dataset == 'seed' else re.sub(r'_subject\d+$', '', name)
    return f'{dataset}/{name}'


def partition(group: str) -> str:
    return 'val' if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'


@cache
def _robot_model(robot):
    from morphmimic.robots import get_robot
    spec = get_robot(robot)
    model = spec.entity_cfg_fn().spec_fn().compile()
    joints = tuple(model.joint(i).name for i in range(1, model.njnt))
    if model.nq != 7 + len(spec.joint_names) or joints != spec.joint_names:
        raise ValueError(f'{robot}: model free-joint/native-joint layout mismatch')
    if len(spec.tracked_bodies) != 14 or spec.root_body not in spec.tracked_bodies:
        raise ValueError(f'{robot}: expected fourteen tracked bodies including the root')
    body_ids = [model.body(n).id for n in spec.tracked_bodies]
    return spec, model, body_ids


def convert_motion(source, output, robot, in_fps, source_joint_names=None, seed=False):
    """FK and derivatives use the same MuJoCo asset and conventions as training."""
    import mujoco
    from scipy.spatial.transform import Rotation
    spec, model, body_ids = _robot_model(robot)
    if seed:
        # Check before pandas can silently rename duplicate CSV columns.
        with Path(source).open(newline='') as f:
            columns = next(csv.reader(f))
        if len(set(columns)) != len(columns):
            raise ValueError(f'{source}: duplicate columns')
        from morphmimic.data.seed_convert import parse_csv
        pos, quat, joints = parse_csv(str(source), spec)
        _validate(pos, quat, joints, in_fps)
    else:
        pos, quat, joints = read_native_csv(source, spec.joint_names, in_fps, source_joint_names)
    source_frames = len(pos)
    pos, quat, joints = resample(pos, quat, joints, in_fps)
    data = mujoco.MjData(model)
    body_pos = np.empty((len(pos), 14, 3))
    body_quat = np.empty((len(pos), 14, 4))
    for i in range(len(pos)):
        data.qpos[:] = np.r_[pos[i], quat[i], joints[i]]
        mujoco.mj_kinematics(model, data)
        body_pos[i] = data.xpos[body_ids]
        body_quat[i] = data.xquat[body_ids]
    body_quat = np.where(body_quat[..., :1] < 0, -body_quat, body_quat)
    before = np.r_[0, np.arange(len(pos) - 2), len(pos) - 2]
    after = np.r_[1, np.arange(2, len(pos)), len(pos) - 1]
    a = Rotation.from_quat(body_quat[before][..., [1, 2, 3, 0]].reshape(-1, 4))
    b = Rotation.from_quat(body_quat[after][..., [1, 2, 3, 0]].reshape(-1, 4))
    angular = (b * a.inv()).as_rotvec().reshape(len(pos), 14, 3) * (FPS / (after - before))[:, None, None]
    arrays = dict(fps=np.array(FPS), joint_pos=joints, joint_vel=np.gradient(joints, 1 / FPS, axis=0),
                  body_pos_w=body_pos, body_quat_w=body_quat,
                  body_lin_vel_w=np.gradient(body_pos, 1 / FPS, axis=0), body_ang_vel_w=angular,
                  body_names=np.array(spec.tracked_bodies), joint_names=np.array(spec.joint_names),
                  robot=np.array(robot), root_body=np.array(spec.root_body))
    arrays = {k: a.astype(np.float32) if a.dtype.kind == 'f' else a for k, a in arrays.items()}
    if not all(np.isfinite(a).all() for a in arrays.values() if a.dtype.kind == 'f'):
        raise ValueError(f'{source}: nonfinite FK/velocities')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.npz.tmp')
    with temporary.open('wb') as f:
        np.savez(f, **arrays)
    temporary.replace(output)
    return {'frames': len(pos), 'source_frames': source_frames,
            'source_duration_seconds': (source_frames - 1) / in_fps,
            'duration_seconds': (len(pos) - 1) / FPS}


def discover(data_root: Path):
    """Use upstream manifests to exclude synthetic commands and document all gaps."""
    seed_root = data_root / 'supermimic_bumi/seed_dances'
    qroot = data_root / 'blinkmimic_q1/20261107'
    qm = json.loads((qroot / 'manifest.json').read_text())
    bm = json.loads((seed_root / 'motions_v1/manifest.json').read_text())
    native = json.loads((seed_root / 'native_manifest.json').read_text())
    lafan = sorted((data_root / 'supermimic_g1/lafan1_g1').glob('*.csv'))
    entries = []
    retarget_repo = Path(os.environ.get('BLINKMORPH_RETARGET_REPO', '/data/kk/supermimic_vibed'))
    for name, meta in sorted(qm['clips'].items()):
        dataset = meta.get('group')
        if dataset not in ('seed', 'lafan1'):
            continue
        original = seed_root / 'native' / f'{name}.csv' if dataset == 'seed' else data_root / 'supermimic_g1/lafan1_g1' / f'{name}.csv'
        base = {'name': name, 'dataset': dataset, 'source_id': f'{dataset}/{name.removesuffix("_M")}',
                'group_id': source_group(dataset, name), 'original_source': str(original),
                'original_sha256': native[name]['sha256'] if dataset == 'seed' else sha256(original)}
        entries.append(dict(base, robot='q1', input=str(qroot / 'motions' / f'{name}.csv'),
                            fps=qm['fps'], joint_names=qm['joint_names'], expected_sha256=meta['sha256'],
                            phases=meta.get('phases', {}), upstream_record=meta))
        if dataset == 'seed':
            entries.append(dict(base, robot='g1', input=str(original), fps=120, seed=True,
                                expected_sha256=native[name]['sha256']))
        else:
            entries.append(dict(base, robot='g1', input=str(original), fps=30))
        if name in bm['clips']:
            entries.append(dict(base, robot='bumi', input=str(seed_root / 'motions_v1' / f'{name}.csv'),
                                fps=bm['fps'], joint_names=bm['joint_names'], expected_sha256=bm['clips'][name]['sha256']))
        elif dataset == 'lafan1':
            entries.append(dict(base, robot='bumi', input=str(original), fps=30,
                                joint_names=bm['joint_names'], retarget=True, retarget_repo=str(retarget_repo)))
    missing = [p.stem for p in lafan if p.stem not in bm['clips']]
    evidence_paths = [qroot / 'manifest.json', seed_root / 'motions_v1/manifest.json', seed_root / 'native_manifest.json', seed_root / 'selection.json']
    evidence = [{'path': str(p), 'sha256': sha256(p)} for p in evidence_paths]
    return entries, missing, evidence


def _adapt_retarget_source(repo):
    path = Path(repo) / 'tools/retarget_bumi.py'
    original = path.read_text()
    old = 'from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list'
    new = 'from morphmimic.data.retarget_compat import G1_JOINT_NAMES, default_joint_pos_list'
    if original.count(old) != 1:
        raise ValueError(f'{path}: expected exactly one data-only metadata import to adapt')
    adapted = original.replace(old, new)
    return adapted, {'original_source': str(path), 'original_sha256': sha256(path),
                     'adapted_source_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
                     'import_replacements': 1, 'original_import': old, 'adapted_import': new,
                     'compat_module_sha256': sha256(Path(__file__).with_name('retarget_compat.py'))}


@cache
def _retarget_function(repo):
    import sys
    from types import ModuleType
    sys.path.insert(0, repo)
    source, adaptation = _adapt_retarget_source(repo)
    path = Path(repo) / 'tools/retarget_bumi.py'
    module = ModuleType('blinkmorph_bumi_retarget')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    module.retarget.adaptation = adaptation
    return module.retarget


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _load_receipt(output, input_hash, asset_hash):
    path = output.with_suffix('.receipt.json')
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record['input_sha256'] != input_hash or record['asset_sha256'] != asset_hash:
        raise ValueError(f'{output}: resume source/asset fingerprint mismatch')
    if record['output_sha256'] != sha256(output):
        raise ValueError(f'{output}: resume output fingerprint mismatch')
    return record


def _verify_legacy_output(entry, output):
    """Adopt pre-receipt files only after source arrays and the old manifest's asset match."""
    import mujoco
    spec, model, ids = _robot_model(entry['robot'])
    if entry.get('retarget'):
        raise ValueError(f'{output}: retarget output without provenance receipt cannot be resumed')
    if entry.get('seed'):
        from morphmimic.data.seed_convert import parse_csv
        raw = parse_csv(entry['input'], spec)
    else:
        raw = read_native_csv(entry['input'], spec.joint_names, entry['fps'], entry.get('joint_names'))
    source_frames = len(raw[0])
    pos, quat, joints = resample(*raw, entry['fps'])
    frames = len(pos)
    with np.load(output, allow_pickle=False) as z:
        shapes = {'joint_pos': (frames, len(spec.joint_names)), 'joint_vel': (frames, len(spec.joint_names)),
                  'body_pos_w': (frames, 14, 3), 'body_quat_w': (frames, 14, 4),
                  'body_lin_vel_w': (frames, 14, 3), 'body_ang_vel_w': (frames, 14, 3)}
        if int(z['fps']) != FPS or tuple(z['joint_names']) != spec.joint_names or tuple(z['body_names']) != spec.tracked_bodies:
            raise ValueError(f'{output}: legacy schema does not match current robot')
        for key, shape in shapes.items():
            if z[key].shape != shape or not np.isfinite(z[key]).all():
                raise ValueError(f'{output}: invalid legacy {key}')
        root = spec.tracked_bodies.index(spec.root_body)
        checks = [(z['joint_pos'], joints), (z['joint_vel'], np.gradient(joints, 1 / FPS, axis=0)),
                  (z['body_pos_w'][:, root], pos),
                  (z['body_lin_vel_w'], np.gradient(z['body_pos_w'].astype(np.float64), 1 / FPS, axis=0))]
        for actual, wanted in checks:
            if not np.allclose(actual, wanted, atol=1e-4, rtol=1e-4):
                raise ValueError(f'{output}: legacy output does not match its full source sequence')
        root_dot = np.abs((z['body_quat_w'][:, root] * quat).sum(axis=1))
        if not np.allclose(root_dot, 1, atol=1e-5, rtol=0):
            raise ValueError(f'{output}: legacy root orientation differs from source')
        # Existing run's matching compiled asset hash covers the model; verify FK anchors too.
        data = mujoco.MjData(model)
        for i in sorted({0, frames // 2, frames - 1}):
            data.qpos[:] = np.r_[pos[i], quat[i], joints[i]]
            mujoco.mj_kinematics(model, data)
            dot = np.abs((z['body_quat_w'][i] * data.xquat[ids]).sum(axis=1))
            if not np.allclose(z['body_pos_w'][i], data.xpos[ids], atol=1e-5) or not np.allclose(dot, 1, atol=1e-5):
                raise ValueError(f'{output}: legacy FK differs from current asset')
    return {'frames': frames, 'source_frames': source_frames,
            'source_duration_seconds': (source_frames - 1) / entry['fps'],
            'duration_seconds': (frames - 1) / FPS,
            'resume_verification': 'all source joints/root frames; matching prior compiled asset; first/middle/last FK; full schema/finite checks'}


def _convert_job(job):
    entry, output, asset_hash = job
    digest = sha256(Path(entry['input']))
    if entry.get('expected_sha256', digest) != digest:
        raise ValueError(f'{entry["input"]}: upstream sha256 mismatch')
    if output.exists():
        record = _load_receipt(output, digest, asset_hash)
        if record is not None:
            if entry.get('retarget'):
                current = _retarget_function(entry['retarget_repo']).adaptation
                if record['retarget_report'].get('adaptation') != current:
                    raise ValueError(f'{output}: retarget adaptation fingerprint mismatch')
            return dict(record, resumed=True)
        result = _verify_legacy_output(entry, output)
        repair = {}
    else:
        source = entry['input']
        repair = {}
        if entry.get('retarget'):
            generated = output.parents[1] / 'generated_sources/bumi' / f'{entry["name"]}.csv'
            generated.parent.mkdir(parents=True, exist_ok=True)
            fn = _retarget_function(entry['retarget_repo'])
            repair = fn(Path(source), generated)
            repair.update(generated_source=str(generated), generated_source_sha256=sha256(generated), adaptation=fn.adaptation)
            source = str(generated)
        result = convert_motion(source, output, entry['robot'], entry['fps'], entry.get('joint_names'), entry.get('seed', False))
    record = dict(entry, input_sha256=digest, asset_sha256=asset_hash, output=str(output), output_sha256=sha256(output),
                  split=partition(entry['group_id']), alignment_eligible=False, retarget_report=repair, **result)
    _write_json(output.with_suffix('.receipt.json'), record)
    return record


def prepare(output, data_root=None, limit=None, workers=1, resume=False):
    import yaml
    output = Path(output).resolve()
    data_root = Path(data_root or os.environ.get('BLINKMORPH_DATA_ROOT', '/data/kk/data')).resolve()
    previous = None
    if output.exists():
        if not resume:
            raise FileExistsError(f'use a new output directory or --resume: {output}')
        previous = json.loads((output / 'manifest.json').read_text())
    elif resume:
        raise FileNotFoundError(f'no preparation to resume: {output}')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if workers < 1:
        raise ValueError('workers must be positive')
    entries, missing, evidence = discover(data_root)
    inventory = {r: {d: sum(e['robot'] == r and e['dataset'] == d for e in entries) for d in ('seed', 'lafan1')} for r in ('q1', 'bumi', 'g1')}
    # A smoke subset includes both datasets for every robot; the limit is per robot/dataset.
    if limit is not None:
        counts = {}
        chosen = []
        for e in entries:
            key = (e['robot'], e['dataset'])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= limit:
                chosen.append(e)
        entries = chosen
    retarget_entries = [e for e in entries if e.get('retarget')]
    retarget_evidence = []
    if retarget_entries:
        repo = Path(retarget_entries[0]['retarget_repo'])
        for relative in ('tools/retarget_bumi.py', 'supermimic/bumi.py', 'supermimic/bumi_ik.py', 'supermimic/robot.py'):
            p = repo / relative
            retarget_evidence.append({'path': str(p), 'sha256': sha256(p)})
    output.mkdir(parents=True, exist_ok=resume)
    manifest = {'schema_version': 1, 'fps': FPS, 'human_data': False, 'alignment_enabled': False,
                'alignment_reason': 'Q1 includes entry/return phases; no common source time mapping has been verified.',
                'inventory': inventory, 'initially_missing_bumi_lafan1': missing, 'retarget_code': retarget_evidence, 'input_manifests': evidence,
                'split_rule': 'sha256(dataset/source_sequence) modulo 10; 0=val, rest=train; mirrors and performers grouped',
                'limit_per_robot_dataset': limit, 'diagnostic_subset': limit is not None, 'complete': False, 'assets': {}, 'clips': []}
    for robot in ('q1', 'bumi', 'g1'):
        spec, model, _ = _robot_model(robot)
        import mujoco
        # Serialized compiled model fingerprints include every loaded mesh and physical parameter.
        buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=buffer)
        manifest['assets'][robot] = {'compiled_mjb_sha256': hashlib.sha256(buffer.tobytes()).hexdigest(),
                                     'joint_names': list(spec.joint_names), 'tracked_bodies': list(spec.tracked_bodies),
                                     'mujoco_version': mujoco.__version__}
    if previous is not None:
        for field in ('assets', 'input_manifests', 'limit_per_robot_dataset'):
            if previous[field] != manifest[field]:
                raise ValueError(f'cannot resume: {field} differs from existing manifest')
        manifest['resumed_from_incomplete'] = not previous.get('complete', False)
    if retarget_entries:
        repo = retarget_entries[0]['retarget_repo']
        adapted, adaptation = _adapt_retarget_source(repo)
        generated = output / 'generated_sources'
        generated.mkdir(exist_ok=True)
        (generated / 'retarget_bumi_adapted.py').write_text(adapted)
        manifest['retarget_adaptation'] = adaptation
    jobs = [(e, output / e['robot'] / f'{e["name"]}.npz', manifest['assets'][e['robot']]['compiled_mjb_sha256']) for e in entries]
    manifest_path = output / 'manifest.json'
    _write_json(manifest_path, manifest)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, clip in enumerate(pool.map(_convert_job, jobs, chunksize=8), 1):
                manifest['clips'].append(clip)
                if i % 100 == 0:
                    print(f'prepared {i}/{len(jobs)}', flush=True)
    finally:
        _write_json(manifest_path, manifest)
    for robot in ('q1', 'bumi', 'g1'):
        for subset in ('train', 'val', 'all'):
            clips = [c for c in manifest['clips'] if c['robot'] == robot and (subset == 'all' or c['split'] == subset)]
            path = output / robot / f'{subset}.yaml'
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump({'root': '.',
                                           'hours': sum(c['frames'] / FPS for c in clips) / 3600,
                                           'clips': [f'{c["name"]}.npz' for c in clips]}, sort_keys=False))
    manifest['retargeted_bumi_lafan1'] = [c['name'] for c in manifest['clips'] if c.get('retarget')]
    manifest['missing_bumi_lafan1'] = sorted(set(missing) - set(manifest['retargeted_bumi_lafan1']))
    manifest['coverage_complete'] = limit is None and not manifest['missing_bumi_lafan1']
    manifest['complete'] = True
    _write_json(manifest_path, manifest)
    return manifest
