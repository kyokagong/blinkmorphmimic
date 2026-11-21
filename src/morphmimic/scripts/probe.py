"""Budgeted, training-only motion probes; emits optional robot-local reset focus.

Example: python -m morphmimic.scripts.probe --ckpt runs/run/q1/ckpt_1000.pt
  --robot q1 --subset-yaml prepared/full-v1/q1/train.yaml --output probes/q1.json
No validation/all subset is accepted. Every selected clip is evaluated from its
first frame to its end or failure; complete reference durations bound the budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_identity(path, robot):
    import torch
    path = Path(path)
    if path.is_symlink() or re.fullmatch(r'ckpt_\d+\.pt', path.name) is None:
        raise ValueError('probe requires a fixed ckpt_<iteration>.pt file, not latest or a symlink')
    digest = file_hash(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if file_hash(path) != digest:
        raise ValueError('probe checkpoint changed while loading; refusing to publish focus')
    cfg = checkpoint['cfg']
    if cfg.get('robot', 'g1') != robot:
        raise ValueError('probe robot does not match checkpoint robot')
    return {'checkpoint': str(path.resolve()), 'checkpoint_sha256': digest,
            'goal_anchor': cfg.get('goal_anchor', 'reference')}


def training_clips(subset_yaml, robot):
    """Require train.yaml plus prepared-manifest proof for every requested clip."""
    from morphmimic.data.manifest import load_subset
    subset_yaml = Path(subset_yaml).resolve()
    if subset_yaml.stem != 'train' or robot not in ('q1', 'bumi', 'g1'):
        raise ValueError('probe accepts a prepared train.yaml only, with a known robot')
    manifest_path = subset_yaml.parent.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('complete'):
        raise ValueError('probe requires a complete prepared manifest')
    lookup = {c['name']: c for c in manifest['clips'] if c['robot'] == robot}
    paths = load_subset(str(subset_yaml))
    if not paths or len({Path(p).stem for p in paths}) != len(paths):
        raise ValueError('probe training subset is empty or contains duplicate clips')
    clips = []
    for text in paths:
        path = Path(text).resolve()
        clip = lookup.get(path.stem)
        if (clip is None or clip.get('split') != 'train' or not clip.get('source_id')
                or path.parent != (manifest_path.parent / robot).resolve() or not path.is_file()):
            raise ValueError(f'probe clip has no matching training-only source proof: {path}')
        clips.append(dict(clip, path=str(path)))
    return clips, {'training_subset': str(subset_yaml), 'training_subset_sha256': file_hash(subset_yaml),
                   'prepared_manifest_sha256': file_hash(manifest_path)}


def validate_probe_clips(clips):
    """A stale manifest must not silently enlarge the reference-frame budget."""
    import numpy as np
    for clip in clips:
        with np.load(clip['path'], allow_pickle=False) as motion:
            if int(motion['fps']) != 50 or len(motion['joint_pos']) != clip['frames']:
                raise ValueError(f'probe motion fps/length differs from prepared metadata: {clip["path"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--robot', choices=('q1', 'bumi', 'g1'), required=True)
    parser.add_argument('--subset-yaml', '--training-subset', dest='training_subset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Focus JSON, normally <probe-dir>/<robot>.json')
    parser.add_argument('--max-clips', type=int, default=256)
    parser.add_argument('--max-frames', type=int, default=100000, help='Sum of complete-clip reference transitions')
    parser.add_argument('--num-envs', type=int, default=64, help='Fixed parallel evaluation capacity, at most 256')
    parser.add_argument('--seed', type=int, default=0, help='Deterministic training-clip rotation')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 256:
        parser.error('--num-envs must be in 1..256')
    from morphmimic.scripts.gpu_guard import select_gpu
    select_gpu(args.gpu)
    from mjlab.utils.torch import configure_torch_backends
    from morphmimic.eval.benchmark import evaluate
    from morphmimic.train.probe_focus import LOOKBACK_SECONDS, SCHEMA_VERSION, select_probe_clips
    configure_torch_backends()
    started = time.monotonic()
    checkpoint = checkpoint_identity(args.ckpt, args.robot)
    clips, proof = training_clips(args.training_subset, args.robot)
    selected = select_probe_clips(clips, args.max_clips, args.max_frames, args.seed)
    validate_probe_clips(selected)
    lookup = {c['name']: c for c in selected}
    import yaml
    with tempfile.TemporaryDirectory(prefix='morphmimic-probe-') as temporary:
        subset = Path(temporary) / 'probe.yaml'
        subset.write_text(yaml.safe_dump({'root': '/', 'clips': [c['path'] for c in selected]}))
        eval_started = time.monotonic()
        results = evaluate(args.ckpt, str(subset), 'cuda:0', robot=args.robot,
                           num_envs=min(args.num_envs, len(selected)), goal_anchor=checkpoint['goal_anchor'])
        evaluation_seconds = time.monotonic() - eval_started
    if file_hash(args.ckpt) != checkpoint['checkpoint_sha256'] or file_hash(args.training_subset) != proof['training_subset_sha256']:
        raise ValueError('checkpoint or training subset changed during probe; refusing to publish focus')
    failures = []
    for row in results.to_dict('records'):
        clip = lookup[row['name']]
        if row['failed']:
            frame = int(row['fail_frame'])
            if not 0 <= frame < clip['frames']:
                raise ValueError(f'benchmark returned invalid failure frame: {row}')
            failures.append({'robot': args.robot, 'source_id': clip['source_id'],
                             'dataset': clip['dataset'], 'clip': clip['name'], 'fail_frame': frame})
    total_seconds = time.monotonic() - started
    cost = {'total_seconds': total_seconds, 'evaluation_seconds': evaluation_seconds,
            'gpu_hours': total_seconds / 3600, 'max_clips': args.max_clips, 'max_frames': args.max_frames,
            'selected_clips': len(selected), 'planned_reference_steps': sum(c['frames'] - 1 for c in selected),
            'actual_active_steps': int(results['steps'].sum()), 'num_envs': min(args.num_envs, len(selected))}
    document = {'schema_version': SCHEMA_VERSION, 'robot': args.robot, 'interface': 'motion', 'fps': 50,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'lookback_seconds': LOOKBACK_SECONDS,
                'seed': args.seed, **checkpoint, **proof, 'cost': cost, 'failures': failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(document, indent=2))
    temporary.replace(args.output)
    results.to_csv(args.output.with_suffix('.csv'), index=False)
    print(json.dumps({'focus_file': str(args.output), 'failures': len(failures), 'cost': cost}, indent=2))


if __name__ == '__main__':
    main()
