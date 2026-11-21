"""Construct an immutable, source-group-safe joint motion/human dataset manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROBOTS = ('q1', 'bumi', 'g1')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def composite_rows(human, motion, human_root, motion_root):
    if human.get('complete') is not True or human.get('errors'):
        raise ValueError('Human manifest is incomplete or has errors')
    confirmation = motion.get('source_fps_confirmation', {})
    if (motion.get('source_fps_requires_confirmation') is not False or
        confirmation.get('by') != 'user' or confirmation.get('value') != motion.get('source_fps')):
        raise ValueError('MotionDecode fps lacks confirmed provenance')
    groups, names, peers, combined = {}, set(), {}, []
    for manifest, root, paired in ((human, human_root, True), (motion, motion_root, False)):
        for row in manifest['clips']:
            robot, name, group, split = (row[k] for k in ('robot', 'name', 'group_id', 'split'))
            if robot not in ROBOTS or Path(name).name != name or split not in ('train', 'val'):
                raise ValueError('Invalid robot, clip name, or split')
            if group in groups and groups[group] != split:
                raise ValueError(f'Source group split leakage: {group}')
            groups[group] = split
            peer_key = (row['dataset'], name)
            if peer_key in peers and peers[peer_key] != (group, split):
                raise ValueError(f'Inconsistent peer source: {peer_key}')
            peers[peer_key] = (group, split)
            if (robot, name) in names:
                raise ValueError(f'Duplicate clip identity: {robot}/{name}')
            names.add((robot, name))
            rel = Path(row['output'])
            if rel.is_absolute() or '..' in rel.parts or rel != Path(robot)/(name+'.npz'):
                raise ValueError(f'Invalid output path: {rel}')
            out = dict(row, output=str(Path(root).resolve()/rel), human_available=paired)
            if paired:
                side = Path(row['human_sidecar'])
                if side != Path('human50')/robot/(name+'.npz'):
                    raise ValueError(f'Invalid human path: {side}')
                out['human_sidecar'] = str(Path(root).resolve()/side)
            else:
                out.pop('human_sidecar', None)
            combined.append(out)
    return combined, list(motion.get('errors', []))


def check_no_motion_sidecar(human_root, row):
    side = Path(human_root)/'human50'/row['robot']/(row['name']+'.npz')
    if side.exists() or side.is_symlink():
        raise ValueError('Unexpected sidecar for motion-only clip: '+str(side))


def prepare(human_root, motion_root, output):
    import numpy as np
    human_root, motion_root, output = map(Path, (human_root, motion_root, output))
    human_path, motion_path = human_root/'manifest.json', motion_root/'manifest.json'
    human, motion = json.loads(human_path.read_text()), json.loads(motion_path.read_text())
    rows, errors = composite_rows(human, motion, human_root, motion_root)
    by_name = {}
    for row in rows:
        by_name.setdefault((row['dataset'], row['name']), set()).add(row['robot'])
    excluded = []
    selected = []
    for row in rows:
        if by_name[row['dataset'], row['name']] != set(ROBOTS):
            if row['human_available']:
                raise ValueError('Incomplete human robot triplet: '+row['name'])
            excluded.append(dict(robot=row['robot'], name=row['name'], reason='incomplete robot triplet'))
            continue
        for key, digest in [('output','output_sha256')] + ([('human_sidecar','human_sha256')] if row['human_available'] else []):
            path = Path(row[key])
            if path.is_symlink() or sha(path) != row[digest]:
                raise ValueError(f'Prepared content SHA differs: {path}')
        if row['human_available']:
            with np.load(row['human_sidecar'], allow_pickle=False) as side:
                valid = np.asarray(side['human_joint_valid'], dtype=bool)
                if valid.shape != (24,) or not valid[[0,7,8,20,21]].all():
                    raise ValueError('Invalid five-point human slots: '+row['human_sidecar'])
                row['human_joint_valid'] = valid.tolist()
        else:
            check_no_motion_sidecar(human_root, row)
            row['human_joint_valid'] = [False]*24
        selected.append(row)
    if {r['dataset'] for r in selected} != {'seed','lafan1','motiondecode'}:
        raise ValueError('All three datasets must contribute')
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = dict(schema_version=1, complete=True, clips=selected, excluded=excluded,
                    source_errors=errors, source_fps_confirmation=motion['source_fps_confirmation'],
                    source_manifests=[dict(path=str(p.resolve()),sha256=sha(p)) for p in (human_path,motion_path)],
                    human_root=str((human_root/'human50').resolve()))
    for robot in ROBOTS:
        (output/robot).mkdir()
        for split in ('train','val'):
            clips=[r['output'] for r in selected if r['robot']==robot and r['split']==split]
            if not clips: raise ValueError(f'Empty {robot}/{split}')
            # JSON is a YAML subset; absolute paths make source ownership explicit.
            (output/robot/(split+'.yaml')).write_text(json.dumps(dict(root='/', clips=clips), indent=2))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--human-root',type=Path,required=True)
    p.add_argument('--motion-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); report=prepare(a.human_root,a.motion_root,a.output)
    print('clips',len(report['clips']),'excluded',len(report['excluded']),'source errors',len(report['source_errors']))

if __name__=='__main__': main()
