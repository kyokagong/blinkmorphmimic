"""Audit raw human sources separately from retargeted robot trajectories."""
from __future__ import annotations
import os
from pathlib import Path

_SKIP = {'.cache', '.git', '__pycache__', '.embodiedbein', 'node_modules',
         'runs', 'reports', 'evaluation', 'checkpoints', 'videos', 'robotwin-assets'}


def verify_prepared_pairs(root, *, robot=None, names=None, expected_sha256=None):
    """Bind the manifest identity to the actual prepared robot and human bytes."""
    import hashlib,json
    root=Path(root).resolve(strict=True)
    raw=(root/'manifest.json').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('Paired manifest SHA differs')
    rows=[r for r in json.loads(raw)['clips'] if robot is None or r['robot']==robot]
    if names is not None:
        wanted=set(names);available={r['name']+'.npz' for r in rows}
        if wanted-available:raise ValueError('Pair manifest has no requested motion')
        rows=[r for r in rows if r['name']+'.npz' in wanted]
    seen=set()
    for row in rows:
        identity=(row['robot'],row['name'])
        if identity in seen:raise ValueError('Duplicate pair identity')
        seen.add(identity)
        for field,key in (('output','output_sha256'),('human_sidecar','human_sha256')):
            relative=Path(row[field]);path=root/relative
            if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or path.resolve()!=path:
                raise ValueError('Paired file is outside its fixed root or a symlink')
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=row[key]:raise ValueError(f'Prepared pair SHA differs: {relative}')
    if not rows:raise ValueError('No prepared pairs selected')
    return digest


def verify_prepared_subset(root,subset,*,robot,split,expected_sha256=None):
    """Verify the paths the loader actually opens and their train/validation role."""
    import json
    from morphmimic.data.manifest import load_subset
    if split not in ('train','val','train_probe'):raise ValueError('Unknown paired subset role')
    root=Path(root).resolve(strict=True)
    rows=[r for r in json.loads((root/'manifest.json').read_text())['clips'] if r['robot']==robot]
    expected={r['name']+'.npz':r for r in rows};paths=[Path(p).absolute() for p in load_subset(str(subset))]
    names=[p.name for p in paths];role='train' if split=='train_probe' else split
    if not paths or len(set(names))!=len(names):raise ValueError('Empty or duplicate paired subset')
    for path,name in zip(paths,names):
        row=expected.get(name)
        if row is None or row['split']!=role or path.resolve()!=root/row['output'] or path!=path.resolve():
            raise ValueError('Paired subset path or train/validation role differs from its manifest')
    if split!='train_probe' and set(names)!={name for name,row in expected.items() if row['split']==role}:
        raise ValueError('Paired subset omits declared train/validation clips')
    return verify_prepared_pairs(root,robot=robot,names=names,expected_sha256=expected_sha256)


def inventory_sources(data_root, max_examples=20):
    root = Path(data_root).resolve(strict=True)
    result = dict(schema_version=1, root=str(root), human_candidates=[], archives=[],
                  robot_csv={}, motiondecode_selected=[], skipped_symlinks=[], errors=[])
    if max_examples < 0:
        raise ValueError('max_examples must be nonnegative')
    def error(exc):
        result['errors'].append(dict(path=exc.filename, error=str(exc)))
    for base, dirs, files in os.walk(root, onerror=error, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not (Path(base)/d).is_symlink())
        for name in sorted(files):
            path = Path(base)/name
            if path.is_symlink():
                result['skipped_symlinks'].append(str(path)); continue
            relative = path.relative_to(root)
            lower = str(relative).lower()
            suffix = path.suffix.lower()
            if suffix in ('.bvh', '.fbx'):
                result['human_candidates'].append(dict(path=str(path), format=suffix[1:],
                    dataset='seed' if 'soma' in lower or 'seed' in lower else
                    'lafan1' if 'lafan' in lower else 'motiondecode' if 'motiondecode' in lower else 'unknown'))
            elif suffix in ('.zip', '.gz', '.xz', '.7z') and any(x in name.lower() for x in ('soma','lafan','motiondecode')):
                result['archives'].append(dict(path=str(path), bytes=path.stat().st_size))
            elif suffix == '.csv':
                dataset = ('motiondecode' if 'motiondecode' in lower else
                           'lafan1' if 'lafan' in lower else 'seed' if 'seed' in lower else None)
                if dataset is None: continue
                group = result['robot_csv'].setdefault(dataset, dict(count=0, examples=[]))
                group['count'] += 1
                if len(group['examples']) < max_examples: group['examples'].append(str(path))
                is_dance = any(p.lower().startswith('5.dance') for p in relative.parts)
                if dataset == 'motiondecode' and ('baji' in name.lower() or is_dance):
                    result['motiondecode_selected'].append(dict(path=str(path),name=path.stem,
                        dataset=dataset, category='baji' if 'baji' in name.lower() else 'dance',
                        human_source=None, fps=None))
    return result


def seed_index(metadata_csv, wanted):
    import csv
    from pathlib import PurePosixPath
    wanted=set(wanted);result={}
    with Path(metadata_csv).open(newline='') as stream:
        for row in csv.DictReader(stream):
            name=row['filename']
            if name not in wanted:continue
            if name in result:raise ValueError(f'duplicate SEED identity: {name}')
            path=PurePosixPath(row['move_soma_uniform_path'])
            if path.is_absolute() or '..' in path.parts or path.name!=name+'.bvh':
                raise ValueError(f'Unsafe or mismatching human path for {name}')
            mirror=row['is_mirror'].lower()
            if mirror not in ('true','false'):raise ValueError(f'Invalid mirror flag: {name}')
            frames=int(row['move_duration_frames'])
            if frames<2:raise ValueError(f'Source too short: {name}')
            result[name]=dict(path=str(path),is_mirror=mirror=='true',source_frames=frames)
    missing=wanted-result.keys()
    if missing:raise ValueError(f'{len(missing)} SEED identities missing from metadata: {sorted(missing)[:3]}')
    return result


def validate_human(human):
    import numpy as np
    from morphmimic.constants import HUMAN_JOINTS
    p,q=human['human_pos_w'],human['human_quat_w'];n=len(p)
    if n<2 or p.shape!=(n,24,3) or q.shape!=(n,24,4):raise ValueError('Invalid human pose shapes')
    if not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('Nonfinite human pose')
    if not np.allclose(np.linalg.norm(q,axis=-1),1,atol=1e-4,rtol=0):raise ValueError('Invalid human quaternion norms')
    if int(human['fps'])!=50 or tuple(human['human_joint_names'])!=HUMAN_JOINTS:raise ValueError('Invalid human fps or joint order')


def five_points(human):
    import numpy as np
    from morphmimic.constants import HUMAN_FIVE_IDX
    validate_human(human)
    valid=human.get('human_joint_valid',np.ones(24,dtype=bool))
    if np.shape(valid)!=(24,) or not np.asarray(valid)[list(HUMAN_FIVE_IDX)].all():raise ValueError('Human five-point joints are unavailable')
    return human['human_pos_w'][:,HUMAN_FIVE_IDX,:],human['human_quat_w'][:,HUMAN_FIVE_IDX,:]


def align_pair(robot, human, *, start, stop, frames):
    import numpy as np
    n=len(robot['joint_pos']);hn=len(human['human_pos_w'])
    if not 0<=start<stop<=n or frames<2 or frames>min(stop-start,hn):raise ValueError('Invalid pair time intersection')
    frame_keys={'joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w',
                'root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','z_min','foot_dist'}
    result={}
    for k,v in robot.items():
        if k in frame_keys:
            if len(v)!=n:raise ValueError(f'Inconsistent robot frame field {k}')
            result[k]=v[start:start+frames].copy()
        else:result[k]=v.copy()
    h={k:(v[:frames].copy() if k in ('human_pos_w','human_quat_w','source_time_s') else v.copy()) for k,v in human.items()}
    record=dict(robot_start_frame=start,robot_stop_frame=start+frames,robot_tail_frames=stop-start-frames,
                human_tail_frames=hn-frames,time_s=np.arange(frames,dtype=np.float64)/50)
    return result,h,record


def _atomic_npz(path, arrays, *, resume=False):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('Refuse symbolic output')
    if path.exists() and resume:
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays) or any(not np.array_equal(saved[k],v) for k,v in arrays.items()):
                raise ValueError(f'Existing pair differs from its source: {path}')
        return
    temporary=path.with_suffix('.npz.tmp')
    created=False
    try:
        with temporary.open('xb') as stream:
            created=True
            np.savez(stream,**arrays)
        os.link(temporary,path)  # Atomic publication without replacing an existing file.
    finally:
        if created:temporary.unlink()


def write_pairs(output_root, records, robots, human, *, source, resume=False):
    import hashlib,math,numpy as np
    root=Path(output_root).resolve();by_robot={r['robot']:r for r in records}
    if set(by_robot)!={'q1','bumi','g1'} or len(records)!=3:raise ValueError('Three unique robot references required')
    names={r['name'] for r in records}
    if len(names)!=1:raise ValueError('Pair names differ')
    name=next(iter(names))
    if Path(name).name!=name or name in ('.','..'):raise ValueError('Unsafe clip name')
    targets={r:(root/r/(name+'.npz'),root/'human50'/r/(name+'.npz')) for r in by_robot}
    for pair in targets.values():
        for p in pair:
            if p.is_symlink() or (p.exists() and not resume):raise FileExistsError(p)
            if not p.parent.resolve().is_relative_to(root):raise ValueError('Output escapes data root')
    validate_human(human);five_points(human)
    windows={}
    for r,record in by_robot.items():
        n=len(robots[r]['joint_pos']);phases=record.get('phases',{})
        if int(robots[r]['fps'])!=50 or n!=record['frames']:raise ValueError('Robot frame metadata differs')
        start,stop=0,n
        if 'entry' in phases and 'dance' in phases:
            a,b=float(phases['entry']),float(phases['dance'])
            if not math.isfinite(a+b) or not 0<=a<=b:raise ValueError('Invalid source-body phases')
            start,stop=min(n,math.ceil(a*50-1e-9)),min(n,math.ceil(b*50-1e-9))
        windows[r]=(start,stop)
    frames=min(len(human['human_pos_w']),*(b-a for a,b in windows.values()))
    result=[]
    for r,record in by_robot.items():
        a,b=windows[r];robot,h,alignment=align_pair(robots[r],human,start=a,stop=b,frames=frames)
        h['source_time_s']=alignment.pop('time_s');h['human_pos_w']=h['human_pos_w'].astype(np.float32);h['human_quat_w']=h['human_quat_w'].astype(np.float32)
        output,sidecar=targets[r];_atomic_npz(output,robot,resume=resume);_atomic_npz(sidecar,h,resume=resume)
        def digest(p):
            with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
        updated={**record,'output':str(output),'output_sha256':digest(output),'human_sidecar':str(sidecar),
                 'human_sha256':digest(sidecar),'pair_source':source,'frames':frames,'phases':{},
                 'fps':50,'parent_input_fps':record.get('fps'),'alignment_eligible':True,
                 'duration_seconds':(frames-1)/50,'source_duration_seconds':(frames-1)/50,
                 'source_body_window_original':[a,b],'pair_alignment':alignment,
                 'human_joint_valid':h.get('human_joint_valid',np.ones(24,dtype=bool)).tolist()}
        result.append(updated)
    return result
