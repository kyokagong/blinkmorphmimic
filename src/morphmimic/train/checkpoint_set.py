"""Committed per-robot inference files and one coordinator state for local execution."""
import hashlib
import json
import os
from pathlib import Path
import re
import torch
from morphmimic.train.optimizations import _write_checkpoint


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _sync(path):
    with Path(path).open('rb') as stream:os.fsync(stream.fileno())


def _json(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(state,sort_keys=True));_sync(tmp);os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_checkpoint_set(root,snapshot):
    """One immutable CPU snapshot, commit only after every file and digest exists."""
    root=Path(root);states=snapshot['robots'];coordinator=snapshot['coordinator']
    identity=coordinator['local_executor'];iteration=coordinator['iteration']
    if type(iteration) is not int or iteration<0:raise ValueError('Invalid checkpoint iteration')
    robots=identity['robots']
    if set(robots)!=set(states) or len(robots)!=len(set(robots)) or not robots:
        raise ValueError('Checkpoint robot inventory differs')
    if identity.get('executor')!='single_gpu_sequential' or identity.get('world_size')!=1:
        raise ValueError('Checkpoint executor identity differs')
    for robot,state in states.items():
        if not re.fullmatch('[a-z0-9_]+',robot):raise ValueError('Invalid robot path')
        if state['iteration']!=iteration or identity.get('iteration')!=iteration:
            raise ValueError('Checkpoint iteration differs')
        if state.get('local_executor')!=identity:raise ValueError('Checkpoint executor identity differs')
    commit=root/'commits'/f'iteration_{iteration}.json'
    paths=[root/r/f'{kind}_{iteration}.{ext}' for r in robots for kind,ext in (('ckpt','pt'),('policy','pt'),('coverage','json'))]+[root/f'coordinator_{iteration}.pt',commit]
    if any(p.exists() for p in paths):raise FileExistsError('Checkpoint set already exists or is incomplete; never overwrite it')
    root.mkdir(parents=True,exist_ok=True)
    files={}
    for robot,state in states.items():
        path=root/robot/f'ckpt_{iteration}.pt'
        _write_checkpoint(path,state)
        for item in (path,path.with_name(f'policy_{iteration}.pt')):
            _sync(item);files[str(item.relative_to(root))]=sha256(item)
        if 'coverage' in state:
            coverage=state['coverage']
            target=path.with_name(f'coverage_{iteration}.json')
            _json(target,{'iteration':iteration,'robot':robot,'clips':coverage['clip_names'],
                         'counts':{k:v.tolist() for k,v in coverage['counts'].items()},
                         'source':coverage.get('source')})
            files[str(target.relative_to(root))]=sha256(target)
    target=root/f'coordinator_{iteration}.pt';tmp=target.with_suffix('.pt.tmp')
    torch.save(coordinator,tmp);_sync(tmp);os.replace(tmp,target)
    files[target.name]=sha256(target)
    receipt={'schema':2,'iteration':iteration,'local_executor':identity,'files':files,
             'coverage_robots':[r for r in robots if 'coverage' in states[r]]}
    _json(commit,receipt)
    _json(root/'commits/latest.json',receipt)
    return commit


def _receipt(root,iteration=None):
    """Validate the complete canonical inventory before deserializing any tensor."""
    path=root/'commits'/('latest.json' if iteration is None else f'iteration_{iteration}.json')
    receipt=json.loads(path.read_text())
    if not isinstance(receipt,dict) or set(receipt)!={'schema','iteration','local_executor','files','coverage_robots'} or type(receipt['schema']) is not int or receipt['schema']!=2:
        raise ValueError('Invalid committed executor receipt schema; schema 2 required')
    saved_iteration=receipt['iteration']
    if type(saved_iteration) is not int or saved_iteration<0 or (iteration is not None and iteration!=saved_iteration):
        raise ValueError('Invalid committed checkpoint iteration')
    identity=receipt['local_executor']
    if (not isinstance(identity,dict) or identity.get('executor')!='single_gpu_sequential'
        or type(identity.get('world_size')) is not int or identity['world_size']!=1
        or type(identity.get('iteration')) is not int or identity['iteration']!=saved_iteration):
        raise ValueError('Invalid committed executor identity/iteration')
    robots=identity.get('robots')
    if (not isinstance(robots,list) or not robots or any(not isinstance(r,str) or re.fullmatch('[a-z0-9_]+',r) is None for r in robots)
        or len(set(robots))!=len(robots)):
        raise ValueError('Invalid committed robot inventory')
    coverage=receipt['coverage_robots']
    if (not isinstance(coverage,list) or any(not isinstance(r,str) or r not in robots for r in coverage)
        or coverage!=[r for r in robots if r in coverage]):
        raise ValueError('Invalid coverage member inventory')
    expected={f'coordinator_{saved_iteration}.pt'}
    expected.update(f'{r}/{kind}_{saved_iteration}.pt' for r in robots for kind in ('ckpt','policy'))
    expected.update(f'{r}/coverage_{saved_iteration}.json' for r in coverage)
    files=receipt['files']
    if not isinstance(files,dict) or set(files)!=expected:
        raise ValueError('Committed checkpoint member inventory differs from canonical set')
    for relative,digest in files.items():
        if not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None:
            raise ValueError('Invalid checkpoint member SHA')
        item=root/relative
        if not item.resolve().is_relative_to(root.resolve()):raise ValueError('Checkpoint member path escapes root')
        if not item.is_file() or sha256(item)!=digest:raise ValueError(f'Checkpoint hash mismatch: {relative}')
    return receipt


def read_checkpoint_set(path):
    path=Path(path)
    if path.is_dir():root=path;iteration=None
    else:
        root=path.parent.parent
        match=re.fullmatch('ckpt_(\\d+)\\.pt',path.name)
        if not match and path.name!='latest.pt':raise ValueError('Resume requires full checkpoint or run directory')
        iteration=int(match[1]) if match else None
    receipt=_receipt(root,iteration);iteration=receipt['iteration']
    if not path.is_dir() and path.parent.name not in receipt['local_executor']['robots']:
        raise ValueError('Resume path is not a committed robot member')
    coordinator=torch.load(root/f'coordinator_{iteration}.pt',map_location='cpu',weights_only=False)
    states={r:torch.load(root/r/f'ckpt_{iteration}.pt',map_location='cpu',weights_only=False)
            for r in receipt['local_executor']['robots']}
    for state in [coordinator,*states.values()]:
        if state.get('iteration')!=iteration or state.get('local_executor')!=receipt['local_executor']:
            raise ValueError('Committed checkpoint iteration/identity differs')
    if receipt['coverage_robots']!=[r for r,state in states.items() if 'coverage' in state]:
        raise ValueError('Coverage member inventory differs from full checkpoints')
    return states,coordinator


def verify_checkpoint_member(path,state):
    """Standard readers reject uncommitted local-executor files; old files unaffected."""
    if 'local_executor' not in state:
        if state.get('cfg',{}).get('executor')=='single_gpu_sequential':raise ValueError('Missing executor metadata')
        return
    path=Path(path);root=path.parent.parent;iteration=state['iteration']
    receipt=_receipt(root,iteration)
    if state['local_executor']!=receipt['local_executor']:raise ValueError('Executor identity differs')
    relative=f'{path.parent.name}/ckpt_{iteration}.pt' if path.name=='latest.pt' else str(path.relative_to(root))
    if receipt['files'].get(relative)!=sha256(path):raise ValueError('Checkpoint member hash differs')
