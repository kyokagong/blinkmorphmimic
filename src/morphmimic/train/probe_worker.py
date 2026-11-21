"""Train-only probe identities, bounded work and weighted failure records."""
from __future__ import annotations
import math
from pathlib import Path
import re

INTERFACES=('motion','human-five','human-full')


def budget_remaining(training_seconds,used_seconds,fraction=.05):
    if any(not math.isfinite(x) for x in (training_seconds,used_seconds,fraction)) or training_seconds<0 or used_seconds<0 or not 0<fraction<=.05:
        raise ValueError('Invalid training time, used time, or probe budget')
    return max(0.,training_seconds*fraction-used_seconds)


def aggregate_failures(records,weights):
    if set(records)!=set(INTERFACES) or set(weights)!=set(INTERFACES):raise ValueError('Unknown probe interface')
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in weights.values()) or not math.isfinite(sum(weights.values())) or sum(weights.values())<=0:
        raise ValueError('Invalid probe interface weights')
    return [dict(row,interface=interface,weight=weights[interface]) for interface in INTERFACES for row in records[interface]]


def latest_completed(directory):
    candidates=[]
    for path in Path(directory).glob('ckpt_*.pt'):
        match=re.fullmatch(r'ckpt_(\d+)\.pt',path.name)
        if match and path.is_file() and not path.is_symlink():candidates.append((int(match.group(1)),path))
    return max(candidates)[1] if candidates else None


def validate_clip_ids(ids,num_clips):
    ids=list(ids)
    if not ids or any(type(i) is not int or i<0 or i>=num_clips for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Clip ids must be unique, nonempty and within library')
    return ids


def validate_training_rows(rows,robot,paths):
    paths=[str(Path(p).resolve()) for p in paths]
    if not paths or len(set(paths))!=len(paths):raise ValueError('Empty or duplicate training subset')
    candidates=[r for r in rows if r.get('robot')==robot]
    lookup={str(Path(r['output']).resolve()):r for r in candidates}
    if len(lookup)!=len(candidates):raise ValueError('Duplicate manifest output')
    expected={path for path,row in lookup.items() if row.get('split')=='train'}
    if set(paths)!=expected:raise ValueError('Subset does not exactly match manifest training paths')
    selected=[]
    for path in paths:
        row=lookup[path]
        if (not row.get('source_id') or not row.get('dataset') or type(row.get('frames')) is not int
            or row['frames']<2 or Path(path).name!=row['name']+'.npz'):
            raise ValueError('Invalid train clip source identity')
        selected.append(row)
    return selected


def needs_probe(completed,robot,iteration):
    if not isinstance(completed,dict) or type(iteration) is not int or iteration<0:
        raise ValueError('Invalid checkpoint iteration or completion ledger')
    last=completed.get(robot,-1)
    if type(last) is not int or last < -1:raise ValueError('Invalid completed checkpoint iteration')
    return iteration>last
