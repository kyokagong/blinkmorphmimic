"""Cross-check pre-warmup random-initialization receipts against frozen checkpoint 0.

This binds recorded evidence to model bytes; runtime launch/mount provenance remains
in the external launch receipt. It does not infer randomness from weights alone.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from scripts.audit_optimized import audit, construct_models, _fixed_checkpoint
from scripts.prepare_optimized import ROBOTS, sha


def parameter_hash(module, saved):
    digest=hashlib.sha256()
    for name,param in module.named_parameters():
        value=saved.get(name)
        if not isinstance(value,torch.Tensor) or value.shape!=param.shape or value.dtype!=param.dtype:
            raise ValueError(f'Invalid initial parameter {name}')
        digest.update(name.encode())
        digest.update(str((tuple(value.shape),str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _empty_optimizer(saved):
    if not isinstance(saved,dict):return False
    if set(saved)=={'muon','adam'}:
        return all(_empty_optimizer(child) for child in saved.values())
    return isinstance(saved.get('state'),dict) and not saved['state'] and bool(saved.get('param_groups'))


def audit_fresh(run_root,manifest_path,raw_proof_root,frozen_config,freeze_receipt):
    report=audit(run_root,manifest_path,[0],list(ROBOTS),frozen_config=frozen_config,freeze_receipt=freeze_receipt)
    experiment=report['experiment']
    if experiment.get('initial_checkpoint_root') is not None or experiment.get('human_teacher_root'):
        raise ValueError('Frozen configuration enables pretrained initialization')
    proofs=[]
    for robot in ROBOTS:
        path=Path(raw_proof_root)/f'random-init-{robot}.json'
        if path.is_symlink() or not path.is_file():raise ValueError('Invalid raw initialization receipt')
        proof_sha=sha(path);proof=json.loads(path.read_text())
        if (proof.get('robot')!=robot or type(proof.get('iteration')) is not int or proof['iteration']!=0
            or proof.get('random_initialization') is not True or proof.get('previous_checkpoint_loaded') is not False
            or proof.get('optimizer_states_empty') is not True):
            raise ValueError(f'Invalid raw initialization claims: {robot}')
        counts=proof.get('normalizer_counts_before_warmup')
        if (not isinstance(counts,dict) or set(counts)!={'obs_norm','goal_norm','human_norm','heading_norm'}
            or any(type(n) is not int or n!=0 for n in counts.values())):
            raise ValueError(f'Raw initialization normalizers were not empty: {robot}')
        ckpt,checkpoint_sha=_fixed_checkpoint(Path(run_root)/robot/'ckpt_0.pt')
        if 'initialized_from' in ckpt or ckpt.get('resumed_from') is not None:
            raise ValueError(f'Checkpoint has prior training provenance: {robot}')
        optim=ckpt.get('optim',{})
        if set(optim)!={'wm','policy','kp','critic','critic_kp'} or not all(_empty_optimizer(v) for v in optim.values()):
            raise ValueError(f'Initial optimizer states are not empty: {robot}')
        policy,wm,critic=construct_models(ckpt['cfg'],ckpt['morph'],controller_identity=ckpt.get('controller_identity'))
        hashes={name:parameter_hash(module,ckpt[name]) for name,module in
                (('policy',policy),('wm',wm),('critic',critic),('critic_kp',critic))}
        if hashes!=proof.get('parameter_sha256'):
            raise ValueError(f'Raw initialization parameter hashes differ from checkpoint 0: {robot}')
        if sha(path)!=proof_sha:raise ValueError('Raw initialization receipt changed during audit')
        expected=next(row['sha256'] for row in report['checkpoints'] if row['robot']==robot)
        if checkpoint_sha!=expected:raise ValueError('Checkpoint changed between audits')
        proofs.append(dict(robot=robot,path=str(path.resolve()),sha256=proof_sha,
                           parameter_sha256=hashes,checkpoint_sha256=checkpoint_sha))
    # Recheck all fixed artifacts after the complete cross-robot pass.
    for row in proofs:
        if sha(row['path'])!=row['sha256']:raise ValueError('Raw initialization receipt changed during audit')
    for row in report['checkpoints']:
        if sha(row['path'])!=row['sha256']:raise ValueError('Checkpoint changed during audit')
    report.update(random_initialization_verified=True,raw_initializations=proofs,
                  proof_scope='Recorded pre-warmup empty state and parameter hashes match frozen checkpoint 0; launch/no-mount provenance is external')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','data-manifest','raw-proof-root','frozen-config','freeze-receipt','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():raise FileExistsError(a.out)
    report=audit_fresh(a.run_root,a.data_manifest,a.raw_proof_root,a.frozen_config,a.freeze_receipt)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=a.out.parent,prefix='.fresh-audit-',delete=False) as stream:
            temporary=stream.name;json.dump(report,stream,indent=2,allow_nan=False)
        os.link(temporary,a.out)
    finally:
        if temporary is not None:os.unlink(temporary)
    print('Verified recorded fresh initialization for all three robots')

if __name__=='__main__':main()
