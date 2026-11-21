"""Adapt a new human encoder from the pinned 30000-update motion teacher."""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
import torch
from morphmimic.train.human_adapt import HumanAdaptCfg


def build_cfg(teacher_cfg,args):
    names={f.name for f in fields(HumanAdaptCfg)}
    values={k:v for k,v in teacher_cfg.items() if k in names}
    data=Path(args.data_root)
    values.update(robots=('q1','bumi','g1'),share='all',gpu_ids=args.gpu_ids,
        subset_yamls=tuple(str(data/r/'train.yaml') for r in ('q1','bumi','g1')),
        human_root=str(data/'human50'),human_teacher_root=args.teacher_root,
        log_dir=args.run_root,run_name=args.run_name,iterations=args.updates,
        lr_decay_iters=args.updates,num_envs=args.num_envs,ckpt_every=1000,log_every=50,
        kp_interface=True,policy_goal='motion',kp_student='return',pool_norms=True,kp_masking='mixed',kp_fixed_horizon=1,
        kp_jitter=False,kp_drift=False,wm_only_iters=0,kp_imag_batch=1024,
        latent_align_weight=0.,warmup_env_steps=args.warmup,compile=args.compile,wandb=False)
    return HumanAdaptCfg(**values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-root',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--run-root',required=True);p.add_argument('--run-name',required=True)
    p.add_argument('--gpu-ids',type=int,nargs=3,default=[0,2,3])
    p.add_argument('--num-envs',type=int,default=2048);p.add_argument('--updates',type=int,default=10000)
    p.add_argument('--compile',choices=['none','default','reduce-overhead'],default='reduce-overhead')
    p.add_argument('--warmup',type=int,default=100);p.add_argument('--resume')
    p.add_argument('--diagnostic-partial-data',action='store_true')
    args=p.parse_args()
    if args.updates<=0 or args.num_envs<=0 or len(set(args.gpu_ids))!=3:p.error('Positive updates/envs and three distinct GPUs required')
    manifest=json.loads((Path(args.data_root)/'manifest.json').read_text())
    if not manifest.get('complete') and not args.diagnostic_partial_data:p.error('Paired source manifest is incomplete')
    run=Path(args.run_root)/args.run_name
    if run.exists() and not args.resume:p.error('Run already exists; choose a new name or explicitly resume')
    ckpt=torch.load(Path(args.teacher_root)/'q1/ckpt_30000.pt',map_location='cpu',weights_only=False)
    cfg=build_cfg(ckpt['cfg'],args)
    from morphmimic.scripts.gpu_guard import select_gpus
    from morphmimic.scripts.train import launch
    select_gpus(args.gpu_ids,True)
    launch(cfg,args.resume)


if __name__=='__main__':main()
