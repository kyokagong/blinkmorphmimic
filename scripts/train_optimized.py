"""Launch the fixed joint training run or print its configuration."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def config_values(data_root, human_root, log_dir, run_name, gpu_ids, initial_checkpoint_root=None):
    if len(gpu_ids)!=3 or len(set(gpu_ids))!=3:
        raise ValueError('Three distinct GPUs required')
    return dict(robots=('q1','bumi','g1'), share='all', gpu_ids=list(gpu_ids),
        subset_yamls=tuple(str(Path(data_root)/r/'train.yaml') for r in ('q1','bumi','g1')),
        human_root=str(human_root), log_dir=str(log_dir),run_name=run_name,
        num_envs=2048, iterations=5000, lr_decay_iters=5000, ckpt_every=1000,
        replay_capacity=256,wm_hidden=(1024,1024,1024),wm_batches=5,wm_batch=1024,
        imag_batch=1024,kp_imag_batch=1024,policy_hidden=(2048,1024,512,256),
        allow_missing_human=True,compile='default',latent_align_weight=0.,wm_only_iters=500,
        prepared_manifest=str(Path(data_root)/'manifest.json'),initial_checkpoint_root=initial_checkpoint_root,
        contact_floor_receipt=str(Path(data_root)/'contact-floor.json'),
        probe_focus_dir=str(Path(log_dir)/run_name/'probe-focus'),
        optimizer='muon',kp_interface=True,policy_goal='motion',kp_masking='five',
        kp_fixed_horizon=1,library_dtype='float16',pool_norms=True,
        optimization_schema=1,async_checkpoint=True,actor_lr_mult=4.,actor_lr_decay_return=.75,
        actor_lr_decay_steps=300,adaptive_warmup=True,wm_only_min_iters=100,
        wm_mirror=.5,wm_joint_dynamics='semi_implicit',wm_state_mode='full',
        contact_agreement_weight=.1,foot_slip_weight=-.001,
        explicit_update_graphs=True,collection_graph=True,masked_reset=True,ref_reconstruction_skip=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--log-dir',required=True)
    p.add_argument('--run-name',required=True);p.add_argument('--gpu-ids',nargs=3,type=int,required=True)
    p.add_argument('--initial-checkpoint-root',help='Original initialization parent; retain this on a frozen resume')
    p.add_argument('--resume')
    p.add_argument('--freeze-receipt')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();manifest=json.loads((a.data_root/'manifest.json').read_text())
    if not manifest.get('complete'):raise ValueError('Composite manifest is incomplete')
    values=config_values(a.data_root,manifest['human_root'],a.log_dir,a.run_name,a.gpu_ids,a.initial_checkpoint_root)
    values['freeze_receipt']=a.freeze_receipt
    if not a.dry_run:
        if not a.resume and (Path(a.log_dir)/a.run_name).exists():
            raise FileExistsError('Run already exists; choose a new name or explicitly resume')
        from morphmimic.scripts.gpu_guard import select_gpus
        select_gpus(a.gpu_ids,True)
    from morphmimic.train.trainer import TrainCfg
    cfg=TrainCfg(**values)
    if a.dry_run:
        print(json.dumps(asdict(cfg),indent=2));return
    from morphmimic.scripts.train import launch
    launch(cfg,a.resume)

if __name__=='__main__':main()
