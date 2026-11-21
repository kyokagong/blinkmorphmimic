"""Continue the optimized joint checkpoint with balanced stand-up practice."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def config_values(parent_cfg, parent_root, windows, log_dir, run_name, gpu_ids, freeze_receipt, iterations=6000, num_envs=2048):
    from morphmimic.train.recovery import file_sha
    if len(gpu_ids) != 3 or len(set(gpu_ids)) != 3 or not 5000 < iterations <= 6000:
        raise ValueError('Recovery requires three distinct GPUs and target in (5000,6000]')
    values=dict(parent_cfg)
    values.update(gpu_ids=list(gpu_ids),run_name=run_name,log_dir=str(log_dir),num_envs=num_envs,
        iterations=iterations,lr_decay_iters=10000,wm_lr=1e-4,policy_lr=5e-5,critic_lr=1e-4,
        ckpt_every=250,log_every=20,initial_checkpoint_root=None,freeze_receipt=str(freeze_receipt),
        finetune_parent_receipt=str(Path(log_dir)/run_name/'parent.json'),
        finetune_parent_receipt_sha256=file_sha(Path(log_dir)/run_name/'parent.json'),
        recovery_windows=str(Path(windows).resolve(strict=True)),recovery_windows_sha256=file_sha(windows),
        recovery_reset_share=.8,probe_focus_dir=None)
    if values.get('optimizer')!='muon' or values.get('share')!='all' or not values.get('kp_interface'):
        raise ValueError('Recovery requires the final optimized joint Muon checkpoint')
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-root','windows','log-dir','run-name','freeze-receipt'):p.add_argument('--'+name,required=True)
    p.add_argument('--gpu-ids',nargs=3,type=int,required=True)
    p.add_argument('--iterations',type=int,default=6000);p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--data-root',type=Path,help='Append-only mixed dataset with transition.json')
    p.add_argument('--recovery-reset-share',type=float,default=.5)
    p.add_argument('--resume',help='Resume this frozen run after interruption')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    import torch
    from morphmimic.train.trainer import TrainCfg
    from morphmimic.train.recovery import file_sha
    path=Path(a.parent_root)/'q1/ckpt_5000.pt'
    if path.is_symlink():raise ValueError('Parent checkpoint must be fixed')
    before=file_sha(path);parent=torch.load(path,map_location='cpu',weights_only=False)
    if parent['iteration']!=5000 or file_sha(path)!=before:raise ValueError('Parent checkpoint identity differs')
    values=config_values(parent['cfg'],a.parent_root,a.windows,a.log_dir,a.run_name,a.gpu_ids,a.freeze_receipt,a.iterations,a.num_envs)
    values['recovery_reset_share'] = a.recovery_reset_share
    if a.data_root:
        from morphmimic.train.dataset_transition import verify_append_transition
        transition_path = a.data_root/'transition.json'
        verify_append_transition(transition_path,file_sha(transition_path),a.data_root/'manifest.json','q1')
        manifest = json.loads((a.data_root/'manifest.json').read_text())
        values.update(prepared_manifest=str(a.data_root/'manifest.json'),
            subset_yamls=tuple(str(a.data_root/r/'train.yaml') for r in ('q1','bumi','g1')),
            human_root=manifest['human_root'],contact_floor_receipt=str(a.data_root/'contact-floor.json'),
            dataset_transition=str(transition_path),dataset_transition_sha256=file_sha(transition_path))
    cfg=TrainCfg(**values)
    if a.dry_run:print(json.dumps(asdict(cfg),indent=2));return
    from morphmimic.scripts.gpu_guard import select_gpus
    from morphmimic.scripts.train import launch
    select_gpus(a.gpu_ids,True)
    launch(cfg,a.resume or str(path),explicit={'failure_reset'})

if __name__=='__main__':main()
