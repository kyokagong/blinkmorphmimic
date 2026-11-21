"""Continue 6000→7000 with online difficulty sampling over the complete mixed dataset."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def config_values(parent_cfg, data_root, log_dir, run_name, gpu_ids, freeze_receipt,
                  iterations=7000, num_envs=2048):
    from morphmimic.train.recovery import file_sha
    if (len(gpu_ids) != 3 or len(set(gpu_ids)) != 3 or not 6000 < iterations <= 7000
            or num_envs not in (32,2048) or parent_cfg.get('optimizer') != 'muon'
            or parent_cfg.get('share') != 'all' or not parent_cfg.get('kp_interface')):
        raise ValueError('Importance continuation requires the three optimized joint Muon parents')
    data_root=Path(data_root).resolve(strict=True)
    manifest=json.loads((data_root/'manifest.json').read_text())
    receipt=Path(log_dir)/run_name/'parent.json'
    values=dict(parent_cfg)
    values.update(gpu_ids=list(gpu_ids), run_name=run_name, log_dir=str(log_dir),
        iterations=iterations,num_envs=num_envs,lr_decay_iters=10000,
        wm_lr=1e-4,policy_lr=5e-5,critic_lr=1e-4,ckpt_every=250,log_every=20,
        initial_checkpoint_root=None,freeze_receipt=str(freeze_receipt),
        finetune_parent_receipt=str(receipt),finetune_parent_receipt_sha256=file_sha(receipt),
        prepared_manifest=str(data_root/'manifest.json'),
        subset_yamls=tuple(str(data_root/r/'train.yaml') for r in ('q1','bumi','g1')),
        human_root=manifest['human_root'],contact_floor_receipt=str(data_root/'contact-floor.json'),
        dataset_transition=str(data_root/'transition.json'),dataset_transition_sha256=file_sha(data_root/'transition.json'),
        recovery_windows=str(data_root/'windows.json'),recovery_windows_sha256=file_sha(data_root/'windows.json'),
        importance_sampling=True,recovery_reset_share=0.,probe_focus_dir=None,failure_reset=0.)
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-root','data-root','log-dir','run-name','freeze-receipt'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpu-ids',nargs=3,type=int,required=True)
    parser.add_argument('--iterations',type=int,default=7000)
    parser.add_argument('--num-envs',type=int,default=2048)
    parser.add_argument('--resume');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    import torch
    from morphmimic.train.recovery import file_sha
    from morphmimic.train.dataset_transition import verify_append_transition
    from morphmimic.train.trainer import TrainCfg
    path=Path(args.parent_root)/'q1/ckpt_6000.pt'
    if path.is_symlink():raise ValueError('Parent checkpoint must be fixed')
    before=file_sha(path);parent=torch.load(path,map_location='cpu',weights_only=False)
    if parent['iteration']!=6000 or file_sha(path)!=before:raise ValueError('Parent checkpoint identity differs')
    data=Path(args.data_root)
    for robot in ('q1','bumi','g1'):
        verify_append_transition(data/'transition.json',file_sha(data/'transition.json'),data/'manifest.json',robot)
    cfg=TrainCfg(**config_values(parent['cfg'],data,args.log_dir,args.run_name,args.gpu_ids,
        args.freeze_receipt,args.iterations,args.num_envs))
    if args.dry_run:print(json.dumps(asdict(cfg),indent=2));return
    from morphmimic.scripts.gpu_guard import select_gpus
    from morphmimic.scripts.train import launch
    select_gpus(args.gpu_ids,True)
    launch(cfg,args.resume or str(path),explicit={'failure_reset'})


if __name__=='__main__':main()
