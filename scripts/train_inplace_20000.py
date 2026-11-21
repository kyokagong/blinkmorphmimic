"""Continue the three pinned 7000 policies on the corrected mixed dataset."""
import argparse
import json
from pathlib import Path

from morphmimic.scripts.gpu_guard import select_gpus
from morphmimic.train.recovery import file_sha, verify_finetune_parent
from morphmimic.train.run_identity import ManifestIdentity, create_receipt, write_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'parent', 'data'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    run, parent, data = (path.resolve(strict=True) for path in (args.run, args.parent, args.data))
    config_path, parent_path, freeze = (run / name for name in ('config.json', 'parent.json', 'freeze.json'))
    identity = ManifestIdentity(data / 'manifest.json')
    if args.prepare:
        import torch
        from dataclasses import asdict, replace
        from morphmimic.train.trainer import TrainCfg
        original = json.loads((parent / 'config.json').read_text())
        release = next(row for row in json.loads((data / 'releases.json').read_text())
                       if row['id'] == 'joint-motion-7000')
        checks = {}
        for robot, entry in release['checkpoints'].items():
            path = parent / robot / 'ckpt_7000.pt'
            if str(path) != entry['path'] or file_sha(path) != entry['sha256']:
                raise ValueError(f'7000 checkpoint path or bytes differ for {robot}')
            checks[robot] = dict(entry, freeze_receipt=original['freeze_receipt'])
        windows_sha = file_sha(data / 'windows.json')
        old_floor_sha = file_sha(Path(original['contact_floor_receipt']))
        receipt = dict(schema_version=2, parent_iteration=7000, target_iteration=20000,
                       prepared_manifest_sha256=identity.sha256, windows_sha256=windows_sha,
                       parent_manifest=release['training_manifest'], parent_contact_floor_sha256=old_floor_sha,
                       checkpoints=checks)
        with parent_path.open('x') as stream:
            json.dump(receipt, stream, indent=2)
        original.update(gpu_ids=[0, 2, 3], iterations=20000, num_envs=2048, ckpt_every=5000,
                        keep_checkpoints=4, lr_decay_iters=20000, run_name=run.name,
                        log_dir=str(run.parent), prepared_manifest=str(identity.path),
                        human_root=str(data / 'human50'), contact_floor_receipt=str(data / 'contact-floor.json'),
                        subset_yamls=tuple(str(data / r / 'train.yaml') for r in original['robots']),
                        recovery_windows=str(data / 'windows.json'), recovery_windows_sha256=windows_sha,
                        dataset_transition=None, dataset_transition_sha256=None, freeze_receipt=str(freeze),
                        finetune_parent_receipt=str(parent_path), finetune_parent_receipt_sha256=file_sha(parent_path))
        cfg = TrainCfg(**original)
        for robot, entry in checks.items():
            checkpoint = torch.load(entry['path'], map_location='cpu', weights_only=False)
            verify_finetune_parent(checkpoint, entry['path'], replace(cfg, robot=robot), identity)
        with config_path.open('x') as stream:
            json.dump(asdict(cfg), stream, indent=2)
        write_receipt(freeze, create_receipt(Path(__file__).resolve().parents[1],
                                            args.image, asdict(cfg), identity.sha256))
        return
    from morphmimic.train.trainer import TrainCfg
    cfg = TrainCfg(**json.loads(config_path.read_text()))
    select_gpus(cfg.gpu_ids, True)
    from morphmimic.scripts.train import launch
    resume = args.resume or parent / 'q1' / 'ckpt_7000.pt'
    launch(cfg, str(resume), explicit={'failure_reset'})


if __name__ == '__main__':
    main()
