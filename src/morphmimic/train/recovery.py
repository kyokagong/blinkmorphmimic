"""Explicit checkpoint continuation and exact recovery reset windows."""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import torch

CATEGORIES = ('lying', 'seated', 'half_squat')

def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def has_optimizer_state(value):
    return isinstance(value, dict) and (bool(value.get('state')) or any(has_optimizer_state(v) for v in value.values()))


def verify_finetune_parent(checkpoint, path, cfg, identity):
    if file_sha(cfg.finetune_parent_receipt) != cfg.finetune_parent_receipt_sha256:
        raise ValueError('Fine-tuning parent receipt SHA differs')
    document = json.loads(Path(cfg.finetune_parent_receipt).read_text())
    parent = document.get('checkpoints', {}).get(cfg.robot, {})
    path = Path(path).absolute()
    parent_iteration = document.get('parent_iteration')
    rewrite = document.get('schema_version') == 2
    valid_range = (parent_iteration == 7000 and cfg.iterations == 20000) if rewrite else (
        document.get('schema_version') == 1 and parent_iteration in (5000, 6000)
        and parent_iteration < cfg.iterations <= parent_iteration + 1000)
    if (type(parent_iteration) is not int or not valid_range
            or document.get('target_iteration') != cfg.iterations
            or document.get('prepared_manifest_sha256') != identity.sha256
            or path.is_symlink() or str(path.resolve(strict=True)) != parent.get('path')
            or file_sha(path) != parent.get('sha256')
            or checkpoint.get('iteration') != parent_iteration
            or checkpoint.get('cfg', {}).get('robot') != cfg.robot
            or checkpoint.get('cfg', {}).get('optimizer') != 'muon'
            or not isinstance(checkpoint.get('ema_policy'), dict) or not checkpoint['ema_policy']
            or type(checkpoint.get('ema_updates')) is not int or checkpoint['ema_updates'] < 1
            or not isinstance(checkpoint.get('optimization_state'), dict)
            or not {'wm', 'policy', 'critic', 'kp', 'critic_kp'}.issubset(checkpoint.get('optim', {}))
            or any(not has_optimizer_state(checkpoint['optim'][key]) for key in ('wm','policy','critic','kp','critic_kp'))
            or any(type(checkpoint.get('optimization_state', {}).get('optimizer_steps', {}).get(key)) is not int
                   or checkpoint['optimization_state']['optimizer_steps'][key] < 1 for key in ('world_model','motion','human'))
            or document.get('windows_sha256') != cfg.recovery_windows_sha256
            or file_sha(cfg.recovery_windows) != cfg.recovery_windows_sha256):
        raise ValueError('Fine-tuning parent, optimizer, EMA or recovery identity differs')
    transition_receipt = None
    if rewrite:
        from morphmimic.train.run_identity import ManifestIdentity
        old = ManifestIdentity(document['parent_manifest']['path'])
        new_manifest = json.loads(identity.path.read_text())
        if (getattr(cfg, 'dataset_transition', None)
                or old.sha256 != document['parent_manifest']['sha256']
                or checkpoint['optimized_training'].get('contact_floor_sha256') != document['parent_contact_floor_sha256']
                or not new_manifest.get('derivation', {}).get('excluded_recovery_references')
                or not new_manifest.get('complete')):
            raise ValueError('Rewritten recovery dataset or historical floor identity differs')
        old.verify_checkpoint(checkpoint, parent.get('freeze_receipt'))
    elif getattr(cfg, 'dataset_transition', None):
        from morphmimic.train.dataset_transition import verify_append_transition
        from morphmimic.train.run_identity import ManifestIdentity
        transition = verify_append_transition(cfg.dataset_transition, cfg.dataset_transition_sha256,
                                              cfg.prepared_manifest, cfg.robot)
        ManifestIdentity(transition['parent_manifest']['path']).verify_checkpoint(checkpoint, parent.get('freeze_receipt'))
        transition_receipt = dict(path=str(Path(cfg.dataset_transition).resolve()),
            sha256=cfg.dataset_transition_sha256,
            parent_manifest_sha256=transition['parent_manifest']['sha256'],
            parent_contact_floor_sha256=transition['parent_contact_floor']['sha256'])
    else:
        identity.verify_checkpoint(checkpoint, parent.get('freeze_receipt'))
    result = dict(path=str(path), sha256=parent['sha256'], iteration=parent_iteration,
                optimizer_state_restored=True, ema_state_restored=True,
                parent_freeze_receipt=checkpoint['optimized_training']['freeze_receipt']
                if 'optimized_training' in checkpoint else None)
    if 'finetuned_from' in checkpoint:
        result['ancestor'] = copy.deepcopy(checkpoint['finetuned_from'])
    if rewrite:
        result['reset_frame_indexed_state'] = True
        result['parent_contact_floor_sha256'] = document['parent_contact_floor_sha256']
    if transition_receipt:
        result['dataset_transition'] = transition_receipt
    return result


def recovery_tensors(document, library, robot, subset_sha256, manifest_sha256):
    if (document.get('schema_version') != 1 or document.get('fps') != library.fps
            or document.get('training_manifest_sha256') != manifest_sha256
            or document.get('subset_sha256', {}).get(robot) != subset_sha256):
        raise ValueError('Recovery window data identity differs')
    max_seconds = document.get('max_recovery_seconds', 10.)
    if type(max_seconds) not in (float, int) or not 10 <= max_seconds <= 15:
        raise ValueError('Invalid recovery window duration limit')
    rows = [r for r in document.get('windows', []) if r.get('robot') == robot]
    if {r.get('category') for r in rows} != set(CATEGORIES):
        raise ValueError('Recovery requires lying, seated and half_squat windows')
    clips = {name: i for i, name in enumerate(library.clip_names)}
    clip_counts = Counter((r['category'], r.get('clip')) for r in rows)
    category_clips = {category: len({r.get('clip') for r in rows if r['category'] == category}) for category in CATEGORIES}
    starts, stops, weights = [], [], []
    for row in rows:
        clip = clips.get(row.get('clip'))
        a, b, stand, hold = (row.get(k) for k in ('start_frame', 'stop_frame', 'stand_frame', 'hold_end_frame'))
        if (clip is None or not row.get('source_id') or any(type(v) is not int for v in (a, b, stand, hold))
                or not 0 <= a < b <= stand < hold <= int(library.clip_len[clip])
                or b >= int(library.clip_len[clip]) or hold - stand < library.fps or hold - a > max_seconds * library.fps):
            raise ValueError('Invalid recovery reset window or standing hold')
        start = int(library.clip_start[clip])
        starts.append(start + a); stops.append(start + b)
        category = row['category']
        weights.append(1 / (3 * category_clips[category] * clip_counts[category, row['clip']]))
    device = library.clip_start.device
    return (torch.tensor(starts, dtype=torch.long, device=device),
            torch.tensor(stops, dtype=torch.long, device=device),
            torch.tensor(weights, dtype=torch.float32, device=device))


def sample_recovery_frames(command, frames):
    recovery = getattr(command, 'recovery', None)
    if recovery is None:
        return frames
    starts, stops, weights = recovery
    choices = torch.multinomial(weights, frames.numel(), replacement=True)
    proposed = starts[choices] + (torch.rand(frames.shape, device=frames.device) * (stops[choices] - starts[choices])).long()
    selected = torch.rand(frames.shape, device=frames.device) < command.recovery_rho
    return torch.where(selected, proposed, frames)


def install_recovery(trainer):
    cfg = trainer.cfg
    if getattr(cfg, 'recovery_windows', None) is None:
        return
    if (not math.isfinite(cfg.recovery_reset_share) or not 0 <= cfg.recovery_reset_share < 1
            or file_sha(cfg.recovery_windows) != cfg.recovery_windows_sha256):
        raise ValueError('Recovery focus fingerprint or reset share differs')
    document = json.loads(Path(cfg.recovery_windows).read_text())
    tensors = recovery_tensors(document, trainer.library, cfg.robot,
        file_sha(cfg.subset_yaml), trainer._manifest_identity.sha256)
    trainer.command.recovery = None if getattr(cfg, 'importance_sampling', False) or cfg.recovery_reset_share == 0 else tensors
    trainer.command.recovery_rho = cfg.recovery_reset_share
    if getattr(trainer, 'collector', None) is not None:
        trainer.collector.refresh_reset_sampler()
