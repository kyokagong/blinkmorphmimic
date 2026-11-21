"""Weight initialization for a new joint run; deliberately distinct from resume."""
import hashlib
from pathlib import Path

import torch
from torch import nn


def initialize_joint(trainer, path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path or path.parent.name != trainer.cfg.robot:
        raise ValueError('Initialization path does not identify the requested robot')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['cfg'].get('robot') != trainer.cfg.robot or checkpoint['iteration'] != trainer.cfg.initial_checkpoint_iteration:
        raise ValueError('Initialization robot or iteration differs')
    state = checkpoint.get('ema_policy', checkpoint['policy'])
    result = trainer.policy.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not name.startswith('decoder.reference_') for name in result.missing_keys):
        raise ValueError(f'Initialization policy architecture differs: {result}')
    for name in ('wm', 'critic', 'critic_kp', 'target_critic', 'target_critic_kp'):
        module = getattr(trainer, name, None)
        if module is None or name not in checkpoint:
            continue
        # Physics/statistics belong to the new corpus and robot; initialize learned
        # tensors only, leaving schema-dependent output heads fresh when sizes differ.
        previous = checkpoint[name]
        parameters = dict(module.named_parameters())
        selected = {key: value for key, value in previous.items()
                    if key in parameters and parameters[key].shape == value.shape}
        module.load_state_dict(selected, strict=False)
    if trainer.cfg.wm_joint_dynamics == 'semi_implicit' and hasattr(trainer.wm, 'dynamics_mlp'):
        last = [layer for layer in trainer.wm.dynamics_mlp.modules() if isinstance(layer, nn.Linear)][-1]
        with torch.no_grad():
            last.weight[:29].zero_()
            last.bias[:29].zero_()
    for name in ('obs_norm', 'goal_norm', 'human_norm', 'heading_norm'):
        module = getattr(trainer, name, None)
        if module is not None:
            if name not in checkpoint:
                raise ValueError(f'Joint initialization requires {name}')
            module.load_state_dict(checkpoint[name])
    if trainer.policy.cfg.ref_reconstruction_skip:
        trainer.policy.set_reference_stats(trainer.goal_norm.mean,
            trainer.goal_norm.var.sqrt() + trainer.goal_norm.eps)
    if any(not bool(value.isfinite().all()) for value in trainer.policy.state_dict().values()
           if value.is_floating_point()):
        raise ValueError('Initialization policy contains nonfinite tensors')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
            raise ValueError('Initialization changed while loading')
    trainer.initialized_from = dict(path=str(path), sha256=digest, iteration=checkpoint['iteration'],
        weights='ema' if 'ema_policy' in checkpoint else 'online', optimizer_state_restored=False,
        world_model_position_head_reset=trainer.cfg.wm_joint_dynamics == 'semi_implicit')
    trainer.it = 0
