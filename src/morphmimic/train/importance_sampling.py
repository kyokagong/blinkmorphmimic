"""Source-balanced resets driven by observed rollout failures and tracking error.

Each visited step contributes bounded geometric tracking error; each finished
rollout contributes its true failure outcome to its reset bin. Unseen bins start
at 0.5 for both EMAs, with no category prior.
"""
from __future__ import annotations

from collections import Counter

import torch


def source_balanced_clip_weights(clip_names, manifest_rows, robot):
  """CPU metadata only: equal dataset mass, then equal clips within each dataset."""
  selected = [row for row in manifest_rows if row.get('robot') == robot and row.get('split') == 'train']
  rows = {row['name']: row for row in selected}
  if not clip_names or len(set(clip_names)) != len(clip_names) or len(rows) != len(selected):
    raise ValueError('Invalid source-balanced training inventory')
  if any(name not in rows for name in clip_names):
    raise ValueError('Source-balanced clip absent from training manifest')
  datasets = [rows[name].get('dataset') for name in clip_names]
  if any(not isinstance(group, str) or not group.strip() or group != group.strip() for group in datasets):
    raise ValueError('Every training clip requires its manifest dataset')
  group_names = tuple(sorted(set(datasets)))
  counts = Counter(datasets)
  weights = torch.tensor([1.0 / (len(group_names) * counts[group]) for group in datasets], dtype=torch.float32)
  clip_groups = torch.tensor([group_names.index(group) for group in datasets], dtype=torch.long)
  return weights, group_names, clip_groups


def create_state(library):
  if any(not hasattr(library, name) for name in ('importance_clip_weights', 'importance_group_names', 'importance_clip_groups')):
    raise ValueError('Install manifest source-balanced clip weights before configuring importance sampling')
  # Pin identical base bits for GPU collection and independent CPU checkpoint audits.
  bin_clip = library.bin_clip.cpu()
  counts = torch.bincount(bin_clip, minlength=library.num_clips)
  weights = library.importance_clip_weights.detach().cpu().float()
  groups = library.importance_clip_groups.detach().cpu()
  names = library.importance_group_names
  if (weights.shape != (library.num_clips,) or groups.shape != weights.shape or groups.dtype != torch.long
      or not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names)
      or not bool(((groups >= 0) & (groups < len(names))).all())
      or not bool((counts > 0).all()) or not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all())):
    raise ValueError('Invalid source-balanced clip metadata')
  group_counts = torch.bincount(groups, minlength=len(names))
  expected = 1.0 / (len(names) * group_counts[groups].float())
  if not bool((group_counts > 0).all()) or not torch.allclose(weights, expected, rtol=1e-6, atol=0):
    raise ValueError('Clip weights must balance datasets and then clips within datasets')
  base = (weights[bin_clip] / counts[bin_clip].float()).to(library.bin_clip.device)
  return dict(base=base, probabilities=base.clone(),
              calibration=base.new_tensor([2., .02, .3, .7, 8., .25, .5, .5, .05]),
              failure_ema=torch.full_like(base, .5), error_ema=torch.full_like(base, .5),
              visits=torch.zeros_like(library.bin_clip), episodes=torch.zeros_like(library.bin_clip),
              failures=torch.zeros_like(library.bin_clip), clip_group=groups.to(library.bin_clip.device).clone(), bin_clip=library.bin_clip.clone(),
              bin_start=library.bin_start.clone())


def update_statistics(state, bins, error, valid, alpha=.02):
  """Fold duplicate-bin observations together; masked rows contribute nothing."""
  weights = valid.to(state['base'])
  count = torch.zeros_like(state['base']).index_add_(0, bins, weights)
  keep = (1.0 - alpha) ** count
  # where excludes invalid NaNs as well as invalid counts.
  values = torch.where(valid, error, 0.).to(weights)
  total = torch.zeros_like(count).index_add_(0, bins, values)
  state['error_ema'].mul_(keep).add_((1.0 - keep) * total / count.clamp_min(1))
  state['visits'].add_(count.long())


def record_outcomes(state, bins, failed, finished, alpha=.02):
  """Real finished rollout outcomes, credited to the bin sampled at its reset."""
  count = torch.zeros_like(state['base']).index_add_(0, bins, finished.to(state['base']))
  total = torch.zeros_like(count).index_add_(0, bins, (failed & finished).to(count))
  keep = (1.0 - alpha) ** count
  state['failure_ema'].mul_(keep).add_((1.0 - keep) * total / count.clamp_min(1))
  state['episodes'].add_(count.long())
  state['failures'].add_(total.long())


def refresh_probabilities(state):
  """30% source-balanced coverage + 70% difficulty, capped at 8 times the base share.

  Weighted water filling respects different bin base shares for unequal clip lengths.
  This sort runs outside capture at refresh, never on each physics step/reset.
  """
  base = state['base']
  priority = base * (.05 + state['failure_ema'] + state['error_ema'])
  priority /= priority.sum()
  ratio, order = (priority / base).sort(descending=True)
  sorted_base = base[order]
  prefix_base = torch.cat((base.new_zeros(1), sorted_base.cumsum(0)))
  prefix_p = torch.cat((base.new_zeros(1), priority[order].cumsum(0)))
  scale = (1.0 - 8.0 * prefix_base) / (1.0 - prefix_p).clamp_min(1e-12)
  next_ratio = torch.cat((ratio, base.new_zeros(1)))
  first = torch.argmax((scale * next_ratio <= 8.0).to(torch.int8))
  capped = base * torch.minimum(priority / base * scale[first], base.new_tensor(8.0))
  state['probabilities'].copy_(.3 * base + .7 * capped)


def tracking_error(command):
  """Position unit=.25 robot height; rotation/joint unit=.5 radians.

  Anchor and floating root are measured in world space; body positions/rotations
  use the same anchor-relative alignment as the tracking reward. The seven terms
  have equal weight. Missing canonical joints never dilute the native joint RMSE.
  """
  lib, frame, robot = command.library, command.global_frame, command.robot
  height = .25 * command.spec.height_scale
  origin = command._env.scene.env_origins
  root = command._importance_root_index
  root_pos = torch.linalg.vector_norm(lib.root_pos_w[frame].float() + origin - robot.data.body_link_pos_w[:, root], dim=-1)
  def angle(a, b):
    return 2 * torch.acos((a * b).sum(-1).abs().clamp(max=1.))
  root_rot = angle(lib.root_quat_w[frame].float(), robot.data.body_link_quat_w[:, root])
  joint = ((command.joint_pos - command.robot_joint_pos).square() * command._importance_joint_mask).sum(-1)
  joint = (joint / command._importance_joint_mask.sum().clamp_min(1)).sqrt()
  m = command.metrics
  calibrated = (m['error_anchor_pos'] / height + m['error_anchor_rot'] / .5
                + m['error_body_pos'] / height + m['error_body_rot'] / .5
                + root_pos / height + root_rot / .5 + joint / .5) / 7.
  return -torch.expm1(-torch.nan_to_num(calibrated, nan=float('inf')).clamp_min(0))


def export_state(state):
  return {name: value.detach().cpu().clone() for name, value in state.items()}


def validate_state(state, saved):
  """Validate all fields without touching live GPU buffers or captured addresses."""
  if not isinstance(saved, dict) or saved.keys() != state.keys():
    raise ValueError('Importance sampler state fields differ')
  staged = {}
  for name, target in state.items():
    value = saved[name]
    if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
      raise ValueError(f'Invalid importance sampler field {name}')
    if not bool(torch.isfinite(value).all()):
      raise ValueError(f'Nonfinite importance sampler field {name}')
    value = value.to(target.device)
    if name in ('base', 'clip_group', 'bin_clip', 'bin_start', 'calibration') and not torch.equal(value, target):
      raise ValueError(f'Importance sampler library changed: {name}')
    if name in ('failure_ema', 'error_ema') and not bool(((value >= 0) & (value <= 1)).all()):
      raise ValueError(f'Importance sampler EMA outside [0, 1]: {name}')
    if name in ('visits', 'episodes', 'failures') and not bool((value >= 0).all()):
      raise ValueError('Negative importance sampler visits')
    staged[name] = value
  if not bool((staged['failures'] <= staged['episodes']).all()):
    raise ValueError('Importance sampler failures exceed finished rollouts')
  p, base = staged['probabilities'], staged['base']
  if (not torch.isclose(p.sum(), p.new_tensor(1.), atol=1e-5)
      or not bool(((p >= .3 * base * (1-1e-5)) & (p <= 8 * base * (1+1e-5))).all())):
    raise ValueError('Invalid importance sampler probabilities')
  return staged


def load_state(state, saved):
  staged = validate_state(state, saved)
  for name, value in staged.items():
    state[name].copy_(value)


def report(state, library):
  s = export_state(state)
  p = s['probabilities']
  clips = []
  for clip, name in enumerate(library.clip_names):
    mask = s['bin_clip'] == clip
    visits = s['visits'][mask]
    weights = s['episodes'][mask].float()
    count = int(visits.sum())
    episodes = int(s['episodes'][mask].sum()); failures = int(s['failures'][mask].sum())
    clips.append(dict(clip=name, group=library.importance_group_names[int(s['clip_group'][clip])], probability=float(p[mask].sum()), visits=count,
                      failure_ema=float((s['failure_ema'][mask] * weights).sum() / weights.sum()) if episodes else .5,
                      error_ema=float((s['error_ema'][mask] * visits).sum() / visits.sum()) if count else .5,
                      episodes=episodes, failures=failures, rollout_failure_rate=failures / episodes if episodes else None,
                      unseen_bins=int((visits == 0).sum()), probability_floor=float((.3 * s['base'][mask]).sum())))
  bin_groups = s['clip_group'][s['bin_clip']]
  groups = []
  for index, name in enumerate(library.importance_group_names):
    mask = bin_groups == index
    groups.append(dict(group=name, base_mass=float(s['base'][mask].sum()),
                       probability_mass=float(p[mask].sum()), probability_floor=float(.3*s['base'][mask].sum()),
                       clips=int((s['clip_group'] == index).sum()), visits=int(s['visits'][mask].sum()),
                       unseen_bins=int((s['visits'][mask] == 0).sum())))
  return dict(semantics_version=2, groups=groups,
              source_balance='Equal dataset base mass, then equal clip and bin mass; actual probabilities remain adaptive, not fixed dataset quotas.',
              failure_definition='terminated / finished rollouts of the actual collector policy by sampled start bin; timeout is a clean short rollout, not full-clip success', ema_alpha=.02,
              base_mix=.3, difficulty_mix=.7, priority_cap=8., unseen_prior=dict(failure=.5, error=.5),
              policy='Actual collector policy selected by trainer masks/configuration; statistics are pooled across environments and not separated by interface.',
              error_calibration=dict(position_robot_heights=.25, rotation_radians=.5, joint_rmse_radians=.5),
              entropy=float(-(p * p.clamp_min(1e-12).log()).sum()),
              effective_bins=float(1 / p.square().sum()), max_base_ratio=float((p / s['base']).max()),
              unseen_bins=int((s['visits'] == 0).sum()), clips=clips,
              bins=[dict(bin=i, clip=int(s['bin_clip'][i]), start_frame=int(s['bin_start'][i]),
                         probability=float(p[i]), base_probability=float(s['base'][i]), probability_floor=float(.3*s['base'][i]),
                         visits=int(s['visits'][i]), episodes=int(s['episodes'][i]), failures=int(s['failures'][i]),
                         rollout_failure_rate=float(s['failures'][i]/s['episodes'][i]) if s['episodes'][i] else None, failure_ema=float(s['failure_ema'][i]),
                         error_ema=float(s['error_ema'][i])) for i in range(p.numel())])
