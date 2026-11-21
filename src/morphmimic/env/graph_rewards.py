"""Tracking rewards with fixed device indices for CUDA capture."""
from functools import partial

import torch
from mjlab.tasks.tracking.mdp import rewards
from mjlab.utils.lab_api.math import quat_error_magnitude


def _indexed_reward(env, command_name, std, body_names=None, *, indices, left, right, rotation=False):
  command = env.command_manager.get_term(command_name)
  a, b = getattr(command, left)[:, indices], getattr(command, right)[:, indices]
  error = quat_error_magnitude(a, b) ** 2 if rotation else torch.sum(torch.square(a - b), dim=-1)
  return torch.exp(-error.mean(-1) / std**2)


def prepare_graph_rewards(cfg, device):
  """Replace only the four equivalent tracking callables on this environment config."""
  mapping = {
    rewards.motion_relative_body_position_error_exp: ('body_pos_relative_w', 'robot_body_pos_w', False),
    rewards.motion_relative_body_orientation_error_exp: ('body_quat_relative_w', 'robot_body_quat_w', True),
    rewards.motion_global_body_linear_velocity_error_exp: ('body_lin_vel_w', 'robot_body_lin_vel_w', False),
    rewards.motion_global_body_angular_velocity_error_exp: ('body_ang_vel_w', 'robot_body_ang_vel_w', False),
  }
  for term in cfg.rewards.values():
    if term is None or term.func not in mapping:
      continue
    selected = term.params.get('body_names')
    bodies = cfg.commands[term.params['command_name']].body_names
    indices = torch.tensor([i for i, name in enumerate(bodies) if selected is None or name in selected],
                           dtype=torch.long, device=device)
    left, right, rotation = mapping[term.func]
    term.func = partial(_indexed_reward, indices=indices, left=left, right=right, rotation=rotation)
