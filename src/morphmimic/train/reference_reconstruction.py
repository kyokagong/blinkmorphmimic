"""Shared latent-to-reference supervision for motion and causal five-point human input."""
from __future__ import annotations

import torch
from torch import Tensor

from morphmimic.constants import GOAL_K, HUMAN_FIVE_IDX, NUM_HUMAN
from morphmimic.model.human import fixed_human_masks, human_features, heading_features, mask_human_goal
from morphmimic.model.rewards_torch import goal_features
from morphmimic.model.state import anchor_quat


def reference_reconstruction_loss(trainer, start, *, human: bool,
                                  point_mask=None, frame_mask=None, heading_mask=None) -> Tensor:
  """Masked normalized-goal MSE, evaluated outside the actor's frozen-module block.

  Gradients reach only the selected encoder and shared reconstruction head. The
  replay state and robot reference targets are detached; human inputs come solely
  from authentic sidecars at the current frame. Missing canonical joints and rows
  missing any of the five required human points contribute no error. Empty paired
  batches keep a zero-valued graph for consistent distributed gradient collectives.
  """
  policy, library = trainer.policy, trainer.library
  batch, device = len(start.state), start.state.device
  with torch.no_grad():
    target = trainer.goal_norm(goal_features(start.state.detach(),
      library.future(start.frame, GOAL_K), trainer.cfg.goal_anchor))
    valid = torch.ones(batch, dtype=torch.bool, device=device)
    if human:
      defaults = fixed_human_masks(batch, 'five', 1, device)
      point_mask = defaults[0] if point_mask is None else point_mask
      frame_mask = defaults[1] if frame_mask is None else frame_mask
      heading_mask = defaults[2] if heading_mask is None else heading_mask
      if point_mask.shape != (batch, NUM_HUMAN) or frame_mask.shape != (batch, GOAL_K) or heading_mask.shape != (batch,):
        raise ValueError('human reconstruction masks have invalid shapes')
      if any(m.dtype != torch.bool for m in (point_mask, frame_mask, heading_mask)):
        raise ValueError('human reconstruction masks must be boolean')
      if bool(frame_mask[:,1:].any()) or not bool(frame_mask[:,0].all()):
        raise ValueError('human reconstruction requires current-frame horizon 1')
      if library.has_human:
        available = library.human_valid_at(start.frame)
        current = library.human_future(start.frame, 1)
      else:
        available = torch.zeros(batch, NUM_HUMAN, dtype=torch.bool, device=device)
        current = start.state.new_zeros(batch, 1, NUM_HUMAN, 7)
      valid = available[:, HUMAN_FIVE_IDX].all(-1)
      valid &= (point_mask[:,0] | point_mask[:,15]) & point_mask.any(-1)
      valid &= (~point_mask | available).all(-1)
      point_mask = point_mask & available
      # Invalid/unobserved sidecar slots may have zero quaternions. Neutral values
      # make feature math defined, while masks remove their input and supervision.
      neutral = torch.zeros_like(current); neutral[...,3] = 1
      visible = (available & valid[:,None])[:,None,:,None]
      current = torch.where(visible, current, neutral)
      poses = current.expand(-1, GOAL_K, -1, -1).contiguous()
      features = trainer.human_norm(human_features(poses[...,:3],poses[...,3:],point_mask))
      heading = trainer.heading_norm(heading_features(anchor_quat(start.state.detach()),poses[...,3:],point_mask))
      inputs = mask_human_goal(features,point_mask,frame_mask,heading,heading_mask)
      target = torch.where(valid[:,None],target,torch.zeros_like(target))
    mask = policy.decoder.joint_mask
    dimensions = torch.cat([mask,mask,mask.new_ones(9)]).repeat(GOAL_K)
  latent = policy.encode_human(inputs) if human else policy.encode(target)
  prediction = policy.reconstruct(latent)
  included = valid[:,None] & dimensions.bool()
  error = torch.where(included,prediction-target,torch.zeros_like(prediction))
  return error.square().sum() / (valid.sum().clamp_min(1) * dimensions.sum())
