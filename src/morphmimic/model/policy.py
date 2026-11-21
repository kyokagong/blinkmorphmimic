"""Policy (motion encoder + action decoder) and critic networks (paper
Table A.4)."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from morphmimic.constants import (
  ACTION_DIM,
  CTX_DIM,
  GOAL_DIM,
  HIST_LEN,
  HUMAN_GOAL_DIM,
  LOCAL_DIM,
  PROPRIO_DIM,
)
from morphmimic.model.nets import fill_missing_buffers, mlp, run_mlp


def _check_morph(buf: Tensor, vec: Tensor) -> None:
  if vec.shape != buf.shape:
    raise ValueError(f"morph must be [{buf.shape[0]}], got {tuple(vec.shape)}.")


@dataclass
class PolicyCfg:
  enc_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  dec_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  latent: int = 32
  act: str = "elu"
  hist_len: int = HIST_LEN
  human: bool = False  # also build the human skeleton goal encoder (`HumanEncoder`)
  recon: bool = False
  """Also build the reconstruction head (`ReconHead`; the direct human policy,
  `TrainCfg.policy_goal="human"`). Needs `human`."""
  ref_reconstruction_skip: bool = False
  """Shared learned latent-to-reference action baseline for motion and human.
  Requires calibrated goal statistics and a morphology-indexed control table."""
  bf16: bool = False
  """bfloat16 autocast for the MLP matmuls (`nets.run_mlp`); outputs are float32."""
  morph_dim: int = 0
  """Width of the actor's morphology vector (`morph_actor`), appended to the decoder
  input only; 0 (default) is the single-robot policy."""
  film: bool = False
  """FiLM-condition each decoder hidden layer on `morph_actor` (needs `morph_dim` >
  0; see `Decoder`)."""


class MotionEncoder(nn.Module):
  """Goal (K=10 future reference frames) [B, 670] -> latent z [B, latent] in (-1, 1)."""

  def __init__(self, cfg: PolicyCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.net = mlp(GOAL_DIM, cfg.enc_hidden, cfg.latent, cfg.act)

  def forward(self, goal: Tensor) -> Tensor:
    return torch.tanh(run_mlp(self.net, goal, self.cfg.bf16))


class HumanEncoder(nn.Module):
  """Masked human goal [B, 2254] (`mask_human_goal`) -> latent z [B, latent] in
  (-1, 1); the motion encoder's widths, feeding the same decoder."""

  def __init__(self, cfg: PolicyCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.net = mlp(HUMAN_GOAL_DIM, cfg.enc_hidden, cfg.latent, cfg.act)

  def forward(self, goal: Tensor) -> Tensor:
    return torch.tanh(run_mlp(self.net, goal, self.cfg.bf16))


# Hidden widths of the reconstruction head (`ReconHead`).
RECON_HIDDEN = (256, 256)


class ReconHead(nn.Module):
  """[z, morph] -> the normalised motion goal [B, 670] (the direct human policy's
  reconstruction term: the human latent decoded back to the robot reference). The
  morphology vector [morph_dim] is expanded to the batch and appended after z, as the
  decoder appends it (no input when `cfg.morph_dim` is 0); MLP `RECON_HIDDEN`."""

  def __init__(self, cfg: PolicyCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.net = mlp(cfg.latent + cfg.morph_dim, RECON_HIDDEN, GOAL_DIM, cfg.act)

  def forward(self, z: Tensor, morph: Tensor) -> Tensor:
    parts = [z]
    if self.cfg.morph_dim:
      parts.append(morph.expand(*z.shape[:-1], -1))
    return run_mlp(self.net, torch.cat(parts, dim=-1), self.cfg.bf16)


class Decoder(nn.Module):
  """[z, flattened obs history (oldest first, as `ObsHistory.flat`), morph_actor] ->
  action [B, 29] (deterministic, linear output).

  `morph_actor` [morph_dim] is the robot's actor morphology vector, a buffer (zeros
  until `set_morphology`; size 0 and no input when `cfg.morph_dim` is 0) expanded to
  the batch. With `cfg.film`, each hidden layer's pre-activation h becomes
  h * (1 + gamma) + beta with (gamma, beta) = `film[i](morph_actor)`, one
  Linear(morph_dim, 2 * width) per hidden layer, zero-initialised (a fresh FiLM
  decoder computes the plain decoder's function).

  `forward(..., morph=v)` uses the vector `v` [morph_dim] instead of the buffer, for an
  export whose morphology vector is an input (`deploy/export_onnx.py`).
  """

  morph_actor: Tensor

  def __init__(self, cfg: PolicyCfg) -> None:
    super().__init__()
    if cfg.morph_dim < 0:
      raise ValueError(f"morph_dim must be >= 0, got {cfg.morph_dim}.")
    if cfg.film and cfg.morph_dim == 0:
      raise ValueError("film needs morph_dim > 0.")
    self.cfg = cfg
    in_dim = cfg.latent + cfg.hist_len * PROPRIO_DIM + cfg.morph_dim
    self.net = mlp(in_dim, cfg.dec_hidden, ACTION_DIM, cfg.act)
    self.register_buffer("morph_actor", torch.zeros(cfg.morph_dim))
    self.register_buffer("joint_mask", torch.ones(ACTION_DIM))
    self.reference_head = ReconHead(cfg) if cfg.ref_reconstruction_skip else None
    if self.reference_head is not None:
      # Preserve warm-start actions while supervised reconstruction trains the head.
      self.reference_skip_gain = nn.Parameter(torch.zeros(()))
      self.register_buffer("reference_goal_mean", torch.zeros(GOAL_DIM))
      self.register_buffer("reference_goal_std", torch.ones(GOAL_DIM))
      self.register_buffer("reference_morphs", torch.zeros(1, cfg.morph_dim))
      self.register_buffer("reference_q0", torch.zeros(1, ACTION_DIM))
      self.register_buffer("reference_action_scale", torch.ones(1, ACTION_DIM))
      self.register_buffer("reference_controls_ready", torch.tensor(False))
    self.film = None
    if cfg.film:
      self.film = nn.ModuleList(
        nn.Linear(cfg.morph_dim, 2 * width) for width in cfg.dec_hidden
      )
      for layer in self.film:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

  def _load_from_state_dict(
    self, state_dict: dict[str, Tensor], prefix: str, *args, **kwargs
  ) -> None:
    # State dicts from before the morphology input.
    if self.reference_head is not None:
      # Table length is checkpoint metadata, fixed before compile/capture.
      for name in ("reference_morphs", "reference_q0", "reference_action_scale"):
        if prefix + name in state_dict:
          setattr(self, name, torch.empty_like(state_dict[prefix + name], device=self.morph_actor.device))
    fill_missing_buffers(state_dict, prefix, {"joint_mask": self.joint_mask})
    if self.cfg.morph_dim == 0:
      fill_missing_buffers(state_dict, prefix, {"morph_actor": self.morph_actor})
    super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  @torch.no_grad()
  def set_morphology(self, vec: Tensor) -> None:
    """Sets `morph_actor` to `vec` [morph_dim] in place (`copy_`: the buffer keeps its
    storage, so compiled graphs that read it see the new values)."""
    _check_morph(self.morph_actor, vec)
    self.morph_actor.copy_(vec)

  def joint_mask_for(self, morph: Tensor | None = None) -> Tensor:
    """Explicit masked-schema morphology carries its own robot's joint mask."""
    if morph is not None and self.cfg.morph_dim in (154, 145):
      return morph[..., -ACTION_DIM:]
    return self.joint_mask

  def _film_forward(self, x: Tensor, morph: Tensor) -> Tensor:
    """`net(x)` with the FiLM modulation, from the morphology vector `morph`, after
    each hidden Linear."""
    assert self.film is not None
    film = iter(self.film)
    last = len(self.net) - 1
    for i, layer in enumerate(self.net):
      x = layer(x)
      if isinstance(layer, nn.Linear) and i < last:
        gamma, beta = next(film)(morph).chunk(2, dim=-1)
        x = x * (1.0 + gamma) + beta
    return x

  def forward(
    self, z: Tensor, obs_hist_flat: Tensor, morph: Tensor | None = None
  ) -> Tensor:
    mask = self.joint_mask_for(morph)
    if morph is None:
      morph = self.morph_actor
    obs_mask = torch.cat([mask, mask, mask.new_ones(*mask.shape[:-1], 6), mask], dim=-1)
    obs_hist_flat = (obs_hist_flat.unflatten(-1, (self.cfg.hist_len, PROPRIO_DIM)) * obs_mask.unsqueeze(-2)).flatten(-2)
    parts = [z, obs_hist_flat]
    if self.cfg.morph_dim:
      parts.append(morph.expand(*z.shape[:-1], -1))
    x = torch.cat(parts, dim=-1)
    if self.film is None:
      net = self.net
    else:
      def net(h: Tensor) -> Tensor:
        return self._film_forward(h, morph)
    action = run_mlp(net, x, self.cfg.bf16)
    if self.reference_head is not None:
      reference = self.reference_head(z, morph) * self.reference_goal_std + self.reference_goal_mean
      matches = (morph.unsqueeze(-2) == self.reference_morphs).all(-1)
      index = matches.to(torch.int64).argmax(-1)
      shape = (*index.shape, ACTION_DIM)
      q0 = self.reference_q0.index_select(0, index.reshape(-1)).reshape(shape)
      scale = self.reference_action_scale.index_select(0, index.reshape(-1)).reshape(shape)
      action = action + self.reference_skip_gain * (reference[..., :ACTION_DIM] - q0) / scale
      valid = matches.any(-1) & self.reference_controls_ready
      # Unknown morphology must be rejected by the existing finite-action boundary.
      action = torch.where(valid.unsqueeze(-1), action, torch.full_like(action, float("nan")))
    return action * mask


class Policy(nn.Module):
  """Motion-tracking policy: a = decoder(encoder(goal), obs history); the decoder
  also sees the actor morphology vector (`set_morphology`), the encoders never do.

  With `cfg.human` it also has `human_encoder`, a second goal encoder for the masked
  human skeleton goal that feeds the same decoder: a = decoder(human_encoder(goal),
  obs history). It is registered after `decoder`, so a human-free policy's parameters
  and state dict are unchanged by the option.
  """

  def __init__(self, cfg: PolicyCfg | None = None) -> None:
    super().__init__()
    self.cfg = cfg if cfg is not None else PolicyCfg()
    self.encoder = MotionEncoder(self.cfg)
    self.decoder = Decoder(self.cfg)
    self.human_encoder: HumanEncoder | None = (
      HumanEncoder(self.cfg) if self.cfg.human else None
    )
    if self.cfg.recon and not self.cfg.human:
      raise ValueError("recon needs the human encoder (human=True).")
    self.recon_head: ReconHead | None = (
      ReconHead(self.cfg) if self.cfg.recon and not self.cfg.ref_reconstruction_skip else None
    )

  @torch.no_grad()
  def set_joint_mask(self, mask: Tensor) -> None:
    if mask.shape != self.decoder.joint_mask.shape or not bool(((mask == 0) | (mask == 1)).all()):
      raise ValueError("joint_mask must be binary [29]")
    self.decoder.joint_mask.copy_(mask)

  def encode(self, goal: Tensor, morph: Tensor | None = None) -> Tensor:
    """Goal [B, 670] -> z [B, latent], masked for the selected robot."""
    mask = self.decoder.joint_mask_for(morph)
    goal_mask = torch.cat([mask, mask, mask.new_ones(*mask.shape[:-1], 9)], dim=-1)
    goal = (goal.unflatten(-1, (10, 67)) * goal_mask.unsqueeze(-2)).flatten(-2)
    return self.encoder(goal)

  @torch.no_grad()
  def set_reference_stats(self, mean: Tensor, std: Tensor) -> None:
    """The same goal normalization used for reconstruction targets and encoder inputs."""
    if self.decoder.reference_head is None:
      raise RuntimeError("reference reconstruction skip is disabled")
    if mean.shape != (GOAL_DIM,) or std.shape != mean.shape or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
      raise ValueError("reference statistics require finite [670] mean and positive std")
    self.decoder.reference_goal_mean.copy_(mean)
    self.decoder.reference_goal_std.copy_(std)

  @torch.no_grad()
  def set_reference_controls(self, morphs: Tensor, q0: Tensor, action_scale: Tensor) -> None:
    """Store all supported robots' physical calibrations before compile/capture.

    Shared checkpoints must hold identical tables. Explicit morphology overrides
    select their own row; unsupported mixtures are rejected at the action boundary.
    Missing joints may have zero scale, replaced by one before the final mask.
    """
    d = self.decoder
    if d.reference_head is None:
      raise RuntimeError("reference reconstruction skip is disabled")
    if morphs.ndim != 2 or morphs.shape[1] != self.cfg.morph_dim or len(morphs) == 0 or q0.shape != (len(morphs), ACTION_DIM) or action_scale.shape != q0.shape:
      raise ValueError("reference controls require [R,morph_dim], [R,29], [R,29]")
    if not all(bool(torch.isfinite(v).all()) for v in (morphs, q0, action_scale)):
      raise ValueError("reference controls must be finite")
    if self.cfg.morph_dim in (154, 145):
      mask = morphs[:, -ACTION_DIM:]
      if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("reference morphology joint masks must be binary")
    else:
      mask = torch.ones_like(q0)
    if not bool(((action_scale > 0) | (mask == 0)).all()):
      raise ValueError("present joints require positive action scales")
    if len(morphs) > 1 and bool(((morphs[:, None] == morphs[None, :]).all(-1).sum(-1) != 1).any()):
      raise ValueError("reference morphology rows must be unique")
    d.reference_morphs = morphs.detach().clone().to(d.morph_actor)
    d.reference_q0 = q0.detach().clone().to(d.morph_actor)
    d.reference_action_scale = torch.where(mask.bool(), action_scale, 1.).detach().clone().to(d.morph_actor)
    d.reference_controls_ready.fill_(True)

  def set_morphology(self, vec: Tensor) -> None:
    """Sets the decoder's `morph_actor` buffer [morph_dim] in place."""
    self.decoder.set_morphology(vec)

  def forward(
    self, obs_hist_flat: Tensor, goal: Tensor, morph: Tensor | None = None
  ) -> Tensor:
    """Obs history [B, hist_len * 93] and goal [B, 670] -> action [B, 29]. `morph`
    replaces the decoder's `morph_actor` buffer when given (`Decoder`)."""
    return self.decoder(self.encode(goal, morph), obs_hist_flat, morph)

  def motion_parameters(self) -> Iterator[nn.Parameter]:
    """Motion encoder then decoder parameters: `parameters()` of a policy without the
    human encoder."""
    return itertools.chain(self.encoder.parameters(), self.decoder.parameters())

  def human_parameters(self) -> Iterator[nn.Parameter]:
    """Human encoder parameters, then the reconstruction head's (if any); raises if
    the policy has no human encoder."""
    params = self._require_human_encoder().parameters()
    head = self.decoder.reference_head if self.cfg.ref_reconstruction_skip else self.recon_head
    if head is None:
      return params
    return itertools.chain(params, head.parameters())

  def reconstruct(self, z: Tensor, morph: Tensor | None = None) -> Tensor:
    """Latent z [B, latent] -> the predicted normalised motion goal [B, 670]
    (`recon_head` with the decoder's `morph_actor`)."""
    head = self.decoder.reference_head if self.cfg.ref_reconstruction_skip else self.recon_head
    if head is None:
      raise RuntimeError("policy has no reconstruction head")
    return head(z, self.decoder.morph_actor if morph is None else morph)

  def encode_human(self, goal: Tensor) -> Tensor:
    """Masked human goal [B, 2254] -> z [B, latent]."""
    return self._require_human_encoder()(goal)

  def forward_human(
    self, obs_hist_flat: Tensor, goal: Tensor, morph: Tensor | None = None
  ) -> Tensor:
    """Obs history [B, hist_len * 93] and masked human goal [B, 2254] -> action
    [B, 29]. `morph` as in `forward`."""
    return self.decoder(self.encode_human(goal), obs_hist_flat, morph)

  def _require_human_encoder(self) -> HumanEncoder:
    if self.human_encoder is None:
      raise RuntimeError("policy has no human encoder")
    return self.human_encoder


@dataclass
class CriticCfg:
  hidden: tuple[int, ...] = (256, 128)
  act: str = "elu"
  layer_norm: bool = False
  """LayerNorm after each hidden activation (SHAC's critic; stabilises TD learning
  without a target network, Gallici et al. 2025)."""
  value_max: float | None = None
  """Smooth upper bound on V (None: unbounded, the paper's critic). With bounded
  rewards r <= r_max the true value satisfies V <= r_max / (1 - gamma); the bound
  `value_max - softplus(value_max - v)` leaves V unchanged well below it."""
  bf16: bool = False
  """bfloat16 autocast for the MLP matmuls (`nets.run_mlp`); V is float32."""
  morph_dim: int = 0
  """Width of the morphology vector (`Critic.morph`), appended last to the input; 0
  (default) is the single-robot critic."""
  priv_dim: int = 0
  """Width of the privileged input (`CRITIC_PRIV_DIM`, `rewards_torch.
  critic_anchor_pos`), appended after the context; 0 (default) is the critic without
  it (goal version 1 and checkpoints from before the input)."""


class Critic(nn.Module):
  """State value V(s, g, xi) [B].

  Inputs are the network-ready features, prepared by the caller exactly as the world
  model prepares its own inputs: `local_feats` [B, 268] are the NORMALISED canonical
  features of the state in its own canonical frame (`WorldModel.local_features`),
  `goal` [B, 670] the goal features and `ctx` [B, 33] the context already multiplied
  by `WorldModel.ctx_scale` (`WorldModel.scale_ctx`), and `priv` [B, priv_dim] the
  privileged input (`critic_anchor_pos`; unused when `cfg.priv_dim` is 0, so callers
  may always pass it). The robot's morphology vector,
  the `morph` buffer [morph_dim] (zeros until `set_morphology`; size 0 and no input
  when `cfg.morph_dim` is 0), is expanded to the batch and appended last.
  """

  morph: Tensor

  def __init__(self, cfg: CriticCfg | None = None) -> None:
    super().__init__()
    self.cfg = cfg if cfg is not None else CriticCfg()
    if self.cfg.morph_dim < 0:
      raise ValueError(f"morph_dim must be >= 0, got {self.cfg.morph_dim}.")
    if self.cfg.priv_dim < 0:
      raise ValueError(f"priv_dim must be >= 0, got {self.cfg.priv_dim}.")
    self.net = mlp(
      LOCAL_DIM + GOAL_DIM + CTX_DIM + self.cfg.priv_dim + self.cfg.morph_dim,
      self.cfg.hidden, 1,
      self.cfg.act, layer_norm=self.cfg.layer_norm,
    )
    self.register_buffer("morph", torch.zeros(self.cfg.morph_dim))

  def _load_from_state_dict(
    self, state_dict: dict[str, Tensor], prefix: str, *args, **kwargs
  ) -> None:
    # State dicts from before the morphology input.
    if self.cfg.morph_dim == 0:
      fill_missing_buffers(state_dict, prefix, {"morph": self.morph})
    super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  @torch.no_grad()
  def set_morphology(self, vec: Tensor) -> None:
    """Sets the `morph` buffer to `vec` [morph_dim] in place (`copy_`: the buffer
    keeps its storage, so compiled graphs that read it see the new values)."""
    _check_morph(self.morph, vec)
    self.morph.copy_(vec)

  def forward(
    self, local_feats: Tensor, goal: Tensor, ctx: Tensor, priv: Tensor | None = None
  ) -> Tensor:
    parts = [local_feats, goal, ctx]
    if self.cfg.priv_dim:
      if priv is None or priv.shape[-1] != self.cfg.priv_dim:
        raise ValueError(
          f"critic expects a privileged input of width {self.cfg.priv_dim}, got"
          f" {None if priv is None else tuple(priv.shape)}"
        )
      parts.append(priv)
    if self.cfg.morph_dim:
      parts.append(self.morph.expand(*local_feats.shape[:-1], -1))
    v = run_mlp(self.net, torch.cat(parts, dim=-1), self.cfg.bf16)
    v = v.squeeze(-1)
    if self.cfg.value_max is not None:
      v = self.cfg.value_max - torch.nn.functional.softplus(self.cfg.value_max - v)
    return v
