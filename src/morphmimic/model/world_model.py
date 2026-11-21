"""Learned world model f_omega: residual dynamics in a canonical frame, analytic reward
and observation heads, a termination head and the k-step fitting loss (paper Eq. 1).

States are world-frame [..., 240] (`morphmimic.model.state`). The networks see the
normalised canonical features of a state in its canonical frame (`WorldModelCfg.frame`:
the full pelvis frame by default, or the pelvis-yaw heading frame), the action and the
context xi = [foot friction 1 | encoder bias 29 | torso CoM offset 3] scaled by the
fixed `ctx_scale` to O(1) and the robot's morphology vector (`WorldModelCfg.morph_dim`,
none by default); in the root frame the dynamics also see the gravity direction and the
body heights (normalised by per-robot height statistics, identity by default). The
network's output is a residual in units of a fixed per-feature scale
(`WorldModelCfg.residual_scale`: the one-step residual RMS by default, or the dataset
std). Plain fp32 code; TF32 matmuls are enabled by the trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from morphmimic.constants import (
  ACTION_DIM,
  CTX_DIM,
  DT,
  LOCAL_DIM,
  NUM_BODIES,
  NUM_JOINTS,
  ROOT_IDX,
)
from morphmimic.model.nets import fill_missing_buffers, mlp, run_mlp
from morphmimic.model.rewards_torch import (
  RewardCfg,
  analytic_obs,
  soft_termination,
  termination_features,
  tracking_reward,
)
from morphmimic.model.state import (
  Frame,
  FrameMode,
  LocalSlices,
  StateNormalizer,
  canonicalize,
  decanonicalize,
  gravity_in_frame,
  split_state,
  state_frame,
)

if TYPE_CHECKING:
  from morphmimic.data.motion_library import MotionLibrary
  from morphmimic.train.replay import WindowBatch

# Brings xi to O(1): friction ~1, encoder bias ~0.01 rad, CoM offset ~0.05 m.
_CTX_SCALE = (1.0,) * 1 + (100.0,) * NUM_JOINTS + (20.0,) * 3
_DEFAULT_REWARD_CFG = RewardCfg()
# Clamp for the logit of the soft analytic termination probability.
_SOFT_LOGIT_EPS = 1e-6
# Hazard head: cap of its gains in units of their initial values, and the floor of the
# failure probability d inside its log (`HazardHead.forward`).
_HAZARD_GAIN_CAP = 10.0
_HAZARD_D_FLOOR = 1e-12
# Extra dynamics inputs per frame mode (`WorldModel.frame_inputs`): root mode feeds the
# gravity direction in the frame (3) and the 14 body heights.
_FRAME_INPUT_DIM = {"heading": 0, "root": 3 + NUM_BODIES}
# Floors of the one-step residual scale per feature class (`residual_floor`).
_RESIDUAL_FLOORS = (
  (LocalSlices.joint_pos, 1e-4),  # rad
  (LocalSlices.joint_vel, 1e-2),  # rad/s
  (LocalSlices.body_pos, 1e-4),  # m
  (LocalSlices.body_rot6d, 1e-4),
  (LocalSlices.body_lin_vel, 1e-3),  # m/s
  (LocalSlices.body_ang_vel, 1e-3),  # rad/s
)


def residual_floor() -> Tensor:
  """Per-dim floor of the one-step residual scale [LOCAL_DIM]: joint
  positions 1e-4 rad, joint velocities 1e-2 rad/s, body positions 1e-4 m, rot6d 1e-4,
  linear velocities 1e-3 m/s, angular velocities 1e-3 rad/s (float32, CPU)."""
  floor = torch.zeros(LOCAL_DIM)
  for part, value in _RESIDUAL_FLOORS:
    floor[part] = value
  return floor


def joint_correction_target(state: Tensor, next_state: Tensor, dt: float,
                            joint_mask: Tensor | None = None) -> Tensor:
  """Actual semi-implicit correction target, radians, independently of loss scales."""
  if not math.isfinite(dt) or dt <= 0 or state.shape != next_state.shape:
    raise ValueError("correction targets require matching states and positive finite dt")
  current, following = split_state(state), split_state(next_state)
  target = following.joint_pos - current.joint_pos - dt * following.joint_vel
  return target if joint_mask is None else target * joint_mask


def joint_correction_rms(state: Tensor, next_state: Tensor, dt: float,
                         joint_mask: Tensor | None = None) -> Tensor:
  """Per-joint replay correction RMS, double accumulation over all sample dimensions."""
  if state.ndim < 2 or state.numel() == 0:
    raise ValueError("correction RMS requires nonempty batched transitions")
  target = joint_correction_target(state, next_state, dt, joint_mask)
  return target.double().square().mean(tuple(range(target.ndim - 1))).sqrt().to(state.dtype)


class HazardHead(nn.Module):
  """Monotone hazard termination head (`WorldModelCfg.termination = "hazard"`): four
  scalars on the anchor deviations e_z, e_tilt of `termination_features`,

    d = 1 - (1 - sigmoid(k_z (e_z - 0.25) + b_z))
          * (1 - sigmoid(k_u (e_tilt - 0.8) + b_u)),

  the probability that either margin to the simulator's termination thresholds
  (`hard_termination`: 0.25 m, 0.8) fails. The gains k = softplus(raw), capped at
  10 k0, start at k0 = (100 /m, 20) and the offsets b at 0, so each sigmoid is 1/2 at
  its threshold; d never decreases with either deviation.
  """

  k0 = (100.0, 20.0)
  """Initial gains of the height (1/m) and tilt margins."""
  thresholds = (0.25, 0.8)
  """Thresholds of e_z (m) and e_tilt, as in `hard_termination`."""

  def __init__(self) -> None:
    super().__init__()
    # raw = softplus^-1(k0) = log(expm1(k0)), in double (expm1(100) overflows float32).
    self.raw_k_z = nn.Parameter(torch.tensor(math.log(math.expm1(self.k0[0]))))
    self.raw_k_u = nn.Parameter(torch.tensor(math.log(math.expm1(self.k0[1]))))
    self.b_z = nn.Parameter(torch.tensor(0.0))
    self.b_u = nn.Parameter(torch.tensor(0.0))

  def k_z(self) -> Tensor:
    """Gain of the height margin (1/m): softplus(raw_k_z), capped at 10 k0[0]."""
    return F.softplus(self.raw_k_z).clamp(max=_HAZARD_GAIN_CAP * self.k0[0])

  def k_u(self) -> Tensor:
    """Gain of the tilt margin: softplus(raw_k_u), capped at 10 k0[1]."""
    return F.softplus(self.raw_k_u).clamp(max=_HAZARD_GAIN_CAP * self.k0[1])

  @torch.no_grad()
  def metrics(self) -> dict[str, Tensor]:
    """The head's scalars for logging, as detached 0-dim tensors on its device (no
    host sync): the capped gains `k_z` (1/m) and `k_u`, and the effective thresholds
    `thr_z` = 0.25 - b_z / k_z (m) and `thr_tilt` = 0.8 - b_u / k_u, the deviations
    at which each sigmoid is 1/2."""
    k_z, k_u = self.k_z(), self.k_u()
    return {
      "k_z": k_z,
      "k_u": k_u,
      "thr_z": self.thresholds[0] - self.b_z / k_z,
      "thr_tilt": self.thresholds[1] - self.b_u / k_u,
    }

  def forward(self, e_z: Tensor, e_tilt: Tensor) -> Tensor:
    """Termination logit [...] (float32) of d for the deviations e_z, e_tilt [...].

    Computed from the log-survival log(1 - d) = logsigmoid(-x_z) + logsigmoid(-x_u),
    as log(d) - log(1 - d) with log(d) = log(-expm1(log(1 - d))) floored at d = 1e-12,
    so the logit and its gradients stay finite when both sigmoids saturate.
    """
    x_z = self.k_z() * (e_z.float() - self.thresholds[0]) + self.b_z
    x_u = self.k_u() * (e_tilt.float() - self.thresholds[1]) + self.b_u
    log_s = F.logsigmoid(-x_z) + F.logsigmoid(-x_u)
    log_d = torch.log((-torch.expm1(log_s)).clamp(min=_HAZARD_D_FLOOR))
    return log_d - log_s


@dataclass
class WorldModelCfg:
  hidden: tuple[int, ...] = (2048, 2048, 2048)
  act: str = "elu"
  term_hidden: tuple[int, ...] = (256, 256)
  """Hidden widths of the learned termination MLP (`termination="learned"` only)."""
  termination: Literal["learned", "soft", "hazard"] = "hazard"
  """Termination head (`WorldModel.termination_logit`): "hazard" (default) is
  `HazardHead`, four scalars monotone in the two anchor margins; "learned" is the MLP
  `term_mlp` on [normalised features, e_z, e_tilt] (the `main` head); both are fit by
  the BCE term of `fit_loss`. "soft" is the fixed analytic `soft_termination`, with no
  loss term."""
  layer_norm: bool = False
  """LayerNorm after each hidden layer of the dynamics MLP."""
  bf16: bool = False
  """bfloat16 autocast for the MLP matmuls (`nets.run_mlp`); state integration, the
  analytic heads and the loss stay float32."""
  ctx_dim: int = CTX_DIM
  action_dim: int = ACTION_DIM
  frame: FrameMode = "root"
  """Canonical frame of the features (`state_frame`): "root" (default) is the full
  pelvis frame, SuperTrack's character-local frame with no yaw singularity, and adds
  the gravity direction in that frame and the 14 body heights to the dynamics inputs
  (`WorldModel.frame_inputs`); "heading" is the previous yaw-only frame, with no extra
  inputs (the layout of `main` checkpoints)."""
  residual_scale: Literal["state", "step"] = "step"
  """Per-feature scale of the dynamics residual and of the state loss
  (`WorldModel.delta_scale`): "step" (default) is the fixed `residual_scale` buffer,
  set once from data by `set_residual_scale` (the trainer uses the RMS of the one-step
  canonical residual over the replay after the warm-up) and 1 until then, so the
  network predicts, and is penalised, in units of the one-step change; "state" is the
  state normaliser's std, the dataset spread of each feature (the `main` behaviour).
  The network inputs are normalised by that std in both modes. Semi-implicit joint
  position corrections instead use the replay-calibrated joint_correction_scale for output
  units; this scale still normalizes the resulting state errors."""
  joint_dynamics: Literal["residual", "semi_implicit"] = "residual"
  dt: float = DT
  joint_position_correction_scale: float = 0.01
  """Initial radians per output unit of q correction after dt * next joint velocity.
  Replay calibration replaces the persistent per-joint joint_correction_scale buffer;
  it remains independent of the state-error scale used by fit_loss."""
  state_mode: Literal["full", "compact_fk"] = "full"
  """Compact mode predicts 29 q, 29 qd and 15 root features, deriving all bodies
  and their velocities from the exact robot tree; it requires RobotKinematics."""
  morph_dim: int = 0
  """Width of the robot's morphology vector (`morphmimic.robots.morphology`), held in
  the `morph` buffer (`WorldModel.set_morphology`) and appended last to the dynamics
  input; 0 (default) is the single-robot model, whose input is unchanged."""

  controller_mode: Literal["legacy", "native_delay_v1"] = "legacy"


class WorldModel(nn.Module):
  """s_{t+1} = f(s_t, a_t, xi) with analytic reward/obs heads and a termination head.

  Dynamics: in the canonical frame F_t of s_t (`frame`, by `cfg.frame`),
  `dynamics_mlp([norm(phi(s_t)), e(s_t), a_t, xi * ctx_scale, m])` outputs 268 deltas
  in scaled units, where e(s_t) are the extra frame inputs (`frame_inputs`: gravity in
  F_t and the normalised body heights in root mode, none in heading mode) and m the
  robot's morphology vector (`morph`, absent when `cfg.morph_dim` is 0). The next
  canonical features are phi(s_t) + sigma * out with sigma = `delta_scale()` (the one-step
  residual scale in step mode, the state normaliser's std in state mode;
  `cfg.residual_scale`): joints, body positions and velocities add the delta; body
  rot6d adds it and is re-orthonormalised by Gram-Schmidt (in `decanonicalize`). The
  result is mapped back to the world with F_t. The MLP's output layer is initialised
  to zero: residual mode predicts s_{t+1} = s_t, while semi-implicit mode retains
  its q + dt * qd integration prior.

  Termination (`cfg.termination`), on the analytic anchor deviations e_z, e_tilt of s'
  against the reference: `"hazard"` d = `hazard(e_z, e_tilt)` (`HazardHead`, monotone
  in both); `"learned"` d = sigmoid(term_mlp([norm(phi(s')), e_z, e_tilt])) with phi(s')
  in the own canonical frame of s'; `"soft"` is the analytic `soft_termination`.

  `normalizer`, `joint_limits` [29, 2] (soft limits for the reward's joint-limit term),
  `default_joint_pos` [29] (q0 of the observation), `ctx_scale` [33] and
  `residual_scale` [268] (step mode's sigma, ones until `set_residual_scale`), the
  morphology vector `morph` [morph_dim] (zeros until `set_morphology`; size 0 by
  default) and the body-height statistics `height_mean`, `height_std` [14] (identity
  until `set_height_stats`) are held as buffers (moved and saved with the model). A
  state dict saved before `morph` and the height statistics existed loads with their
  defaults (the identity; `morph` only when `morph_dim` is 0).
  """

  joint_limits: Tensor
  default_joint_pos: Tensor
  ctx_scale: Tensor
  residual_scale: Tensor
  joint_correction_scale: Tensor
  morph: Tensor
  height_mean: Tensor
  height_std: Tensor

  def __init__(
    self,
    cfg: WorldModelCfg,
    normalizer: StateNormalizer,
    joint_limits: Tensor,
    default_joint_pos: Tensor,
    reward_cfg: RewardCfg = _DEFAULT_REWARD_CFG,
    imu_idx: int = ROOT_IDX,
    joint_mask: Tensor | None = None,
    kinematics: nn.Module | None = None,
    controller: nn.Module | None = None,
  ) -> None:
    super().__init__()
    if cfg.joint_dynamics not in ("residual", "semi_implicit"):
      raise ValueError(f"Unknown joint dynamics {cfg.joint_dynamics!r}")
    if not math.isfinite(cfg.dt) or cfg.dt <= 0:
      raise ValueError("dt must be finite and positive")
    if not math.isfinite(cfg.joint_position_correction_scale) or cfg.joint_position_correction_scale <= 0:
      raise ValueError("joint_position_correction_scale must be finite and positive")
    if cfg.state_mode not in ("full", "compact_fk"):
      raise ValueError(f"Unknown state mode {cfg.state_mode!r}")
    if (cfg.state_mode == "compact_fk") != (kinematics is not None):
      raise ValueError("compact_fk requires kinematics; full mode must not attach kinematics")
    if cfg.controller_mode not in ("legacy", "native_delay_v1"):
      raise ValueError("unknown controller mode")
    if (cfg.controller_mode == "native_delay_v1") != (controller is not None):
      raise ValueError("native_delay_v1 requires its actual controller; legacy must not attach one")
    self.controller = controller
    self.kinematics = kinematics
    if cfg.ctx_dim != len(_CTX_SCALE):
      raise ValueError(f"ctx_dim must be {len(_CTX_SCALE)}, got {cfg.ctx_dim}.")
    if cfg.termination not in ("learned", "soft", "hazard"):
      raise ValueError(f"Unknown termination head {cfg.termination!r}.")
    if cfg.frame not in _FRAME_INPUT_DIM:
      raise ValueError(f"Unknown frame mode {cfg.frame!r}.")
    if cfg.residual_scale not in ("state", "step"):
      raise ValueError(f"Unknown residual scale {cfg.residual_scale!r}.")
    if joint_limits.shape != (NUM_JOINTS, 2):
      raise ValueError(
        f"joint_limits must be [29, 2], got {tuple(joint_limits.shape)}."
      )
    if cfg.morph_dim < 0:
      raise ValueError(f"morph_dim must be >= 0, got {cfg.morph_dim}.")
    if default_joint_pos.shape != (NUM_JOINTS,):
      raise ValueError(
        f"default_joint_pos must be [29], got {tuple(default_joint_pos.shape)}."
      )
    if joint_mask is None:
      joint_mask = torch.ones(NUM_JOINTS)
    if joint_mask.shape != (NUM_JOINTS,) or not bool(((joint_mask == 0) | (joint_mask == 1)).all()):
      raise ValueError("joint_mask must be binary [29]")
    self.register_buffer("joint_mask", joint_mask.float().detach().clone())
    feature_mask = torch.ones(LOCAL_DIM)
    feature_mask[LocalSlices.joint_pos] = joint_mask.cpu()
    feature_mask[LocalSlices.joint_vel] = joint_mask.cpu()
    self.register_buffer("feature_mask", feature_mask)
    self.cfg = cfg
    self.reward_cfg = reward_cfg
    self.imu_idx = imu_idx  # tracked slot of the IMU body, for `obs`
    self.normalizer = normalizer
    self.register_buffer("joint_limits", joint_limits.detach().clone())
    self.register_buffer("default_joint_pos", default_joint_pos.detach().clone())
    self.register_buffer("ctx_scale", torch.tensor(_CTX_SCALE))
    self.register_buffer("residual_scale", torch.ones(LOCAL_DIM))
    self.register_buffer("joint_correction_scale", torch.full((NUM_JOINTS,), cfg.joint_position_correction_scale))
    self.register_buffer("morph", torch.zeros(cfg.morph_dim))
    self.register_buffer("height_mean", torch.zeros(NUM_BODIES))
    self.register_buffer("height_std", torch.ones(NUM_BODIES))
    self.register_buffer("output_indices", torch.tensor(
      list(range(58)) + list(range(58, 61)) + list(range(100, 106))
      + list(range(184, 187)) + list(range(226, 229))
      if cfg.state_mode == "compact_fk" else list(range(LOCAL_DIM)), dtype=torch.long))
    in_dim = LOCAL_DIM + _FRAME_INPUT_DIM[cfg.frame] + cfg.action_dim + cfg.ctx_dim
    self.dynamics_mlp = mlp(
      in_dim + cfg.morph_dim + (290 if controller is not None else 0),
      cfg.hidden, len(self.output_indices), cfg.act, layer_norm=cfg.layer_norm,
    )
    # Zero learned residual; semi-implicit joints retain their velocity integration.
    nn.init.zeros_(self.dynamics_mlp[-1].weight)
    nn.init.zeros_(self.dynamics_mlp[-1].bias)
    self.term_mlp = (
      mlp(LOCAL_DIM + 2, cfg.term_hidden, 1, cfg.act)
      if cfg.termination == "learned"
      else None
    )
    self.hazard = HazardHead() if cfg.termination == "hazard" else None

  def _load_from_state_dict(
    self, state_dict: dict[str, Tensor], prefix: str, *args, **kwargs
  ) -> None:
    # State dicts from before the morphology input and the height statistics.
    defaults = {
      "joint_correction_scale": self.joint_correction_scale,
      "output_indices": self.output_indices,
      "joint_mask": self.joint_mask,
      "feature_mask": self.feature_mask,
      "height_mean": torch.zeros_like(self.height_mean),
      "height_std": torch.ones_like(self.height_std),
    }
    if self.cfg.morph_dim == 0:
      defaults["morph"] = self.morph
    fill_missing_buffers(state_dict, prefix, defaults)
    super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  @torch.no_grad()
  def set_morphology(self, vec: Tensor) -> None:
    """Sets the `morph` buffer to the robot's morphology vector `vec` [morph_dim], in
    place (`copy_`: the buffer keeps its storage, so compiled graphs that read it see
    the new values)."""
    if vec.shape != self.morph.shape:
      raise ValueError(
        f"morph must be [{self.cfg.morph_dim}], got {tuple(vec.shape)}."
      )
    self.morph.copy_(vec)

  @torch.no_grad()
  def set_height_stats(self, mean: Tensor, std: Tensor) -> None:
    """Sets the body-height statistics `height_mean`, `height_std` [14] (m; the robot's
    per-body height mean and std, e.g. over its motion library) that normalise the
    heights of the root-mode frame inputs, in place (`copy_`, as `set_morphology`).
    Heading mode has no height inputs and ignores them. `std` must be positive."""
    for name, value in (("height_mean", mean), ("height_std", std)):
      if value.shape != (NUM_BODIES,):
        raise ValueError(f"{name} must be [{NUM_BODIES}], got {tuple(value.shape)}.")
    if not bool((std > 0).all()):
      raise ValueError("height_std must be positive.")
    self.height_mean.copy_(mean)
    self.height_std.copy_(std)

  # Network inputs.

  def frame(self, state: Tensor) -> Frame:
    """The configured canonical frame of states [..., 240] (`state_frame(state,
    cfg.frame)`)."""
    return state_frame(state, self.cfg.frame)

  def frame_inputs(self, state: Tensor, frame: Frame) -> Tensor:
    """Extra dynamics inputs of states [..., 240] in their canonical `frame`. Root
    mode: [..., 17], the gravity direction in the frame (3) then the heights of the 14
    tracked bodies (`body_pos` z, `TRACKED_BODY_NAMES` order) as (z - height_mean) /
    height_std, SuperTrack's up-vector and body heights (its §3.3). With the identity
    defaults the heights are the raw metres (single robot: both are O(1)); a
    multi-robot run sets each robot's statistics (`set_height_stats`) so bodies of
    different size give comparable inputs. Heading mode: empty [..., 0]."""
    if self.cfg.frame == "heading":
      return state.new_zeros(*state.shape[:-1], 0)
    heights = split_state(state).body_pos[..., 2]
    heights = (heights - self.height_mean) / self.height_std
    return torch.cat([gravity_in_frame(frame), heights], dim=-1)

  def local_features(self, state: Tensor) -> Tensor:
    """Normalised canonical features [..., 268] of states [..., 240], each in its own
    canonical frame (`frame`; the input of the termination head and of the critic)."""
    return self.normalizer.normalize(canonicalize(state, self.frame(state))) * self.feature_mask

  def scale_ctx(self, ctx: Tensor) -> Tensor:
    """Context [..., 33] -> network input xi * ctx_scale."""
    mask = torch.cat([self.joint_mask.new_ones(1), self.joint_mask, self.joint_mask.new_ones(3)])
    return ctx * self.ctx_scale * mask

  # Dynamics.

  def delta_scale(self) -> Tensor:
    """Per-feature scale sigma [268] of the dynamics output and of the state loss:
    the `residual_scale` buffer in step mode, the state normaliser's std in state
    mode (`cfg.residual_scale`)."""
    if self.cfg.residual_scale == "state":
      return self.normalizer.std
    return self.residual_scale

  @torch.no_grad()
  def set_residual_scale(self, sigma: Tensor) -> None:
    """Sets the `residual_scale` buffer to `sigma` [268] clamped from below at
    `residual_floor()`, in place (`copy_`: the buffer keeps its storage, so compiled
    graphs that read it see the new values). Call it before the first compiled
    forward; step mode reads it, state mode ignores it."""
    if sigma.shape != (LOCAL_DIM,):
      raise ValueError(f"sigma must be [{LOCAL_DIM}], got {tuple(sigma.shape)}.")
    floor = residual_floor().to(self.residual_scale)
    self.residual_scale.copy_(sigma.to(self.residual_scale).clamp(min=floor))

  @torch.no_grad()
  def set_joint_correction_scale(self, sigma: Tensor) -> None:
    """Calibrate once from replay before compile; restore this buffer on warm resume.

    Zero RMS (including absent joints) uses a 1e-5 rad floor. This changes output
    units only; delta_scale and all physical-state error diagnostics are unchanged.
    """
    if sigma.shape != (NUM_JOINTS,) or not bool(torch.isfinite(sigma).all() and (sigma >= 0).all()):
      raise ValueError("joint correction RMS must be finite nonnegative [29]")
    self.joint_correction_scale.copy_(sigma.to(self.joint_correction_scale).clamp_min(1e-5))

  def step(self, state: Tensor, action: Tensor, ctx: Tensor, *, controller_targets: Tensor | None = None) -> Tensor:
    """One model step: state [B, 240], action [B, 29], ctx [B, 33] -> next state
    [B, 240]. Differentiable w.r.t. state and action. The `morph` buffer is expanded
    to the batch and appended last (when `morph_dim` > 0)."""
    frame = self.frame(state)
    local = canonicalize(state, frame) * self.feature_mask
    parts = [
      self.normalizer.normalize(local) * self.feature_mask,
      self.frame_inputs(state, frame),
      action * self.joint_mask if self.controller is None else torch.zeros_like(action),
      self.scale_ctx(ctx),
    ]
    if self.controller is not None:
      if controller_targets is None or controller_targets.shape != (state.shape[0],10,29):
        raise ValueError("controller mode requires effective targets [B,10,29]")
      parts.append(self.controller.features(controller_targets))
    elif controller_targets is not None:
      raise ValueError("legacy world model cannot consume controller targets")
    if self.cfg.morph_dim:
      parts.append(self.morph.expand(*state.shape[:-1], -1))
    x = torch.cat(parts, dim=-1)
    raw = run_mlp(self.dynamics_mlp, x, self.cfg.bf16)
    if self.cfg.state_mode == "compact_fk":
      delta = torch.zeros_like(local).index_copy(
        -1, self.output_indices, raw * self.delta_scale()[self.output_indices])
    else:
      delta = raw * self.delta_scale()
    if self.cfg.joint_dynamics == "semi_implicit":
      next_velocity = local[..., LocalSlices.joint_vel] + delta[..., LocalSlices.joint_vel]
      correction = raw[..., :NUM_JOINTS] * self.joint_correction_scale
      delta = torch.cat([self.cfg.dt * next_velocity + correction, delta[..., NUM_JOINTS:]], -1)
    # decanonicalize Gram-Schmidt-orthonormalises the updated rot6d.
    result = decanonicalize((local + delta) * self.feature_mask, frame)
    if self.kinematics is not None:
      p = split_state(result)
      result = self.kinematics(p.joint_pos, p.joint_vel, p.body_pos[..., ROOT_IDX, :],
                               p.body_quat[..., ROOT_IDX, :], p.body_lin_vel[..., ROOT_IDX, :],
                               p.body_ang_vel[..., ROOT_IDX, :])
    return result

  def rollout(self, state0: Tensor, actions: Tensor, ctx: Tensor, *, controller_targets: Tensor | None = None) -> Tensor:
    """Autoregressive rollout from state0 [B, 240] on actions [B, k, 29] -> predicted
    states s_1 .. s_k [B, k, 240]."""
    state, states = state0, []
    for j in range(actions.shape[1]):
      state = self.step(state, actions[:, j], ctx, controller_targets=None if controller_targets is None else controller_targets[:,j])
      states.append(state)
    return torch.stack(states, dim=1)

  # Heads.

  def reward(
    self, next_state: Tensor, ref: Tensor, action: Tensor, prev_action: Tensor,
    *, reference_floor: Tensor | None = None, reference_floor_valid: Tensor | None = None
  ) -> Tensor:
    """Analytic tracking reward [...] of the transition reaching `next_state`, against
    the reference `ref` of that transition."""
    total, _ = tracking_reward(
      next_state, ref, action, prev_action, self.joint_limits, self.reward_cfg, self.joint_mask,
      reference_floor=reference_floor, reference_floor_valid=reference_floor_valid
    )
    return total

  def mask_obs(self, obs: Tensor) -> Tensor:
    mask = torch.cat([self.joint_mask, self.joint_mask, self.joint_mask.new_ones(6), self.joint_mask])
    return obs * mask

  def obs(self, state: Tensor, encoder_bias: Tensor, prev_action: Tensor) -> Tensor:
    """Analytic noise-free policy observation [..., 93]."""
    return analytic_obs(
      state, encoder_bias, self.default_joint_pos, prev_action, self.imu_idx, self.joint_mask
    )

  def termination_logit(self, next_state: Tensor, ref: Tensor) -> Tensor:
    """Termination logit [...] of `next_state` [..., 240] against the reference `ref`
    [..., 240] of the transition reaching it. For the soft head, the clamped logit of
    `soft_termination`; for the hazard head, `HazardHead`'s float32 logit."""
    if self.cfg.termination == "soft":
      prob = soft_termination(next_state, ref)
      return torch.logit(prob, eps=_SOFT_LOGIT_EPS)
    e_z, e_tilt = termination_features(next_state, ref)
    if self.hazard is not None:
      return self.hazard(e_z, e_tilt)
    x = torch.cat(
      [self.local_features(next_state), e_z.unsqueeze(-1), e_tilt.unsqueeze(-1)], dim=-1
    )
    return run_mlp(self.term_mlp, x, self.cfg.bf16).squeeze(-1)

  def termination_prob(self, next_state: Tensor, ref: Tensor) -> Tensor:
    """Termination probability [...] (see `termination_logit`)."""
    if self.cfg.termination == "soft":
      return soft_termination(next_state, ref)
    return torch.sigmoid(self.termination_logit(next_state, ref))

  # Fitting (Eq. 1).

  def fit_loss(
    self, batch: WindowBatch, library: MotionLibrary, refs: Tensor | None = None
  ) -> tuple[Tensor, dict[str, Tensor]]:
    """k-step loss on replay windows: roll out from the logged s_0 on the logged
    actions and compare each predicted s_j (j = 1..k) with the logged s_j.

    Both are canonicalised in the LOGGED state's canonical frame (`frame`), so xy and
    yaw drift show up as pelvis position / rotation errors (in its own frame a rigidly
    moved prediction would look perfect). The state term is Eq. 1's squared norm
    ‖ŝ_j − s_j‖² of the 268 features scaled by sigma = `delta_scale()` (the one-step
    residual scale in step mode, the normaliser std in state mode), summed over the
    dims and averaged over windows and steps j (so it is the paper's Σ_j over j up to
    the constant 1/k, which scales the BCE term alike); it is reported as `state_loss`
    (a per-step summed squared error in units of sigma, not a per-dim mean). For the
    hazard and learned heads, `term_bce` is the BCE (mean over windows and steps) of
    the termination logit of each predicted s_j against the flag of transition j-1,
    whose reference is `library.state(batch.frame[:, j-1])` (the frame can hold at a
    clip end, so it is never assumed to advance by one); loss = state_loss + term_bce.
    The soft head adds no loss term, its BCE is only reported.

    info: state_loss (above), term_bce, pred_err_8step (normalised per-dim RMS error
    at step k in dataset-std units, i.e. the normaliser std, of the configured frame
    (`cfg.frame`, root by default), comparable across residual-scale modes; the frame
    still differs from heading-mode runs), err8_pelvis_m (RMS over windows of the
    pelvis position error at step k, metres, world frame), err8_joint_rad (RMS over
    windows and joints of the joint-position error at step k, radians; both comparable
    across residual-scale modes) and term_acc (accuracy of logit > 0), as detached
    0-dim tensors on the model's device (no host sync; the caller converts them when
    it logs).
    """
    pred = self.rollout(batch.state[:, 0], batch.action, batch.ctx,
                        controller_targets=None if batch.controller is None else batch.controller.targets)
    logged = batch.state[:, 1:]
    frame = self.frame(logged)
    # In state mode, normalize(a) - normalize(b) == (a - b) / std.
    diff = canonicalize(pred, frame) - canonicalize(logged, frame)
    diff = diff * self.feature_mask
    err = diff / self.delta_scale()
    # Eq. 1: squared norm over the 268 dims, mean over windows and steps.
    state_loss = err.square().sum(-1).mean()

    if refs is None:
      refs = library.state(batch.frame[:, :-1])
    elif refs.shape != logged.shape:
      raise ValueError("reference override must match the window transition states")
    labels = batch.terminated.float()
    if self.cfg.termination != "soft":
      logits = self.termination_logit(pred, refs)
      term_bce = F.binary_cross_entropy_with_logits(logits, labels)
      loss = state_loss + term_bce
    else:
      with torch.no_grad():
        logits = self.termination_logit(pred, refs)
        term_bce = F.binary_cross_entropy_with_logits(logits, labels)
      loss = state_loss

    with torch.no_grad():
      last_pred, last_logged = split_state(pred[:, -1]), split_state(logged[:, -1])
      pelvis_err = last_pred.body_pos[:, ROOT_IDX] - last_logged.body_pos[:, ROOT_IDX]
      joint_err = last_pred.joint_pos - last_logged.joint_pos
      info = {
        "state_loss": state_loss.detach(),
        "term_bce": term_bce.detach(),
        "pred_err_8step": (diff[:, -1] / self.normalizer.std).square().mean().sqrt(),
        "err8_pelvis_m": pelvis_err.square().sum(-1).mean().sqrt(),
        "err8_joint_rad": joint_err.square().mean().sqrt(),
        "term_acc": ((logits > 0) == batch.terminated).float().mean(),
      }
    return loss, info
