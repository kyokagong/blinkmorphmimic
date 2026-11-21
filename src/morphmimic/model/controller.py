"""Conditional simulator controller dynamics; not a hardware actuator calibration.

Queues are newest first in canonical absolute joint radians. The final transition
is the extra native write shared by all worlds when *any* world resets. Its clock
is an explicit exogenous input, never inferred from a single imagined robot.
"""
import copy
import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

QUEUE_LEN = 11
MAX_SUBSTEPS = 10


@dataclass(frozen=True)
class ControllerCfg:
  max_lag: int
  decimation: int
  update_period: int
  physics_dt: float
  min_lag: int = 0
  per_env_phase: bool = False

  def __post_init__(self):
    if not 0 <= self.min_lag <= self.max_lag < QUEUE_LEN:
      raise ValueError("native_delay_v1 supports lag bounds 0..10")
    if not 1 <= self.decimation <= MAX_SUBSTEPS or self.update_period < 0 or not math.isfinite(self.physics_dt) or self.physics_dt <= 0:
      raise ValueError("invalid native controller timing")


class ControllerState(NamedTuple):
  queue: Tensor  # [B, 11, 29], unused tail zero
  meta: Tensor  # [B, 4] int64: lag, pushes, step_count, phase


class ControllerRecord(NamedTuple):
  before: ControllerState
  after: ControllerState
  targets: Tensor
  lags: Tensor
  clock: Tensor


class ControllerTransition(NamedTuple):
  after: ControllerState  # immediately after the physics substeps, before reset
  next: ControllerState  # surviving world's state after the explicit reset clock
  targets: Tensor  # [B, 10, 29], actual effective controls, unused tail zero
  lags: Tensor  # [B, 10], actual lag setting, unused tail zero


class NativeDelayController(nn.Module):
  def __init__(self, cfg: ControllerCfg, q0: Tensor, scale: Tensor, mask: Tensor,
               lower: Tensor | None = None, upper: Tensor | None = None):
    super().__init__()
    self.cfg = cfg
    self.plant_identity = None
    if q0.shape != (29,) or scale.shape != (29,) or mask.shape != (29,):
      raise ValueError("controller mapping requires canonical29 vectors")
    if not torch.isfinite(q0).all() or not torch.isfinite(scale).all() or not (scale[mask.bool()] > 0).all():
      raise ValueError("controller mapping requires finite defaults and positive active scales")
    if (lower is None) != (upper is None):
      raise ValueError("controller clipping requires both bounds or neither")
    self.clip_enabled = lower is not None
    if self.clip_enabled and (lower.shape != (29,) or upper.shape != (29,)
        or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
        or not (lower <= upper).all()):
      raise ValueError("controller clipping requires finite ordered canonical29 bounds")
    for name, value in (("q0", q0), ("scale", scale), ("mask", mask.bool()),
                        ("lower", torch.zeros_like(q0) if lower is None else lower),
                        ("upper", torch.zeros_like(q0) if upper is None else upper)):
      self.register_buffer(name, value.detach().clone())

  def identity(self):
    return dict(schema="native_delay_v1", timing=asdict(self.cfg),
                plant=copy.deepcopy(self.plant_identity),clip_enabled=self.clip_enabled,
                mapping={name: getattr(self, name).detach().cpu().tolist()
                         for name in ("q0", "scale", "mask", "lower", "upper")},
                future_clock="contiguous_replay_global_reset_bootstrap_v1",
                scope="conditional_simulator_prediction; no hardware calibration",
                resume="fresh native reset and replay warmup; no physical-state continuation")

  def empty_state(self, batch: int) -> ControllerState:
    return ControllerState(self.q0.new_zeros(batch, QUEUE_LEN, 29),
                           torch.zeros(batch, 4, device=self.q0.device, dtype=torch.long))

  def reset_state(self, state: ControllerState, mask: Tensor, phase: Tensor | None = None):
    if self.cfg.per_env_phase and self.cfg.update_period and phase is None:
      raise ValueError("native phase innovations are required when resetting phased delays")
    meta = torch.zeros_like(state.meta)
    if phase is not None:
      meta = torch.stack((meta[:, 0], meta[:, 1], meta[:, 2], phase), -1)
    return ControllerState(torch.where(mask[:, None, None], 0., state.queue),
                           torch.where(mask[:, None], meta, state.meta))

  def target(self, action: Tensor, encoder_bias: Tensor):
    target = self.q0 + action * self.scale
    if self.clip_enabled:
      target = target.clamp(self.lower,self.upper)
    return (target - encoder_bias) * self.mask

  def innovations(self, batch: int):
    return torch.randint(self.cfg.min_lag, self.cfg.max_lag + 1,
                         (batch, self.cfg.decimation + 1), device=self.q0.device)

  def _push(self, state, target, candidate):
    lag, pushes, steps, phase = state.meta.unbind(-1)
    queued = torch.cat((target[:, None], state.queue[:, :self.cfg.max_lag]), 1)
    queued = torch.where((pushes == 0)[:, None, None], target[:, None], queued)
    if self.cfg.max_lag + 1 < QUEUE_LEN:
      queued = torch.cat((queued, torch.zeros_like(state.queue[:, self.cfg.max_lag+1:])), 1)
    update = ((steps + phase) % self.cfg.update_period == 0) if self.cfg.update_period else torch.ones_like(lag, dtype=torch.bool)
    lag = torch.where(update, candidate, lag)
    pushes = pushes + 1
    selected = lag.minimum(pushes - 1)
    effective = queued.gather(1, selected[:, None, None].expand(-1, 1, 29)).squeeze(1)
    return ControllerState(queued, torch.stack((lag, pushes, steps + 1, phase), -1)), effective

  def transition(self, state: ControllerState, action: Tensor, encoder_bias: Tensor,
                 candidate_lags: Tensor, reset_clock: Tensor | None) -> ControllerTransition:
    if reset_clock is None:
      raise ValueError("an explicit, sourced global reset clock is required")
    b = action.shape[0]
    if candidate_lags.shape != (b, self.cfg.decimation + 1) or reset_clock.shape != (b,):
      raise ValueError("controller innovations or reset clock have wrong shape")
    target = self.target(action, encoder_bias)
    targets, lags = [], []
    for i in range(self.cfg.decimation):
      state, effective = self._push(state, target, candidate_lags[:, i])
      targets.append(effective)
      lags.append(state.meta[:, 0])
    after = state
    advanced, _ = self._push(after, target, candidate_lags[:, -1])
    following = ControllerState(torch.where(reset_clock[:, None, None], advanced.queue, after.queue),
                                torch.where(reset_clock[:, None], advanced.meta, after.meta))
    for _ in range(self.cfg.decimation, MAX_SUBSTEPS):
      targets.append(torch.zeros_like(target))
      lags.append(torch.zeros_like(state.meta[:, 0]))
    return ControllerTransition(after, following, torch.stack(targets, 1), torch.stack(lags, 1))

  def features(self, targets: Tensor):
    active = targets[:, :self.cfg.decimation]
    normalized = (active - self.q0) / self.scale.clamp_min(1e-8) * self.mask
    return torch.cat((normalized, torch.zeros_like(targets[:, self.cfg.decimation:])), 1).flatten(1)


def validate_controller_checkpoint(wm, checkpoint):
  """Validate before mutating any network, optimizer or plant buffer on resume."""
  saved_mode = checkpoint.get('cfg',{}).get('controller_mode','legacy')
  if saved_mode != wm.cfg.controller_mode:
    raise ValueError('checkpoint controller mode differs; cross-mode resume is unsupported')
  if wm.controller is None:
    if checkpoint.get('controller_identity') is not None:
      raise ValueError('legacy checkpoint unexpectedly carries a controller identity')
    return
  if checkpoint.get('controller_identity') != wm.controller.identity():
    raise ValueError('checkpoint controller identity differs from the actual simulator plant')
  saved = checkpoint.get('wm',{})
  expected = {'controller.'+name for name in wm.controller.state_dict()}
  if {name for name in saved if name.startswith('controller.')} != expected:
    raise ValueError('checkpoint controller buffers have missing or unexpected keys')
  for name, value in wm.controller.state_dict().items():
    other = saved.get('controller.'+name)
    if not isinstance(other,Tensor) or other.dtype != value.dtype or other.shape != value.shape or not torch.equal(other.cpu(),value.detach().cpu()):
      raise ValueError('checkpoint controller identity disagrees with its stored buffers')
