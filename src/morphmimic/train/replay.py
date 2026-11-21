"""Ring buffer of collected transitions with window and start-state sampling.

Row (env, t) holds the state s_t and clean policy obs o_t the policy acted on, the
executed (noisy) action a_t, the reward r_{t+1} and failure flag d_{t+1} of that step,
the time-out flag (episode or clip-end time-out, false when the step also failed), the
context, the reference frame f_t shown at t, and `ep_step` (0 on an episode's first
row). When an episode ends, a terminal row follows with the final state s_{t+1} and obs
o_{t+1}, frame and ep_step + 1, `no_action` set and zero action / reward / flags; the
next row of that env starts the next episode at ep_step 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from morphmimic.constants import ACTION_DIM, CTX_DIM, PROPRIO_DIM, STATE_DIM
from morphmimic.model.controller import ControllerState, ControllerRecord

# Rejection-sampling rounds before the exhaustive search for valid rows, the number
# of candidates drawn per missing sample in each round, and the candidates checked at
# once by the exhaustive search.
_MAX_ROUNDS = 8
_OVERSAMPLE = 4
_SEARCH_CHUNK = 1 << 20


@dataclass
class ReplayCfg:
  num_envs: int
  capacity: int = 1000
  device: str = "cpu"
  controller_horizon: int = 0  # zero keeps the legacy allocation and sampling contract


class WindowBatch(NamedTuple):
  """k-step windows t..t+k of one episode, for world-model fitting."""

  state: Tensor  # s_t .. s_{t+k} [B, k+1, 240]
  action: Tensor  # a_t .. a_{t+k-1} [B, k, 29]
  reward: Tensor  # r_{t+1} .. r_{t+k} [B, k]
  terminated: Tensor  # d_{t+1} .. d_{t+k} (bool) [B, k]
  ctx: Tensor  # [B, 33]
  frame: Tensor  # f_t .. f_{t+k} (int64) [B, k+1]
  controller: ControllerRecord | None = None


class StartBatch(NamedTuple):
  """Imagination start states. The previous executed action is the last 29 entries of
  `obs_hist[:, -1]`."""

  state: Tensor  # s_t [B, 240]
  obs_hist: Tensor  # clean obs o_{t-H+1} .. o_t, oldest first [B, H, 93]
  ctx: Tensor  # [B, 33]
  frame: Tensor  # f_t (int64) [B]
  controller: ControllerState | None = None
  reset_clock: Tensor | None = None  # explicit bootstrap of real global reset history


class ReplayBuffer:
  """Per-field tensors [num_envs, capacity, ...]: one dense ring per env.

  Every env has its own write pointer `ptr[e]` and fill level `size[e]` (LongTensor
  [num_envs]). `add_step` writes one row at `ptr[e]` for every env e and advances all
  pointers; `add_terminal_rows(env_ids, ...)` writes one row only for `env_ids` and
  advances only their pointers, so envs with more finished episodes hold more terminal
  rows and their rings wrap earlier. Each env's rows are dense and time ordered modulo
  `capacity`: the newest is at `ptr[e] - 1`, the oldest at `ptr[e] - size[e]`; once
  full the oldest row is overwritten. `valid[e, j]` marks the filled slots
  (j < size[e]); `num_valid_rows = size.sum()`.

  Windows (rows t..t+k) and histories (rows t-H+1..t) run along an env's ring and stay
  within its `size[e]` newest rows, never crossing its write pointer. Rows are
  addressed by their age index a in [0, size[e]) from the oldest row, at slot
  `(ptr[e] - size[e] + a) % capacity`.
  """

  def __init__(self, cfg: ReplayCfg):
    self.cfg = cfg
    n, cap = cfg.num_envs, cfg.capacity
    if n < 1 or cap < 1:
      raise ValueError(f"num_envs and capacity must be >= 1, got {n}, {cap}")
    self.device = torch.device(cfg.device)

    def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> Tensor:
      return torch.zeros(n, cap, *shape, dtype=dtype, device=self.device)

    if not 0 <= cfg.controller_horizon <= cap:
      raise ValueError("controller horizon must fit the global replay clock ring")
    self.controller_before_queue = None
    if cfg.controller_horizon:
      self.controller_before_queue = zeros(11,29)
      self.controller_after_queue = zeros(11,29)
      self.controller_before_meta = zeros(4,dtype=torch.long)
      self.controller_after_meta = zeros(4,dtype=torch.long)
      self.controller_targets = zeros(10,29)
      self.controller_lags = zeros(10,dtype=torch.long)
      self.controller_clock = zeros(dtype=torch.bool)
      self.global_clock = torch.zeros(cap,dtype=torch.bool,device=self.device)
      self.clock_ptr = torch.zeros((),dtype=torch.long,device=self.device)
      self.clock_size = torch.zeros((),dtype=torch.long,device=self.device)
    self.state = zeros(STATE_DIM)
    self.obs = zeros(PROPRIO_DIM)
    self.action = zeros(ACTION_DIM)
    self.reward = zeros()
    self.terminated = zeros(dtype=torch.bool)
    self.timeout = zeros(dtype=torch.bool)
    self.no_action = zeros(dtype=torch.bool)
    self.ctx = zeros(CTX_DIM)
    self.frame = zeros(dtype=torch.long)
    self.ep_step = zeros(dtype=torch.int32)
    self.valid = zeros(dtype=torch.bool)
    self.ptr = torch.zeros(n, dtype=torch.long, device=self.device)
    self.size = torch.zeros(n, dtype=torch.long, device=self.device)
    self._all_envs = torch.arange(n, device=self.device)

  @property
  def num_envs(self) -> int:
    return self.cfg.num_envs

  @property
  def capacity(self) -> int:
    return self.cfg.capacity

  @property
  def num_valid_rows(self) -> int:
    return int(self.size.sum())

  # Writing.

  def add_step(
    self,
    state: Tensor,
    obs: Tensor,
    action: Tensor,
    reward: Tensor,
    terminated: Tensor,
    timeout: Tensor,
    ctx: Tensor,
    frame: Tensor,
    ep_step: Tensor,
    controller: ControllerRecord | None = None,
  ) -> None:
    """One row for every env (leading dim num_envs), with `no_action=False`."""
    values = {
      "state": state,
      "obs": obs,
      "action": action,
      "reward": reward,
      "terminated": terminated,
      "timeout": timeout,
      "ctx": ctx,
      "frame": frame,
      "ep_step": ep_step,
    }
    if self.cfg.controller_horizon:
      if controller is None:
        raise ValueError("controller mode requires real before/after queue and targets")
      values.update(controller_before_queue=controller.before.queue,
                    controller_before_meta=controller.before.meta,
                    controller_after_queue=controller.after.queue,
                    controller_after_meta=controller.after.meta,
                    controller_targets=controller.targets,controller_lags=controller.lags,
                    controller_clock=controller.clock)
    elif controller is not None:
      raise ValueError("controller records require controller-enabled replay")
    self._check_shapes(values, self.num_envs)
    self._append(self._all_envs, values, no_action=False)
    if self.cfg.controller_horizon:
      # Exactly one global tick per control step, independent of terminal row inserts.
      self.global_clock.index_copy_(0,self.clock_ptr.reshape(1),(terminated|timeout).any().reshape(1))
      self.clock_ptr.add_(1).remainder_(self.capacity)
      self.clock_size.add_(1).clamp_(max=self.capacity)

  def add_terminal_rows(
    self,
    env_ids: Tensor,
    state: Tensor,
    obs: Tensor,
    ctx: Tensor,
    frame: Tensor,
    ep_step: Tensor,
  ) -> None:
    """One terminal row (`no_action=True`, zero action / reward / flags) for each env
    of `env_ids` (inputs have leading dim len(env_ids)); other envs are untouched.

    Precondition (not checked, to avoid a host sync): `env_ids` are distinct, as the
    collector's `done.nonzero()` ids are. A repeated id would write one row, not two.
    """
    env_ids = env_ids.to(self.device, torch.long)
    if env_ids.ndim != 1:
      raise ValueError("env_ids must be a 1-D tensor of env indices")
    values = {
      "state": state,
      "obs": obs,
      "ctx": ctx,
      "frame": frame,
      "ep_step": ep_step,
    }
    self._check_shapes(values, env_ids.numel())
    num = env_ids.numel()
    values["action"] = self.action.new_zeros(num, ACTION_DIM)
    values["reward"] = self.reward.new_zeros(num)
    values["terminated"] = self.terminated.new_zeros(num)
    values["timeout"] = self.timeout.new_zeros(num)
    self._append(env_ids, values, no_action=True)

  def add_terminal_rows_masked(
    self, mask: Tensor, state: Tensor, obs: Tensor, ctx: Tensor, frame: Tensor, ep_step: Tensor
  ) -> None:
    """Write terminal rows using a fixed-size mask; nonselected rings remain untouched."""
    if mask.dtype != torch.bool or tuple(mask.shape) != (self.num_envs,):
      raise ValueError("terminal mask must be boolean with one entry per environment")
    mask = mask.to(self.device)
    values = {"state": state, "obs": obs, "ctx": ctx, "frame": frame, "ep_step": ep_step}
    self._check_shapes(values, self.num_envs)
    values.update(action=self.action.new_zeros(self.num_envs, ACTION_DIM),
                  reward=self.reward.new_zeros(self.num_envs),
                  terminated=self.terminated.new_zeros(self.num_envs),
                  timeout=self.timeout.new_zeros(self.num_envs))
    envs, slot = self._all_envs, self.ptr
    for name, value in values.items():
      dst = getattr(self, name)
      selected = mask.reshape(self.num_envs, *([1] * (dst.ndim - 2)))
      dst[envs, slot] = torch.where(selected, value.to(self.device, dst.dtype), dst[envs, slot])
    self.no_action[envs, slot] |= mask
    self.valid[envs, slot] |= mask
    self.ptr.copy_(torch.where(mask, (slot + 1) % self.capacity, slot))
    self.size.add_(mask.long()).clamp_(max=self.capacity)

  def _check_shapes(self, values: dict[str, Tensor], num: int) -> None:
    """Exact shapes (no broadcasting), checked before anything is written."""
    for name, value in values.items():
      expected = (num, *getattr(self, name).shape[2:])
      if tuple(value.shape) != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {tuple(value.shape)}")

  def _append(
    self, env_ids: Tensor, values: dict[str, Tensor], no_action: bool
  ) -> None:
    """Write one row per env of `env_ids` at its pointer and advance it."""
    slot = self.ptr[env_ids]
    for name, value in values.items():
      dst = getattr(self, name)
      dst[env_ids, slot] = value.to(self.device, dst.dtype)
    self.no_action[env_ids, slot] = no_action
    self.valid[env_ids, slot] = True
    self.ptr[env_ids] = (slot + 1) % self.capacity
    self.size[env_ids] = (self.size[env_ids] + 1).clamp(max=self.capacity)

  # Sampling.

  def sample_windows(self, batch: int, k: int, env_range: tuple[int, int] | None = None) -> WindowBatch:
    """`batch` windows of k steps, uniformly over all valid windows: rows t..t+k
    consecutive along their env's ring and within its held rows, one episode
    (`ep_step[t+j] == ep_step[t] + j`), `no_action` false on rows t..t+k-1 (a terminal
    row may only end the window). Raises RuntimeError if there is no valid window."""
    if not 1 <= k < self.capacity:
      raise ValueError(f"k must be in [1, capacity), got {k}")
    if env_range is not None and not 0 <= env_range[0] < env_range[1] <= self.num_envs:
      raise ValueError("Replay environment range must be nonempty and within the buffer")
    offsets = torch.arange(k + 1, device=self.device)

    def accept(env: Tensor, age: Tensor) -> Tensor:
      slots = self._slot(env[:, None], age[:, None] + offsets)
      steps = self.ep_step[env[:, None], slots].long()
      ok = age + k < self.size[env]
      if env_range is not None:
        ok &= (env >= env_range[0]) & (env < env_range[1])
      ok &= (steps == steps[:, :1] + offsets).all(dim=1)
      ok &= ~self.no_action[env[:, None], slots[:, :-1]].any(dim=1)
      return ok

    env, age = self._sample(batch, accept, "windows", env_range=env_range)
    env2 = env[:, None]
    slots = self._slot(env2, age[:, None] + offsets)
    return WindowBatch(
      state=self.state[env2, slots],
      action=self.action[env2, slots[:, :-1]],
      reward=self.reward[env2, slots[:, :-1]],
      terminated=self.terminated[env2, slots[:, :-1]],
      ctx=self.ctx[env, slots[:, 0]],
      frame=self.frame[env2, slots],
      controller=self.controller_windows(env2,slots[:,:-1]),
    )

  def sample_starts(
    self,
    batch: int,
    hist_len: int,
    accept_frame: Callable[[Tensor], Tensor] | None = None,
  ) -> StartBatch:
    """`batch` start rows, uniformly over the held rows with `no_action` false whose
    history is still held, with their clean obs history: rows t-H+1..t, clamped at
    the episode's first row (row t - ep_step[t] along the env's ring) by repeating its
    obs, the rule of `ObsHistory.reset` + `push`. Starts whose history reaches rows
    already overwritten are rejected, and so are rows whose motion frame fails
    `accept_frame` (frames [N] -> bool [N]) when it is given."""
    self.check_controller_clock()
    if not 1 <= hist_len <= self.capacity:
      raise ValueError(f"hist_len must be in [1, capacity], got {hist_len}")
    offsets = torch.arange(1 - hist_len, 1, device=self.device)

    def history_ages(env: Tensor, age: Tensor) -> Tensor:
      episode_start = age - self.ep_step[env, self._slot(env, age)].long()
      return torch.maximum(age[:, None] + offsets, episode_start[:, None])

    def accept(env: Tensor, age: Tensor) -> Tensor:
      ok = age < self.size[env]
      ok &= ~self.no_action[env, self._slot(env, age)]
      if accept_frame is not None:
        ok &= accept_frame(self.frame[env, self._slot(env, age)])
      return ok & (history_ages(env, age)[:, 0] >= 0)

    env, age = self._sample(batch, accept, "start rows")
    slot = self._slot(env, age)
    hist_slots = self._slot(env[:, None], history_ages(env, age))
    return StartBatch(
      state=self.state[env, slot],
      obs_hist=self.obs[env[:, None], hist_slots],
      ctx=self.ctx[env, slot],
      frame=self.frame[env, slot],
      controller=self.controller_start(env,slot),
      reset_clock=self.sample_controller_clock(batch),
    )

  def controller_start(self, env, slot):
    if not self.cfg.controller_horizon:
      return None
    return ControllerState(self.controller_before_queue[env,slot],self.controller_before_meta[env,slot])

  def controller_windows(self, env, slots):
    if not self.cfg.controller_horizon:
      return None
    return ControllerRecord(
      ControllerState(self.controller_before_queue[env,slots],self.controller_before_meta[env,slots]),
      ControllerState(self.controller_after_queue[env,slots],self.controller_after_meta[env,slots]),
      self.controller_targets[env,slots],self.controller_lags[env,slots],self.controller_clock[env,slots])

  def clear_controller_clock(self):
    """A manual all-world reset breaks the observed contiguous control timeline."""
    if self.cfg.controller_horizon:
      self.clock_ptr.zero_()
      self.clock_size.zero_()

  def check_controller_clock(self):
    if self.cfg.controller_horizon and int(self.clock_size) < self.cfg.controller_horizon:
      raise RuntimeError("replay has insufficient real global reset clock history")

  def sample_controller_clock(self, batch):
    if not self.cfg.controller_horizon:
      return None
    # Availability is checked by the caller outside capture. All indices and changing
    # ring counts stay device tensors so captured sampling never freezes the inventory.
    h=self.cfg.controller_horizon
    ages=torch.arange(self.capacity,device=self.device)
    valid=ages+h<=self.clock_size
    start=torch.multinomial(valid.float(),batch,replacement=True)
    slots=(self.clock_ptr-self.clock_size+start[:,None]+torch.arange(h,device=self.device))%self.capacity
    return self.global_clock[slots]

  def _slot(self, env: Tensor, age: Tensor) -> Tensor:
    """Slot of each env's row `age` rows after its oldest held row."""
    return (self.ptr[env] - self.size[env] + age) % self.capacity

  def _sample(
    self, batch: int, accept: Callable[[Tensor, Tensor], Tensor], what: str,
    env_range: tuple[int, int] | None = None,
  ) -> tuple[Tensor, Tensor]:
    """`batch` (env, age) pairs drawn uniformly from the held rows `accept` keeps:
    rejection sampling of candidates uniform over all held rows (env drawn in
    proportion to `size`, age uniform in [0, size[env])), falling back to checking
    every row when acceptance is too rare."""
    if batch < 1:
      raise ValueError(f"batch must be >= 1, got {batch}")
    # An empty buffer draws from all-ones weights (multinomial rejects all zeros):
    # every candidate is then rejected and the exhaustive search raises. No host sync.
    allowed = torch.ones_like(self.size, dtype=torch.bool)
    if env_range is not None:
      allowed = (self._all_envs >= env_range[0]) & (self._all_envs < env_range[1])
    weights = self.size.float() * allowed
    weights += (weights.sum() == 0).float() * allowed
    envs: list[Tensor] = []
    ages: list[Tensor] = []
    missing = batch
    for _ in range(_MAX_ROUNDS):
      num = _OVERSAMPLE * missing + 256
      env = torch.multinomial(weights, num, replacement=True)
      size = self.size[env]
      age = (torch.rand(num, device=self.device) * size).long()
      age = age.minimum((size - 1).clamp(min=0))
      keep = accept(env, age).nonzero().squeeze(-1)[:missing]
      envs.append(env[keep])
      ages.append(age[keep])
      missing -= keep.numel()
      if missing == 0:
        break
    if missing > 0:
      env, age = self._search(accept)
      if env.numel() == 0:
        raise RuntimeError(f"replay buffer holds no valid {what}")
      pick = torch.randint(env.numel(), (missing,), device=self.device)
      envs.append(env[pick])
      ages.append(age[pick])
    return torch.cat(envs), torch.cat(ages)

  def _search(
    self, accept: Callable[[Tensor, Tensor], Tensor]
  ) -> tuple[Tensor, Tensor]:
    """All held (env, age) pairs that `accept` keeps (it rejects ages >= size),
    checked in chunks of whole envs."""
    cap = self.capacity
    all_ages = torch.arange(cap, device=self.device)
    envs_per_chunk = max(1, _SEARCH_CHUNK // cap)
    envs: list[Tensor] = []
    ages: list[Tensor] = []
    for start in range(0, self.num_envs, envs_per_chunk):
      chunk = self._all_envs[start : start + envs_per_chunk]
      env = chunk.repeat_interleave(cap)
      age = all_ages.repeat(chunk.numel())
      ok = accept(env, age)
      envs.append(env[ok])
      ages.append(age[ok])
    return torch.cat(envs), torch.cat(ages)
