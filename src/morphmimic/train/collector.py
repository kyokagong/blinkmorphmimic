"""Simulator data collection (Algorithm A.1 step 1).

The current policy acts on the stacked noisy `policy` obs (what the deployed policy
sees) and the goals (`GOAL_GROUPS`: the motion goal and the keypoint goal); the env
executes the action plus N(0, σ²) noise; every step is logged to the replay with the
clean obs (imagination re-adds fresh sensor noise). The env runs with auto_reset=False:
after a step, finished envs get a terminal row with the physical terminal state the step
returned, then are reset by the collector.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from morphmimic.constants import HIST_LEN, PROPRIO_DIM, R_MAX
from morphmimic.env.actions import CanonicalJointPositionAction
from morphmimic.env.commands import MultiMotionCommand
from morphmimic.env.morphmimic_env import MorphMimicEnv
from morphmimic.train.obs_history import ObsHistory
from morphmimic.train.replay import ReplayBuffer

GOAL_GROUPS = ("goal", "goal_kp")
"""Goal observation groups the policy function is given, keyed by group name."""
OBS_GROUPS = ("policy", "policy_clean", *GOAL_GROUPS, "state", "context")

PolicyFn = Callable[[Tensor, dict[str, Tensor]], Tensor]
"""(obs history [N, H*93] as `ObsHistory.flat`, goals {group: [N, dim]}) -> action."""


class Collector:
  """Steps `env` with `policy_fn(obs_hist_flat [N, H*93], goals) -> [N, 29]` (called
  under no_grad; `goals` maps each of `GOAL_GROUPS` to its obs: "goal" [N, 670] and
  "goal_kp" [N, 1684], the raw human poses then the robot pelvis quaternion) plus
  N(0, action_noise_std²) noise and logs to `replay`.

  Per step, before `env.step`, it reads the frame f_t = `command.global_frame`; row t
  of each env holds the `state` and `policy_clean` obs it acted on, the executed
  action, the reward / failure of the step, a time-out flag (`truncated` and not
  `terminated`), the `context` obs, f_t and its episode step. Envs that finished then
  get a terminal row with the returned (terminal) `state` / `policy_clean` / `context`,
  the post-step frame and episode step + 1; then they are reset and their history is
  refilled with the post-reset noisy obs.

  `on_reset(env_ids)`, if given, is called with the ids of every reset the collector
  performs (all envs at the end of `reset_all`, the finished ones at the end of a
  step's reset), once the reset is complete and before the next `policy_fn` call: the
  hook for per-episode state of the policy function.

  `obs` holds the current observation groups (what the next step acts on), `history`
  the stacked noisy obs, `ep_step` / `ep_return` the per-env step count and reward sum
  of the running episodes.
  """

  def __init__(
    self,
    env: MorphMimicEnv,
    replay: ReplayBuffer,
    policy_fn: PolicyFn,
    action_noise_std: float = 0.1,
    hist_len: int = HIST_LEN,
    on_reset: Callable[[Tensor], None] | None = None,
    det_frac: float = 0.0,
    clean_action_record: bool = False,
    reward_max: float = R_MAX,
    physics_graph: bool = False,
    masked_reset: bool = False,
    on_reset_masked: Callable[[Tensor], None] | None = None,
  ):
    """`det_frac`: the first round(det_frac * N) envs act without the exploration noise
    (`--collect-det-frac`), so the replay also holds the states the deterministic
    policy reaches. `clean_action_record` (`--clean-action-record`): the env's action
    term executes a + eps and records the clean a (mjlab's `last_action`, so the
    `prev_action` observation and the action-rate penalty), and the replay logs the
    executed action; off, the env sees and records the noisy action."""
    if replay.num_envs != env.num_envs:
      raise ValueError(
        f"replay holds {replay.num_envs} envs, the env has {env.num_envs}"
      )
    self.env = env
    self.reward_max = reward_max
    self.replay = replay
    self.policy_fn = policy_fn
    self.action_noise_std = action_noise_std
    self.on_reset = on_reset
    self.noise_scale = torch.ones(env.num_envs, 1, device=env.device)
    self.action_term: CanonicalJointPositionAction | None = None
    self.set_exploration(det_frac, clean_action_record)
    command = env.command_manager.get_term(env.motion_command_name)
    assert isinstance(command, MultiMotionCommand)
    self.command = command
    n, device = env.num_envs, env.device
    self.history = ObsHistory(n, hist_len, PROPRIO_DIM, device)
    self.ep_step = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_return = torch.zeros(n, device=device)
    self.obs: dict[str, Tensor] = {}
    self._physics_graph = None
    self._masked_reset = None
    self.on_reset_masked = on_reset_masked
    self._graph_warmed = False
    if masked_reset and not physics_graph:
      raise ValueError("Masked reset requires a physics graph")
    if masked_reset and on_reset is not None and on_reset_masked is None:
      raise ValueError("Masked reset requires a mask-aware reset callback")
    if physics_graph:
      if torch.device(env.device).type != "cuda":
        raise ValueError("Collection graphs require a CUDA environment")
      if getattr(env, "tensor_step_stream", None) is None:
        raise ValueError("Collection graphs require an environment constructed with graph_stream")
      from morphmimic.env.control_graph import PhysicsGraph
      self._physics_graph = PhysicsGraph(env)
      if masked_reset:
        from morphmimic.env.masked_reset import MaskedResetGraph
        self._masked_reset = MaskedResetGraph(env)

  def refresh_reset_sampler(self) -> None:
    """Refresh cached online probabilities, or recapture the legacy reset sampler."""
    if getattr(getattr(self, 'command', None), 'importance_state', None) is not None:
      self.command.refresh_importance_probabilities()
    elif self._masked_reset is not None:
      self._masked_reset.reset()

  def set_exploration(self, det_frac: float, clean_action_record: bool) -> None:
    """Sets which envs explore (`noise_scale`: 0 for the first round(det_frac * N))
    and where the noise is added (the constructor's `det_frac` and
    `clean_action_record`); a resume applies a checkpoint's values."""
    if not 0.0 <= det_frac <= 1.0:
      raise ValueError(f"det_frac must be in [0, 1], got {det_frac}")
    n, device = self.env.num_envs, self.env.device
    num_det = round(det_frac * n)
    self.noise_scale = (torch.arange(n, device=device) >= num_det).float()[:, None]
    term = self.env.action_manager.get_term("joint_pos")
    assert isinstance(term, CanonicalJointPositionAction)
    if clean_action_record:
      term.noise_std = self.action_noise_std
      term.noise_scale.copy_(self.noise_scale)
      self.action_term = term
    else:
      term.noise_std = 0.0
      term.noise_scale.fill_(1.0)
      self.action_term = None

  def reset_all(self) -> None:
    """Reset every env, refill the histories and restart the episode counters. An
    episode cut short here gets no terminal row (its last rows just end; the next row
    starts at ep_step 0, so windows and histories do not cross it)."""
    if self._physics_graph is not None:
      self._physics_graph.reset()
    if self._masked_reset is not None:
      self._masked_reset.reset()
    if self._physics_graph is not None and not self._graph_warmed:
      self._warmup_graph()
      self._graph_warmed = True
    obs, _ = self.env.reset()
    if self._physics_graph is not None:
      self._check_graph_overflow('reset')
    self.obs = {group: obs[group].clone() for group in OBS_GROUPS}
    self.history.reset(None, self.obs["policy"])
    self.ep_step.zero_()
    self.ep_return.zero_()
    self.replay.clear_controller_clock()
    if self._masked_reset is not None and self.on_reset_masked is not None:
      self.on_reset_masked(torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device))
    elif self.on_reset is not None:
      self.on_reset(torch.arange(self.env.num_envs, device=self.env.device))

  def _warmup_graph(self) -> None:
    """Preflight physics without counting or folding its temporary episode."""
    if 'reset' in self.env.event_manager.available_modes:
      raise ValueError('Graph warmup does not support reset-event side effects')
    names = ('G_b', 'past_softmax', 'global_frame', 'start_bin', 'disc_return',
             'gamma_pow', 'horizon_pow', 'episode_steps', 'time_left', 'command_counter',
             'restart_frame', 'consecutive_local')
    command_state = {name: getattr(self.command, name).clone() for name in names}
    importance = self.command.importance_sampling_state_dict() if getattr(getattr(self, 'command', None), 'importance_state', None) is not None else None
    metrics = {name: value.clone() for name, value in self.command.metrics.items()}
    manager = self.env.termination_manager
    terminated = manager._terminated_buf.clone()
    truncated = manager._truncated_buf.clone()
    terms = {name: value.clone() for name, value in manager._term_dones.items()}
    rewards = {name: value.clone() for name, value in self.env.reward_manager._episode_sums.items()}
    pending_forward = self.command._pending_forward
    counters = self.env._sim_step_counter, self.env.common_step_counter
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(self.env.device)
    try:
      self.env.reset()
      self._physics_graph.warmup(torch.zeros(self.env.num_envs, 29, device=self.env.device))
    finally:
      if importance is not None:
        self.command.load_importance_sampling_state_dict(importance)
      for name, value in command_state.items():
        getattr(self.command, name).copy_(value)
      for name, value in metrics.items():
        self.command.metrics[name].copy_(value)
      manager._terminated_buf.copy_(terminated)
      manager._truncated_buf.copy_(truncated)
      for name, value in terms.items():
        manager._term_dones[name].copy_(value)
      for name, value in rewards.items():
        self.env.reward_manager._episode_sums[name].copy_(value)
      self.command._pending_forward = pending_forward
      self.env._sim_step_counter, self.env.common_step_counter = counters
      torch.random.set_rng_state(cpu_rng)
      torch.cuda.set_rng_state(cuda_rng, self.env.device)

  def _check_graph_overflow(self, stage: str) -> None:
    diagnostics = self.env.physics_diagnostics
    if bool(diagnostics.bits_tensor() != 0):
      raise RuntimeError(f'Native physics overflow after {stage}: {diagnostics.report()}')

  @torch.no_grad()
  def step(self, n: int) -> dict[str, float | int]:
    """Run n env steps for all envs. Returns stats of these steps: `mean_reward` per
    env step, `episodes_finished`, `failures` (terminated episodes), and over the
    finished episodes `mean_episode_length` (rewarded steps) and `mean_norm_return`
    (Σr / (R_MAX · length)); NaN when no episode finished. `steps` = n · num_envs."""
    if not self.obs:
      raise RuntimeError("call reset_all() before step()")
    num_envs = self.env.num_envs
    device = self.env.device
    reward_sum = torch.zeros((), device=device)
    length_sum = torch.zeros((), dtype=torch.long, device=device)
    norm_return_sum = torch.zeros((), device=device)
    failures = torch.zeros((), dtype=torch.long, device=device)
    finished = torch.zeros((), dtype=torch.long, device=device) if self._masked_reset else 0

    for _ in range(n):
      obs = self.obs
      frame = self.command.global_frame.clone()
      goals = {group: obs[group] for group in GOAL_GROUPS}
      action = self.policy_fn(self.history.flat(), goals)
      if self.action_term is None:  # The env sees (and records) the noisy action.
        action = action + torch.randn_like(action) * (
          self.action_noise_std * self.noise_scale
        )
      if self._physics_graph is None:
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
      else:
        if not bool(torch.isfinite(action).all()):
          raise RuntimeError("Nonfinite collection action before physics graph")
        next_obs, reward, terminated, truncated, _ = self._physics_graph.step(action)
        self._check_graph_overflow('physics step')
      executed = self.env.action_manager.action if self.action_term is None else self.action_term.executed
      # The env reuses its buffers; keep this step's values past the reset below.
      reward, terminated = reward.clone(), terminated.clone()
      timeout = truncated & ~terminated
      controller_record = None
      recorder = getattr(self.env, "controller_recording", None)
      if recorder is not None:
        from morphmimic.model.controller import ControllerRecord
        controller_record = ControllerRecord(recorder.before,recorder.after,recorder.targets,
          recorder.lags,(terminated|timeout).any().expand(num_envs))
      self.replay.add_step(
        state=obs["state"],
        obs=obs["policy_clean"],
        action=executed,
        reward=reward,
        terminated=terminated,
        timeout=timeout,
        ctx=obs["context"],
        frame=frame,
        ep_step=self.ep_step,
        controller=controller_record,
      )
      self.command.library.record_usage("collected", frame)
      self.ep_step += 1
      self.ep_return += reward
      reward_sum += reward.sum()
      self.history.push(next_obs["policy"])
      self.obs = {group: next_obs[group].clone() for group in OBS_GROUPS}

      if self._masked_reset is not None:
        mask = terminated | timeout
        self.replay.add_terminal_rows_masked(
          mask, state=self.obs["state"], obs=self.obs["policy_clean"],
          ctx=self.obs["context"], frame=self.command.global_frame, ep_step=self.ep_step,
        )
        finished += mask.sum()
        length_sum += torch.where(mask, self.ep_step, 0).sum()
        norm_return_sum += torch.where(
          mask, self.ep_return / (self.reward_max * self.ep_step.clamp_min(1)), 0
        ).sum()
        failures += terminated.sum()
        reset_obs = self._masked_reset(mask)
        self._check_graph_overflow('masked reset')
        for group in OBS_GROUPS:
          selection = mask.reshape(num_envs, *([1] * (self.obs[group].ndim - 1)))
          self.obs[group] = torch.where(selection, reset_obs[group], self.obs[group])
        self.history.reset_masked(mask, self.obs["policy"])
        self.ep_step.masked_fill_(mask, 0)
        self.ep_return.masked_fill_(mask, 0)
        if self.on_reset_masked is not None:
          self.on_reset_masked(mask)
        continue

      done_ids = (terminated | timeout).nonzero().squeeze(-1)
      if done_ids.numel() == 0:
        continue
      self.replay.add_terminal_rows(
        done_ids,
        state=self.obs["state"][done_ids],
        obs=self.obs["policy_clean"][done_ids],
        ctx=self.obs["context"][done_ids],
        frame=self.command.global_frame[done_ids],
        ep_step=self.ep_step[done_ids],
      )
      length = self.ep_step[done_ids]
      finished += done_ids.numel()
      length_sum += length.sum()
      norm_return_sum += (self.ep_return[done_ids] / (self.reward_max * length)).sum()
      failures += terminated.sum()

      reset_obs, _ = self.env.reset(env_ids=done_ids)
      if self._physics_graph is not None:
        self._check_graph_overflow('partial reset')
      for group in OBS_GROUPS:
        self.obs[group][done_ids] = reset_obs[group][done_ids]
      self.history.reset(done_ids, self.obs["policy"][done_ids])
      self.ep_step[done_ids] = 0
      self.ep_return[done_ids] = 0.0
      if self.on_reset is not None:
        self.on_reset(done_ids)

    steps = n * num_envs
    if isinstance(finished, Tensor):
      finished = int(finished.item())
    nan = float("nan")
    return {
      "mean_reward": reward_sum.item() / steps if steps else nan,
      "episodes_finished": finished,
      "mean_episode_length": length_sum.item() / finished if finished else nan,
      "mean_norm_return": norm_return_sum.item() / finished if finished else nan,
      "failures": int(failures.item()),
      "steps": steps,
    }
