"""`ManagerBasedRlEnv` whose rewards and terminations see the post-step state."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv, types

from morphmimic.env.commands import MultiMotionCommand
from morphmimic.env.physics import PhysicsDiagnostics


class MorphMimicEnv(ManagerBasedRlEnv):
  """mjlab 1.6.0's `ManagerBasedRlEnv` with two changes in `step()`: right after the
  decimation loop it runs `sim.forward()` and refreshes the motion command's relative
  body poses, before terminations and rewards; the later `sim.forward()` only runs when
  envs were auto-reset.

  mjlab computes them on kinematics one physics substep stale and on relative body
  poses built from the previous step's robot anchor. With the refresh, the reward of
  the transition (s_t, a_t) -> s_{t+1} equals `rewards_torch.tracking_reward(s_{t+1},
  ref[f_t], a_t, a_{t-1})` on the logged states, where f_t is the frame the command
  showed at time t; the command then advances to f_t + 1.
  """

  motion_command_name = "motion"

  def __init__(self, *args, graph_stream=None, **kwargs):
    import warp as wp
    self.tensor_step_stream = graph_stream
    self._reset_forward_graph = self._reset_forward_signature = None
    if graph_stream is None:
      super().__init__(*args, **kwargs)
      self.physics_diagnostics = PhysicsDiagnostics.from_native(self.sim.wp_data, self.sim.wp_model.is_sparse)
    else:
      wp.init()
      from morphmimic.env.graph_rewards import prepare_graph_rewards
      cfg = args[0] if args else kwargs['cfg']
      graph_stream.wait_stream(torch.cuda.current_stream(graph_stream.device))
      with torch.cuda.stream(graph_stream), wp.ScopedStream(wp.stream_from_torch(graph_stream)):
        prepare_graph_rewards(cfg, graph_stream.device)
        super().__init__(*args, **kwargs)
        from morphmimic.env.graph_delay import prepare_graph_delays
        prepare_graph_delays(self.scene)
        self.sim.use_cuda_graph = False
        self.sim.create_graph()
        self.physics_diagnostics = PhysicsDiagnostics.from_native(self.sim.wp_data, self.sim.wp_model.is_sparse)
      torch.cuda.current_stream(graph_stream.device).wait_stream(graph_stream)

  def _sim_array_signature(self):
    """Identity of all top-level native arrays captured by the simulator."""
    import warp as wp
    return tuple((scope, name, value.ptr, value.shape, str(value.dtype))
                 for scope, obj in (("model", self.sim.wp_model), ("data", self.sim.wp_data))
                 for name, value in vars(obj).items() if isinstance(value, wp.array))

  def reset(self, *args, **kwargs):
    if self.tensor_step_stream is None:
      result = super().reset(*args, **kwargs)
      if getattr(self, 'physics_diagnostics', None) is not None:
        self.physics_diagnostics.sample_forward()
      return result
    import warp as wp
    import mujoco_warp as mjwarp
    current = torch.cuda.current_stream(self.tensor_step_stream.device)
    self.tensor_step_stream.wait_stream(current)
    warp_stream = wp.stream_from_torch(self.tensor_step_stream)
    with torch.cuda.stream(self.tensor_step_stream), wp.ScopedStream(warp_stream):
      if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Dynamic reset must run outside CUDA capture')
      signature = self._sim_array_signature()
      if self._reset_forward_signature is not None and signature != self._reset_forward_signature:
        raise RuntimeError('Simulator arrays changed; recreate the environment and captured graphs')
      if self._reset_forward_graph is None:
        with wp.ScopedCapture(stream=warp_stream) as capture:
          mjwarp.forward(self.sim.wp_model, self.sim.wp_data)
        self._reset_forward_graph = capture.graph
        self._reset_forward_signature = signature
      if self.sim.use_cuda_graph:
        raise RuntimeError('Unexpected native simulator graph mode during graph reset')
      self.sim.forward_graph = self._reset_forward_graph
      self.sim.use_cuda_graph = True
      try:
        result = super().reset(*args, **kwargs)
        if getattr(self, 'physics_diagnostics', None) is not None:
          self.physics_diagnostics.sample_forward()
      finally:
        self.sim.use_cuda_graph = False
    current.wait_stream(self.tensor_step_stream)
    return result

  def reset_native_masked(self, mask: torch.Tensor) -> None:
    """Reset native simulation fields for a fixed-shape world mask on the active stream."""
    if mask.dtype != torch.bool or tuple(mask.shape) != (self.num_envs,) or mask.device != torch.device(self.device):
      raise ValueError("native reset mask must be a device boolean per environment")
    import mujoco_warp as mjwarp
    self.sim._reset_mask.copy_(mask)
    mjwarp.reset_data(self.sim.wp_model, self.sim.wp_data, reset=self.sim._reset_mask_wp)

  def validate_tensor_step(self) -> None:
    """Check host-side prerequisites before each no-reset tensor step or replay."""
    if self.cfg.auto_reset:
      raise ValueError("Tensor step requires explicit resets")
    if set(self.event_manager.available_modes) & {"step", "interval"}:
      raise ValueError("Tensor step does not support step or interval events")
    if self.recorder_manager.active_terms:
      raise ValueError("Tensor step does not support recorders")
    if self.metrics_manager.active_terms:
      raise ValueError("Tensor step does not support replaying Python-backed metrics")
    if self.sim.nan_guard.enabled:
      raise ValueError("Tensor step does not support the simulator NaN guard")
    if any(group is not None and group.nan_policy != "disabled" for group in self.cfg.observations.values()):
      raise ValueError("Tensor step does not support observation NaN policies")
    if self.command_manager.active_terms != [self.motion_command_name]:
      raise ValueError("Tensor step requires the single motion command")
    motion = self.command_manager.get_term(self.motion_command_name)
    if motion._pending_forward or not bool((motion.time_left > self.step_dt).all()):
      raise ValueError("Tensor step requires an unexpired motion command")
    if bool(self._manual_reset_pending.any()):
      raise ValueError("Tensor step requires reset after a finished episode")

  def step_no_reset_tensor(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """Fixed-shape compute; caller validates before every capture/replay and resets finished worlds."""
    self.extras["log"] = {}
    controller_recording = getattr(self, "controller_recording", None)
    if controller_recording is not None:
      controller_recording.begin()
    action_mask = self.action_manager.get_term("joint_pos").joint_mask
    self.action_manager.process_action(action.to(self.device) * action_mask)
    for substep in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      if controller_recording is not None:
        controller_recording.substep(substep)
      self.sim.step()
      self.physics_diagnostics.sample()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()
    if controller_recording is not None:
      controller_recording.finish()
    self.sim.forward()
    self.physics_diagnostics.sample_forward()
    motion = self.command_manager.get_term(self.motion_command_name)
    assert isinstance(motion, MultiMotionCommand)
    motion.update_relative_body_poses()
    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()
    motion._update_metrics()
    motion.time_left -= self.step_dt
    motion._update_command(None)
    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    self._manual_reset_pending.logical_or_(self.reset_buf)
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    # Body copied from mjlab 1.6.0 ManagerBasedRlEnv.step; only the blocks marked
    # "MorphMimic" differ.
    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
      pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      raise RuntimeError(
        f"Environments {pending_ids.cpu().tolist()} must be reset via "
        "reset(env_ids=...) before calling step() again when auto_reset=False."
      )

    self.extras["log"] = {}
    controller_recording = getattr(self, "controller_recording", None)
    if controller_recording is not None:
      controller_recording.begin()
    action_mask = self.action_manager.get_term("joint_pos").joint_mask
    self.action_manager.process_action(action.to(self.device) * action_mask)

    for substep in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      if controller_recording is not None:
        controller_recording.substep(substep)
      self.sim.step()
      self.physics_diagnostics.sample()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()
    if controller_recording is not None:
      controller_recording.finish()

    # MorphMimic: post-step kinematics and relative body poses for terminations and
    # rewards.
    self.sim.forward()
    self.physics_diagnostics.sample_forward()
    motion = self.command_manager.get_term(self.motion_command_name)
    assert isinstance(motion, MultiMotionCommand)
    motion.update_relative_body_poses()

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations and compute rewards.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # Events fire before auto-reset, on the terminal state, matching the
    # explicit flow.
    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_pre_reset(reset_env_ids)
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # MorphMimic: mjlab's second forward() only matters for freshly auto-reset envs;
    # the forward() above already refreshed every other env. (The cfg has no step or
    # interval events that would write state after it.)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.sim.forward()
      self.physics_diagnostics.sample_forward()

    # Pass dt=0 for freshly reset envs so their command timers start full,
    # matching reset().
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      command_dt = self._command_dt
      command_dt.fill_(self.step_dt)
      command_dt[reset_env_ids] = 0.0
      self.command_manager.compute(dt=command_dt)
    else:
      self.command_manager.compute(dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_post_reset(reset_env_ids)
    elif len(reset_env_ids) > 0:
      self._manual_reset_pending[reset_env_ids] = True

    self.recorder_manager.record_post_step()

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )
