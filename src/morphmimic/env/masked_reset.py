"""Fixed-shape reset for the validated graph-collection environment.

Random proposals are generated for every world; only selected worlds receive the
proposals. This preserves the sampling distribution, with a different RNG sequence
from a reset that draws only for selected worlds.
"""
import math
import torch
from morphmimic.train.probe_focus import mix_probe_focus
from morphmimic.env.commands import fold_returns, return_sampling_probs


def fold_finished_masked(g, past, bins, returns, finished, mask, alpha, temp, mix, cap):
  """Fold only selected episodes and update sampler EMA only when a reset occurs."""
  updated = fold_returns(g, bins, returns, finished * mask.to(finished), alpha)
  _, softmax = return_sampling_probs(updated, past, temp, mix, cap)
  next_past = (1.0 - alpha) * past + alpha * softmax
  return updated, torch.where(mask.any(), next_past, past)


def sample_start_candidates(command):
  """Draw full-batch clip/frame proposals without mutating command or episode state."""
  cfg, lib, n = command.cfg, command.library, command.num_envs
  importance = getattr(command, 'importance_state', None)
  restart = torch.full_like(command.global_frame, -1)
  streak = command.consecutive_local
  if importance is None and cfg.failure_reset > 0.0:
    failed = command._env.termination_manager.terminated & (command.episode_steps >= cfg.failure_min_steps)
    local = (failed & (streak < cfg.failure_max_consecutive)
             & (torch.rand(n, device=command.device) < cfg.failure_reset))
    jitter = torch.randint(-cfg.failure_jitter, cfg.failure_jitter + 1, (n,), device=command.device)
    start = lib.clip_start[lib.frame_clip[command.global_frame]]
    candidate = torch.maximum(command.global_frame - cfg.failure_lookback + jitter, start)
    restart = torch.where(local, candidate, restart)
    streak = torch.where(local, streak + 1, torch.zeros_like(streak))
  entropy = None
  if cfg.sampling == 'start':
    frames = lib.clip_start[command.env_clip]
    bins = lib.frame_bin(frames)
  elif cfg.sampling == 'uniform':
    frames = torch.randint(0, lib.num_frames, (n,), device=command.device)
    bins = lib.frame_bin(frames)
    entropy = torch.ones((), device=command.device)
  elif cfg.sampling == 'return':
    if importance is not None:
      probs = importance['probabilities']
    else:
      probs, _ = return_sampling_probs(command.G_b, command.past_softmax, cfg.softmax_temp, cfg.mix, cfg.softmax_cap)
      probs = mix_probe_focus(probs, command.probe_probabilities, command.probe_rho)
    bins = torch.multinomial(probs, n, replacement=True)
    frames = lib.sample_frames_in_bins(bins)
    from morphmimic.train.recovery import sample_recovery_frames
    if importance is None and getattr(command, 'recovery', None) is not None:
      frames = sample_recovery_frames(command, frames)
      bins = lib.frame_bin(frames)
    entropy = -(probs * (probs + 1e-12).log()).sum()
    entropy = entropy / math.log(lib.num_bins) if lib.num_bins > 1 else entropy.new_tensor(1.0)
  else:
    raise ValueError(f'Unknown sampling mode {cfg.sampling!r}')
  local = restart >= 0
  bins = torch.where(local, lib.frame_bin(restart.clamp(min=0)), bins)
  frames = torch.where(local, restart, frames)
  return torch.minimum(frames, lib.clip_end(frames) - 1), bins, streak, entropy


def write_reference_masked(command, mask, root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
  """Write proposed native reference states only for selected worlds."""
  from mjlab.utils.lab_api.math import quat_apply_inverse
  if mask.dtype != torch.bool or mask.ndim != 1:
    raise ValueError('reference reset mask must be a boolean vector')
  robot = command.robot
  indexing = robot.indexing
  sim = robot.data.data
  selected = mask[:, None]
  limits = robot.data.soft_joint_pos_limits
  joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
  root_pose = torch.cat((root_pos, root_quat), dim=-1)
  root_velocity = torch.cat((root_lin_vel, quat_apply_inverse(root_quat, root_ang_vel)), dim=-1)
  for array, index, value in ((sim.qpos, indexing.free_joint_q_adr, root_pose),
                              (sim.qpos, indexing.joint_q_adr, joint_pos),
                              (sim.qvel, indexing.free_joint_v_adr, root_velocity),
                              (sim.qvel, indexing.joint_v_adr, joint_vel)):
    array[:, index] = torch.where(selected, value, array[:, index])


def reset_action_manager_masked(manager, mask):
  """Reset known action-history buffers without dynamic environment indices."""
  for value in (manager._prev_action, manager._prev_prev_action, manager._action,
                *(term._raw_actions for term in manager._terms.values())):
    value.masked_fill_(mask.reshape(mask.shape[0], *([1] * (value.ndim - 1))), 0)


def reset_reward_manager_masked(manager, mask):
  """Reset known episodic reward sums and retain selected-world averages."""
  if manager._class_term_cfgs:
    raise ValueError('masked reset does not support class-based reward terms')
  count = mask.sum().clamp_min(1)
  extras = {}
  for name, value in manager._episode_sums.items():
    extras['Episode_Reward/' + name] = torch.where(mask, value, 0).sum() / count / manager._env.max_episode_length_s
    value.masked_fill_(mask, 0)
  return extras


def reset_command_bookkeeping_masked(command, mask):
  """Apply selected command-accounting resets and return full-batch frame proposals."""
  from morphmimic.env.commands import normalized_return
  if mask.dtype != torch.bool or tuple(mask.shape) != (command.num_envs,) or mask.device != torch.device(command.device):
    raise ValueError('command reset mask must be a device boolean per environment')
  cfg = command.cfg
  if getattr(command, 'importance_state', None) is not None:
    command.record_importance_failures(mask)
  else:
    norm_pow = command.horizon_pow if cfg.reset_norm == 'remaining' else command.gamma_pow
    returns = normalized_return(command.disc_return, norm_pow, cfg.gamma, cfg.r_max).clamp(min=0.0)
    finished = (command.episode_steps > 0).float()
    g, past = fold_finished_masked(command.G_b, command.past_softmax, command.start_bin,
                                   returns, finished, mask, cfg.ema_alpha, cfg.softmax_temp,
                                   cfg.mix, cfg.softmax_cap)
    command.G_b.copy_(g)
    command.past_softmax.copy_(past)
  frames, bins, streak, entropy = sample_start_candidates(command)
  for value in command.metrics.values():
    value.masked_fill_(mask, 0)
  if entropy is not None:
    value = command.metrics['sampling_entropy']
    value.copy_(torch.where(mask.any(), entropy.expand_as(value), value))
  command.disc_return.masked_fill_(mask, 0)
  command.gamma_pow.masked_fill_(mask, 1)
  command.episode_steps.masked_fill_(mask, 0)
  command.command_counter.copy_(torch.where(mask, torch.ones_like(command.command_counter), command.command_counter))
  low, high = cfg.resampling_time_range
  candidate_time = torch.empty_like(command.time_left).uniform_(low, high)
  command.time_left.copy_(torch.where(mask, candidate_time, command.time_left))
  command.global_frame.copy_(torch.where(mask, frames, command.global_frame))
  command.start_bin.copy_(torch.where(mask, bins, command.start_bin))
  horizon = command.library.remaining(frames).clamp(max=command._env.max_episode_length)
  command.horizon_pow.copy_(torch.where(mask, cfg.gamma ** horizon.float(), command.horizon_pow))
  if getattr(command, 'importance_state', None) is None and cfg.failure_reset > 0.0:
    command.consecutive_local.copy_(torch.where(mask, streak, command.consecutive_local))
    command.restart_frame.masked_fill_(mask, -1)
  return frames


def reference_state_candidates(command, frames, pose_ranges, velocity_ranges):
  """Construct full-batch native reference proposals with the configured RSI noise."""
  from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
  lib, n, device = command.library, command.num_envs, command.device
  pos = lib.root_pos_w[frames].float() + command._env.scene.env_origins
  quat = lib.root_quat_w[frames].float()
  lin = lib.root_lin_vel_w[frames].float()
  ang = lib.root_ang_vel_w[frames].float()
  noise = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (n, 6), device)
  pos = pos + noise[:, :3]
  quat = quat_mul(quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), quat)
  noise = sample_uniform(velocity_ranges[:, 0], velocity_ranges[:, 1], (n, 6), device)
  lin, ang = lin + noise[:, :3], ang + noise[:, 3:]
  joint = lib.joint_pos[frames].float()
  low, high = command.cfg.joint_position_range
  joint = joint + sample_uniform(low, high, joint.shape, device)
  translator = command.spec.translator
  joint = translator.to_native(joint, command.robot.data.default_joint_pos[0])
  velocity = translator.to_native_delta(lib.joint_vel[frames].float())
  return pos, quat, lin, ang, joint, velocity


def reset_entity_masked(entity, mask):
  """Clear the known target and actuator-delay state for selected worlds."""
  data = entity._data
  if data.is_actuated:
    for name in ('joint_pos_target', 'joint_vel_target', 'joint_effort_target',
                 'tendon_len_target', 'tendon_vel_target', 'tendon_effort_target', 'site_effort_target'):
      value = getattr(data, name)
      value.masked_fill_(mask.reshape(mask.shape[0], *([1] * (value.ndim - 1))), 0)
  for actuator in entity.actuators:
    if actuator._delay_buffer is not None:
      actuator._delay_buffer.reset_masked(mask)


class MaskedEnvReset:
  """Fixed-shape reset with native builtin sensor views and validated managers."""
  def __init__(self, env):
    from morphmimic.env.actions import CanonicalJointPositionAction
    from morphmimic.env.graph_delay import GraphDelayBuffer
    from mjlab.actuator.actuator import Actuator
    from mjlab.sensor.builtin_sensor import BuiltinSensor
    self.env = env
    if env.tensor_step_stream is None or env.cfg.auto_reset:
      raise ValueError('masked reset requires a graph-stream environment with manual resets')
    if set(env.event_manager.available_modes) & {'reset', 'step', 'interval'}:
      raise ValueError('masked reset does not support runtime event terms')
    # Exact BuiltinSensor instances cache only a persistent sensordata view.
    # Native reset/forward below updates that storage in place; there is no
    # per-world Python sensor history to reset. Subclasses may add history.
    unsupported_sensors = [f'{name} ({type(sensor).__module__}.{type(sensor).__qualname__})'
                           for name, sensor in env.scene.sensors.items()
                           if type(sensor) is not BuiltinSensor]
    if unsupported_sensors:
      raise ValueError(f'masked reset does not support sensor types: {unsupported_sensors}')
    for label, manager in (('curriculum', env.curriculum_manager),
                           ('metrics', env.metrics_manager), ('recorder', env.recorder_manager)):
      if manager.active_terms:
        raise ValueError(f'masked reset does not support {label} terms: {manager.active_terms}')
    if env.reward_manager._class_term_cfgs or env.termination_manager._class_term_cfgs:
      raise ValueError('masked reset does not support class-based reward or termination terms')
    obs = env.observation_manager
    if (any(obs._group_obs_term_delay_buffer.values()) or any(obs._group_obs_term_history_buffer.values())
        or any(obs._group_obs_class_term_cfgs.values()) or any(obs._group_obs_class_instances.values())):
      raise ValueError('masked reset does not support stateful observation terms')
    if any(not isinstance(term, CanonicalJointPositionAction) for term in env.action_manager._terms.values()):
      raise ValueError('masked reset requires canonical joint position actions')
    self.delays = []
    seen = set()
    for entity in env.scene.entities.values():
      for actuator in entity.actuators:
        if type(actuator).reset is not Actuator.reset:
          raise ValueError('masked reset does not support stateful actuator subclasses')
        delay = actuator._delay_buffer
        if delay is not None and not isinstance(delay, GraphDelayBuffer):
          raise ValueError('masked reset requires graph-safe actuator delay buffers')
        if delay is not None and id(delay) not in seen:
          self.delays.append(delay);seen.add(id(delay))
    command = env.command_manager.get_term(env.motion_command_name)
    self.command = command
    keys = ('x','y','z','roll','pitch','yaw')
    self.pose_ranges = torch.tensor([command.cfg.pose_range.get(k,(0.,0.)) for k in keys],device=env.device)
    self.velocity_ranges = torch.tensor([command.cfg.velocity_range.get(k,(0.,0.)) for k in keys],device=env.device)
    translator = command.spec.translator
    translator._tensors(command.robot.data.joint_pos)
    translator._native_indices(command.robot.data.joint_pos)

  def __call__(self, mask):
    env = self.env
    self._validate_mask(mask)
    import warp as wp
    current = torch.cuda.current_stream(env.device)
    env.tensor_step_stream.wait_stream(current)
    with torch.cuda.stream(env.tensor_step_stream), wp.ScopedStream(wp.stream_from_torch(env.tensor_step_stream)):
      result = self._apply(mask)
    current.wait_stream(env.tensor_step_stream)
    return result

  def _validate_mask(self, mask):
    env = self.env
    if mask.dtype != torch.bool or tuple(mask.shape) != (env.num_envs,) or mask.device != torch.device(env.device):
      raise ValueError('masked reset requires one device boolean per environment')

  def _apply(self, mask):
    env, command = self.env, self.command
    present = mask.any()
    env.reset_native_masked(mask)
    for entity in env.scene.entities.values():
      reset_entity_masked(entity, mask)
    reset_action_manager_masked(env.action_manager, mask)
    reset_reward_manager_masked(env.reward_manager, mask)
    frames = reset_command_bookkeeping_masked(command, mask)
    proposals = reference_state_candidates(command, frames, self.pose_ranges, self.velocity_ranges)
    write_reference_masked(command, mask, *proposals)
    reset_entity_masked(command.robot, mask)
    env.episode_length_buf.masked_fill_(mask, 0)
    env._manual_reset_pending.masked_fill_(mask, False)
    old_ctrl = env.sim.data.ctrl.clone()
    for delay in self.delays: delay._enabled.copy_(present)
    env.scene.write_data_to_sim()
    env.sim.data.ctrl[:] = torch.where(present, env.sim.data.ctrl, old_ctrl)
    for delay in self.delays: delay._enabled.fill_(True)
    env.sim.forward()
    env.physics_diagnostics.sample_forward()
    previous = dict(command.metrics)
    command._update_metrics()
    for name, old in previous.items():
      old.copy_(torch.where(present, command.metrics[name], old))
      command.metrics[name] = old
    advanced = torch.minimum(command.global_frame + 1, command.library.clip_end(command.global_frame))
    command.global_frame.copy_(torch.where(mask, advanced, command.global_frame))
    old_pos, old_quat = command.body_pos_relative_w, command.body_quat_relative_w
    command.update_relative_body_poses()
    old_pos.copy_(torch.where(present, command.body_pos_relative_w, old_pos))
    old_quat.copy_(torch.where(present, command.body_quat_relative_w, old_quat))
    command.body_pos_relative_w, command.body_quat_relative_w = old_pos, old_quat
    env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    result = env.obs_buf
    return result


class MaskedResetGraph:
  """Capture one fixed-shape reset and replay it with a changing world mask."""
  def __init__(self, env):
    self.env = env
    self.resetter = MaskedEnvReset(env)
    self.graph = self.warp_graph = self.mask = self.output = self.signature = None

  def reset(self):
    if self.graph is not None:
      self.env.tensor_step_stream.synchronize()
    self.graph = self.warp_graph = self.mask = self.output = self.signature = None

  def __call__(self, mask):
    self.resetter._validate_mask(mask)
    signature = self.env._sim_array_signature()
    if self.graph is not None and signature != self.signature:
      raise RuntimeError('Simulator arrays changed; recreate the environment and captured graphs')
    if self.graph is None:
      self.mask = mask.clone()
    else:
      self.mask.copy_(mask)
    import warp as wp
    stream = self.env.tensor_step_stream
    current = torch.cuda.current_stream(self.env.device)
    stream.wait_stream(current)
    ws = wp.stream_from_torch(stream)
    with torch.cuda.stream(stream), wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
      if self.graph is None:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
          wp.capture_begin(stream=ws, external=True)
          try:
            output = self.resetter._apply(self.mask)
          finally:
            self.warp_graph = wp.capture_end(stream=ws)
        self.graph, self.output, self.signature = graph, output, signature
      self.graph.replay()
    current.wait_stream(stream)
    return self.output
