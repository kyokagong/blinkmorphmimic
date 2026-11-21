"""Observe actual native PD-delay state without changing its cadence or RNG."""
import torch

from morphmimic.model.controller import (
  QUEUE_LEN, MAX_SUBSTEPS, ControllerCfg, ControllerState, NativeDelayController,
)


def read_delay_state(delay, canonicalize):
  """Read circular storage in newest-first order; no lag draw or queue mutation."""
  ring = delay._buffer
  if ring._buffer is None:
    raise RuntimeError("controller observation requires the native initial reset write")
  offsets = torch.arange(ring._max_len, device=ring._buffer.device)
  indices = (ring._pointer - offsets).remainder(ring._max_len)
  queue = canonicalize(ring._buffer.index_select(0, indices).transpose(0, 1))
  queue = torch.where((ring._num_pushes > 0)[:, None, None], queue, 0.)
  if ring._max_len < QUEUE_LEN:
    queue = torch.cat((queue, queue.new_zeros(queue.shape[0], QUEUE_LEN-ring._max_len, 29)), 1)
  return ControllerState(queue, torch.stack((delay._current_lags, ring._num_pushes,
                                             delay._step_count, delay._phase_offsets), -1))


class NativeControllerRecorder:
  """One fused builtin position-delay group, the actual configuration of Q1/Bumi/G1.

  Unsupported controller configurations fail during construction. No Q1 hardware
  calibration is applied to other robots (or to this simulator mode).
  """
  def __init__(self, env, spec):
    from morphmimic.env.actions import CanonicalJointPositionAction
    if env.cfg.auto_reset:
      raise ValueError("controller recording requires explicit collector resets")
    robot = env.scene['robot']
    group_manager = robot._builtin_group
    if len(group_manager._delayed_groups) != 1 or group_manager._index_groups or robot._fused_actuator_group._groups:
      raise ValueError("native_delay_v1 requires one fused builtin position-delay group")
    group = group_manager._delayed_groups[0]
    if group.target_attr != 'joint_pos_target' or group.hold_prob or not group.delay_buffer.per_env:
      raise ValueError("native_delay_v1 requires independent uniform lag and position targets")
    tr = spec.translator
    if sorted(group.target_ids.tolist()) != list(range(tr.num_native)):
      raise ValueError("controller group must cover every native joint exactly once")
    term = env.action_manager.get_term('joint_pos')
    if not isinstance(term, CanonicalJointPositionAction):
      raise ValueError("controller recording requires the canonical joint-position action")
    self.order = group.target_ids.argsort()
    self.ctrl_ids = robot.data.indexing.ctrl_ids[group.ctrl_ids]
    self.env, self.tr, self.group = env, tr, group
    self.delay = group.delay_buffer
    q0_native = torch.as_tensor(term._offset, device=env.device).expand(env.num_envs, tr.num_native)
    scale_native = torch.as_tensor(term._scale, device=env.device).expand_as(q0_native)
    if not torch.equal(q0_native, q0_native[:1].expand_as(q0_native)) or not torch.equal(scale_native, scale_native[:1].expand_as(scale_native)):
      raise ValueError("native_delay_v1 requires constant per-robot action scale and default offset")
    if not bool((scale_native > 0).all()):
      raise ValueError("native_delay_v1 requires positive native action scales")
    lower, upper = None, None
    if term.cfg.clip is not None:
      if not torch.equal(term._clip, term._clip[:1].expand_as(term._clip)):
        raise ValueError("native_delay_v1 requires constant per-robot target clipping")
      limits = tr.limits_to_canonical(term._clip[0]); lower, upper = limits.unbind(-1)
    self.controller = NativeDelayController(
      ControllerCfg(group.max_lag,env.cfg.decimation,group.update_period,env.physics_dt,
                    group.min_lag,group.per_env_phase),
      tr.to_canonical(q0_native[0]),tr.to_canonical_delta(scale_native[0]).abs(),
      term.joint_mask,lower,upper).to(env.device)
    model = env.sim.mj_model
    ctrl_ids = self.ctrl_ids.cpu().numpy()
    self.controller.plant_identity = dict(robot=spec.name,
      native_joint_names=list(spec.joint_names),canonical_to_native=list(tr.perm),
      canonical_sign=list(tr.sign),canonical_offset=list(tr.offset),
      native_target_ids=group.target_ids.tolist(),native_ctrl_ids=self.ctrl_ids.tolist(),
      actuators={name:getattr(model,name)[ctrl_ids].tolist() for name in (
        'actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange',
        'actuator_ctrllimited','actuator_forcelimited','actuator_gaintype','actuator_biastype',
        'actuator_dyntype','actuator_dynprm','actuator_gear')})
    self.before = self.controller.empty_state(env.num_envs)
    self.after = self.controller.empty_state(env.num_envs)
    self.targets = torch.zeros(env.num_envs, MAX_SUBSTEPS, 29, device=env.device)
    self.lags = torch.zeros(env.num_envs, MAX_SUBSTEPS, dtype=torch.long, device=env.device)

  def canonicalize(self, values):
    return self.tr.to_canonical(values.index_select(-1,self.order))

  def _read_into(self, destination):
    for dst, src in zip(destination, read_delay_state(self.delay,self.canonicalize)):
      dst.copy_(src)

  def begin(self):
    self._read_into(self.before)

  def substep(self, index):
    actual_ctrl = self.env.sim.data.ctrl.index_select(-1,self.ctrl_ids)
    self.targets[:,index].copy_(self.canonicalize(actual_ctrl))
    self.lags[:,index].copy_(self.delay.current_lags)

  def finish(self):
    self._read_into(self.after)


def controller_from_config(cfg):
  """Resolve the actual plant on CPU for audits, without a simulation or checkpoint tensors.

  Scene compilation, builtin actuator initialization, fused grouping and canonical
  action construction are the same native routines used by the live environment.
  The small inspection shell supplies only native configuration/default-pose fields;
  it never steps physics, observes a queue, or fabricates rollout history.
  """
  from types import SimpleNamespace
  from mjlab.scene import Scene
  from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
  from mjlab.entity.entity import BuiltinActuatorGroup, FusedActuatorGroup
  from mjlab.utils.string import resolve_expr
  from morphmimic.env.env_cfg import morphmimic_env_cfg
  from morphmimic.robots import get_robot

  if cfg.get('controller_mode','legacy') != 'native_delay_v1':
    raise ValueError('CPU controller resolver requires native_delay_v1')
  required=('robot','actuator_delay','actuator_delay_max_ms','actuator_delay_fixed')
  if (any(key not in cfg for key in required) or cfg['actuator_delay'] is not True
      or type(cfg['actuator_delay_fixed']) is not bool):
    raise ValueError('controller configuration lacks the actual native delay settings')
  spec=get_robot(cfg['robot'])
  env_cfg=morphmimic_env_cfg(cfg['robot'],num_envs=1,
    actuator_delay=True,actuator_delay_max_s=cfg['actuator_delay_max_ms']/1000,
    actuator_delay_fixed=cfg['actuator_delay_fixed'],goal_anchor=cfg.get('goal_anchor','reference'))
  scene=Scene(env_cfg.scene,'cpu');model=scene.compile();robot=scene['robot']
  if not all(type(a) is BuiltinPositionActuator for a in robot.actuators):
    raise ValueError('controller audit requires the actual builtin position plant')
  robot.indexing=robot._compute_indexing(model,'cpu')
  for actuator in robot.actuators:
    actuator.initialize(model,None,SimpleNamespace(nworld=1),'cpu')
  builtin,custom=BuiltinActuatorGroup.process(robot.actuators)
  builtin.initialize(1,'cpu')
  fused,custom=FusedActuatorGroup.process(custom)
  fused.initialize(1,'cpu')
  if custom:raise ValueError('controller audit does not support custom actuators')
  robot._builtin_group,robot._fused_actuator_group=builtin,fused
  if robot.cfg.init_state.joint_pos is None:
    raise ValueError('controller audit requires explicitly configured native default joints')
  default=torch.tensor(resolve_expr(robot.cfg.init_state.joint_pos,robot.joint_names,0.0))[None]
  robot._data=SimpleNamespace(default_joint_pos=default,indexing=robot.indexing)
  env=SimpleNamespace(cfg=env_cfg,scene=scene,num_envs=1,device='cpu',
    physics_dt=env_cfg.sim.mujoco.timestep,sim=SimpleNamespace(mj_model=model))
  action=env_cfg.actions['joint_pos'].build(env)
  env.action_manager=SimpleNamespace(get_term=lambda name:action)
  return NativeControllerRecorder(env,spec).controller
