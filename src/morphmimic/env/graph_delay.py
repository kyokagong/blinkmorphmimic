"""Per-environment graph-safe replacement for mjlab's delay ring pointer."""
import torch
from mjlab.utils.buffers import CircularBuffer, DelayBuffer


class GraphCircularBuffer(CircularBuffer):
  def __init__(self, max_len, batch_size, device):
    super().__init__(max_len, batch_size, device)
    self._pointer = torch.full((), -1, dtype=torch.long, device=device)
    self._enabled = torch.ones((), dtype=torch.bool, device=device)

  def append(self, data):
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")
    data = data.to(self._device)
    if self._buffer is None:
      self._pointer.fill_(-1)
      self._buffer = torch.empty((self._max_len, *data.shape), dtype=data.dtype, device=self._device)
    next_pointer = (self._pointer + 1).remainder(self._max_len)
    self._pointer.copy_(torch.where(self._enabled, next_pointer, self._pointer))
    slot = self._pointer.clamp_min(0).reshape(1)
    old = self._buffer.index_select(0, slot)
    self._buffer.index_copy_(0, slot, torch.where(self._enabled, data.unsqueeze(0), old))
    first = ((self._num_pushes == 0) & self._enabled).view(1, self._batch_size, *([1] * (data.ndim - 1)))
    torch.where(first, data.unsqueeze(0), self._buffer, out=self._buffer)
    self._num_pushes.add_(self._enabled.long())



class GraphDelayBuffer(DelayBuffer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    old = self._buffer
    self._buffer = GraphCircularBuffer(old._max_len, old._batch_size, old._device)
    self._enabled = self._buffer._enabled

  def reset_masked(self, mask):
    if mask.dtype != torch.bool or tuple(mask.shape) != (self.batch_size,):
      raise ValueError("delay reset mask must be boolean with one entry per environment")
    mask = mask.to(self.device)
    ring = self._buffer
    ring._num_pushes.masked_fill_(mask, 0)
    if ring._buffer is not None:
      shape = (1, self.batch_size, *([1] * (ring._buffer.ndim - 2)))
      ring._buffer.masked_fill_(mask.reshape(shape), 0)
    self._current_lags.masked_fill_(mask, 0)
    self._step_count.masked_fill_(mask, 0)
    if self.update_period > 0 and self.per_env_phase:
      phases = torch.randint(0, self.update_period, (self.batch_size,), dtype=torch.long,
                             device=self.device, generator=self.generator)
      self._phase_offsets.copy_(torch.where(mask, phases, self._phase_offsets))

  def _update_lags(self):
    if self.update_period > 0:
      should_update = (self._step_count + self._phase_offsets) % self.update_period == 0
    else:
      should_update = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
    new_lags = self._sample_lags(should_update)
    self._current_lags.copy_(torch.where(should_update & self._enabled, new_lags, self._current_lags))
    self._step_count.add_(self._enabled.long())


def prepare_graph_delays(scene):
  """Replace only this scene's actuator delay buffers, preserving existing state."""
  for entity in scene.entities.values():
    if not entity.actuators:
      continue
    replacements = {}
    for actuator in entity.actuators:
      old = actuator._delay_buffer
      if old is None or isinstance(old, GraphDelayBuffer):
        continue
      if id(old) not in replacements:
        new = object.__new__(GraphDelayBuffer)
        new.__dict__.update(old.__dict__)
        ring = old._buffer
        new_ring = GraphCircularBuffer(ring._max_len, ring._batch_size, ring._device)
        pointer = ring._pointer
        new_ring.__dict__.update(ring.__dict__)
        new_ring._pointer = torch.as_tensor(pointer, dtype=torch.long, device=ring._device)
        new._buffer = new_ring
        new._enabled = new_ring._enabled
        replacements[id(old)] = new
      actuator._delay_buffer = replacements[id(old)]
    for group in entity._builtin_group._delayed_groups:
      group.delay_buffer = replacements.get(id(group.delay_buffer), group.delay_buffer)
    for group in entity._fused_actuator_group._groups:
      group.delay_buffer = replacements.get(id(group.delay_buffer), group.delay_buffer)
