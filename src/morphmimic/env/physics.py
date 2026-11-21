"""Device-only native overflow sampling; CPU reporting at logging/UI boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def engine_metadata() -> dict:
  from mujoco_warp._src import types
  patch = Path('/opt/blinkmimic-physics-patch.json')
  return {
    'epa_horizon': int(types.MJ_MAX_EPAHORIZON),
    'types_sha256': hashlib.sha256(Path(types.__file__).read_bytes()).hexdigest(),
    'patch': json.loads(patch.read_text()) if patch.exists() else None,
    'overflow_flags': {flag.name: int(flag) for flag in types.OverflowType},
    'coverage': 'native overflow bits; direct forward NEFC, NJMAX_NNZ, BROADPHASE and NARROWPHASE checks when bound; EPA face exhaustion has no diagnostic bit',
  }


class PhysicsDiagnostics:
  """Sticky process-local evidence survives native per-world episode resets."""
  def __init__(self, source: torch.Tensor, constraint_rows: torch.Tensor | None = None,
               constraint_capacity: int | None = None, sparse_rows: tuple | None = None,
               contact_counts: tuple | None = None):
    if source.ndim != 1 or source.dtype not in (torch.int32, torch.int64):
      raise ValueError('native overflow flags must be a one-dimensional integer tensor')
    self.source = source
    if constraint_rows is not None or constraint_capacity is not None:
      if (not isinstance(constraint_rows, torch.Tensor) or constraint_rows.shape != source.shape
          or constraint_rows.dtype not in (torch.int32, torch.int64) or constraint_rows.device != source.device
          or type(constraint_capacity) is not int or constraint_capacity <= 0):
        raise ValueError('constraint rows must be a matching integer tensor with a positive row capacity')
    self.constraint_rows = constraint_rows
    self.constraint_capacity = constraint_capacity
    def valid_tensor(value, shape):
      return (isinstance(value, torch.Tensor) and value.shape == shape
              and value.dtype in (torch.int32, torch.int64) and value.device == source.device)
    if sparse_rows is not None:
      if not isinstance(sparse_rows, tuple) or len(sparse_rows) != 3 or constraint_rows is None:
        raise ValueError('sparse rows require constraint row counts and two index matrices')
      adr, nnz, capacity = sparse_rows
      if (not isinstance(adr, torch.Tensor) or adr.ndim != 2 or adr.shape[0] != source.numel()
          or adr.shape[1] < constraint_capacity or not valid_tensor(adr, adr.shape)
          or not valid_tensor(nnz, adr.shape) or type(capacity) is not int or capacity < 0):
        raise ValueError('invalid sparse constraint buffers or capacity')
    if contact_counts is not None:
      if not isinstance(contact_counts, tuple) or len(contact_counts) != 3:
        raise ValueError('contact counts require two global counters and a capacity')
      collision, contacts, capacity = contact_counts
      if (not valid_tensor(collision, (1,)) or not valid_tensor(contacts, (1,))
          or type(capacity) is not int or capacity <= 0):
        raise ValueError('invalid global contact counters or capacity')
    self.sparse_rows, self.contact_counts = sparse_rows, contact_counts
    self.overflow_bits = torch.zeros_like(source)
    self.checked_physics_steps = torch.zeros((), dtype=torch.long, device=source.device)
    self.flagged_observations = torch.zeros_like(self.checked_physics_steps)
    self.checked_forward_calls = torch.zeros_like(self.checked_physics_steps)
    self.flagged_forward_observations = torch.zeros_like(self.checked_physics_steps)
    self._bit_values = torch.tensor([1 << i for i in range(32)], device=source.device)
    self.engine = {**engine_metadata(), 'direct_forward_nefc': constraint_rows is not None,
                   'constraint_capacity': constraint_capacity, 'direct_forward_sparse': sparse_rows is not None,
                   'sparse_nnz_capacity': sparse_rows[2] if sparse_rows is not None else None,
                   'direct_forward_contacts': contact_counts is not None,
                   'contact_capacity': contact_counts[2] if contact_counts is not None else None}
    self._nefc_bit = self.engine['overflow_flags']['NEFC'] if constraint_rows is not None else 0

  @classmethod
  def from_native(cls, data, is_sparse: bool):
    """Bind the same arrays and limits passed to pinned MuJoCo Warp's next-time kernel."""
    import warp as wp
    if data.efc.type.shape[1] < data.njmax:
      raise ValueError('allocated constraint buffer is smaller than native njmax')
    sparse = (wp.to_torch(data.efc.J_rowadr), wp.to_torch(data.efc.J_rownnz), data.njmax_nnz) if is_sparse else None
    contacts = (wp.to_torch(data.ncollision), wp.to_torch(data.nacon), data.naconmax)
    return cls(wp.to_torch(data.overflow), wp.to_torch(data.nefc), data.njmax, sparse, contacts)

  def sample(self) -> None:
    """Call immediately after every physics substep, before resets can clear flags."""
    self.overflow_bits.bitwise_or_(self.source)
    self.checked_physics_steps.add_(1)
    self.flagged_observations.add_((self.source != 0).sum())

  def bits_tensor(self) -> torch.Tensor:
    """All worlds' sticky bitwise union as one device scalar, without host reads."""
    present = (self.overflow_bits.long().unsqueeze(-1) & self._bit_values) != 0
    return (present.any(dim=0).long() * self._bit_values).sum()

  def sample_forward(self) -> None:
    """Record collision/constraint forward flags without inventing a physics step."""
    flags = self.source
    if self.constraint_rows is not None:
      flags = flags.bitwise_or((self.constraint_rows > self.constraint_capacity).to(flags.dtype) * self._nefc_bit)
    if self.sparse_rows is not None:
      adr, nnz, capacity = self.sparse_rows
      index = (self.constraint_rows - 1).clamp(0, self.constraint_capacity - 1).long().unsqueeze(1)
      used = adr.gather(1, index).squeeze(1) + nnz.gather(1, index).squeeze(1)
      overflow = (self.constraint_rows > 0) & (self.constraint_rows <= self.constraint_capacity) & (used > capacity)
      flags = flags.bitwise_or(overflow.to(flags.dtype) * self.engine['overflow_flags']['NJMAX_NNZ'])
    if self.contact_counts is not None:
      collision, contacts, capacity = self.contact_counts
      flags = flags.bitwise_or((collision > capacity).to(flags.dtype) * self.engine['overflow_flags']['BROADPHASE'])
      flags = flags.bitwise_or((contacts > capacity).to(flags.dtype) * self.engine['overflow_flags']['NARROWPHASE'])
    self.overflow_bits.bitwise_or_(flags)
    self.checked_forward_calls.add_(1)
    self.flagged_forward_observations.add_((flags != 0).sum())

  def report(self) -> dict:
    values = torch.cat((self.overflow_bits.long(), self.checked_physics_steps[None],
                        self.flagged_observations[None], self.checked_forward_calls[None],
                        self.flagged_forward_observations[None])).detach().cpu().tolist()
    bits = 0
    for value in values[:-4]:
      bits |= value & 0xffffffff
    flags, known = [], 0
    for name, flag in self.engine['overflow_flags'].items():
      known |= flag
      if bits & flag:
        flags.append(name)
    if bits & ~known:
      flags.append(f'UNKNOWN_0x{bits & ~known:x}')
    checks, flagged, forwards, flagged_forward = values[-4:]
    return dict(scope='current_process; preserved across episode resets',
                overflow_bits=bits, overflow_flags=flags,
                overflow_worlds=sum(value != 0 for value in values[:-4]),
                checked_physics_steps=checks, checked_step_worlds=checks * self.source.numel(),
                flagged_step_world_observations=flagged,
                checked_forward_calls=forwards,
                flagged_forward_world_observations=flagged_forward,
                count_semantics='flagged step/forward-world observations, not independent overflow events',
                zero_overflow_observed=checks > 0 and bits == 0, engine=self.engine)
