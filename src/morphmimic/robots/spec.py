"""Robot metadata and sparse canonical29/native joint translation.

Absent canonical joints use perm=-1, zero observations and limits, and ignored
actions. Real native joints without canonical slots remain at their default pose.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import math

import torch
from torch import Tensor

from morphmimic.constants import JOINT_NAMES

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class Translator:
  """Canonical 29-slot <-> native joint map.

  q_canon = sign * (q_native[perm] - offset).
  """

  perm: tuple[int, ...]  # len 29: native index, -1 for an absent joint
  sign: tuple[float, ...]  # len 29, each +1.0 or -1.0
  offset: tuple[float, ...]  # len 29, radians
  num_native: int
  _tensor_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)
  _native_index_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)

  def __post_init__(self) -> None:
    if len(self.perm) != 29 or len(self.sign) != 29 or len(self.offset) != 29:
      raise ValueError("translator requires 29 canonical slots")
    tracked = [p for p in self.perm if p >= 0]
    if self.num_native < 1 or any(p < -1 or p >= self.num_native for p in self.perm):
      raise ValueError("invalid native joint index")
    if len(set(tracked)) != len(tracked):
      raise ValueError("canonical slots must not share a native joint")
    if any(s not in (-1., 1.) for s in self.sign) or not all(map(math.isfinite, self.offset)):
      raise ValueError("translator requires signs +/-1 and finite offsets")

  @property
  def joint_mask(self) -> tuple[bool, ...]:
    return tuple(p >= 0 for p in self.perm)

  def _tensors(self, ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    key = (ref.device, ref.dtype)
    tensors = self._tensor_cache.get(key)
    if tensors is None:
      tensors = (torch.as_tensor(self.perm, dtype=torch.long, device=ref.device),
                 torch.as_tensor(self.sign, dtype=ref.dtype, device=ref.device),
                 torch.as_tensor(self.offset, dtype=ref.dtype, device=ref.device))
      if len(self._tensor_cache) < 16:
        self._tensor_cache[key] = tensors
    return tensors

  def _native_indices(self, ref: Tensor) -> tuple[Tensor, Tensor]:
    indices = self._native_index_cache.get(ref.device)
    if indices is None:
      active = [i for i, native in enumerate(self.perm) if native >= 0]
      indices = (torch.tensor(active, dtype=torch.long, device=ref.device),
                 torch.tensor([self.perm[i] for i in active], dtype=torch.long, device=ref.device))
      if len(self._native_index_cache) < 16:
        self._native_index_cache[ref.device] = indices
    return indices

  def to_canonical(self, q: Tensor) -> Tensor:
    """Native joint positions [..., J] -> canonical [..., 29]."""
    perm, sign, offset = self._tensors(q)
    return torch.where(perm >= 0, sign * (q[..., perm.clamp_min(0)] - offset), 0.)

  def to_canonical_delta(self, x: Tensor) -> Tensor:
    """Velocities, biases, actions [..., J] -> [..., 29]: sign * x[perm], no offset."""
    perm, sign, _ = self._tensors(x)
    return torch.where(perm >= 0, sign * x[..., perm.clamp_min(0)], 0.)

  def to_native(self, q: Tensor, default: Tensor) -> Tensor:
    """Canonical [..., 29] + default [J] -> native [..., J]; untracked = default."""
    _, sign, offset = self._tensors(q)
    out = default.to(device=q.device, dtype=q.dtype)
    out = out.expand(*q.shape[:-1], self.num_native).clone()
    active, native = self._native_indices(q)
    out.index_copy_(-1, native, (sign * q + offset).index_select(-1, active))
    return out

  def to_native_delta(self, x: Tensor) -> Tensor:
    """Canonical deltas [..., 29] -> native [..., J]; untracked = 0."""
    _, sign, _ = self._tensors(x)
    out = x.new_zeros(*x.shape[:-1], self.num_native)
    active, native = self._native_indices(x)
    out.index_copy_(-1, native, (sign * x).index_select(-1, active))
    return out

  def limits_to_canonical(self, limits: Tensor) -> Tensor:
    """Native [J, 2] (lo, hi) -> canonical [29, 2], bounds swapped where sign < 0."""
    perm, sign, offset = self._tensors(limits)
    lo, hi = limits[perm.clamp_min(0)].unbind(-1)
    a = sign * (lo - offset)
    b = sign * (hi - offset)
    bounds = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], -1)
    return torch.where((perm >= 0).unsqueeze(-1), bounds, 0.)

  @property
  def untracked(self) -> tuple[int, ...]:
    """Native indices with no canonical slot, ascending."""
    tracked = set(self.perm)
    return tuple(i for i in range(self.num_native) if i not in tracked)

  @classmethod
  def identity(cls, n: int) -> Translator:
    return cls(
      perm=tuple(range(n)), sign=(1.0,) * n, offset=(0.0,) * n, num_native=n
    )

  @classmethod
  def from_names(
    cls,
    native: Sequence[str],
    canonical_to_native: Mapping[str, str],
    sign: Mapping[str, float],
    offset: Mapping[str, float],
  ) -> Translator:
    """Build from joint names. All three mappings are keyed by canonical joint name and
    are sparse: a missing key means the same native name, sign +1.0, offset 0.0.
    Names absent from native produce a missing canonical slot (perm=-1)."""
    for what, m in (("map", canonical_to_native), ("sign", sign), ("offset", offset)):
      unknown = sorted(set(m) - set(JOINT_NAMES))
      if unknown:
        raise ValueError(f"{what} keys are not canonical joints: {unknown}")
    targets = [canonical_to_native.get(c, c) for c in JOINT_NAMES]
    if len(set(native)) != len(native):
      raise ValueError("native joint names must be unique")
    matched = [n for n in targets if n in native]
    if len(set(matched)) != len(matched):
      raise ValueError(f"canonical joints map to the same native joint: {targets}")
    return cls(
      perm=tuple(native.index(n) if n in native else -1 for n in targets),
      sign=tuple(float(sign.get(c, 1.0)) for c in JOINT_NAMES),
      offset=tuple(float(offset.get(c, 0.0)) for c in JOINT_NAMES),
      num_native=len(native),
    )


@dataclass(frozen=True)
class RobotSpec:
  """Everything robot-specific the motion library, env and trainer read."""

  name: str
  entity_cfg_fn: Callable[[], EntityCfg]
  action_scale: dict[str, float]  # regex over actuator target names -> scale
  joint_names: tuple[str, ...]  # native hinge order
  root_body: str
  anchor_body: str
  imu_body: str
  """The body the robot's IMU sits on (its gyro site); the policy reads the gyro and the
  gravity direction in this body's frame. Must be a tracked body."""
  tracked_bodies: tuple[str, ...]  # 14, canonical functional order
  foot_geom_regex: str
  com_body: str
  translator: Translator
  csv_joint_columns: tuple[str, ...]  # native order, as the robot's CSVs name them
  height_scale: float
  contact_budget: tuple[int, int]  # (nconmax, njmax)
  physics_timestep: float = .005
  physics_decimation: int = 4
  hip_pitch_body: str = "left_hip_pitch_link"
  ankle_pitch_body: str = "left_ankle_pitch_link"
  # Bodies left out of the lowest-body-height statistic.
  zmin_exclude_regex: str | None = None

  @property
  def imu_slot(self) -> int:
    """Tracked slot of `imu_body` (0 for the G1 and X2 pelvis, 7 for the T2 trunk)."""
    if self.imu_body not in self.tracked_bodies:
      raise ValueError(f"{self.name}: IMU body {self.imu_body!r} is not tracked")
    return self.tracked_bodies.index(self.imu_body)

  def native_index(self, joint: str) -> int:
    if joint not in self.joint_names:
      raise ValueError(f"{self.name} has no joint {joint!r}")
    return self.joint_names.index(joint)


def leg_length(entity_cfg_fn: Callable[[], EntityCfg], knee_body: str) -> float:
  """Left thigh + shank at qpos0, between body origins.

  |hip pitch -> knee| + |knee -> ankle pitch|, world frame after `mj_kinematics`.
  """
  import mujoco
  import numpy as np

  model = entity_cfg_fn().spec_fn().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  hip, knee, ankle = (
    data.body(b).xpos
    for b in ("left_hip_pitch_link", knee_body, "left_ankle_pitch_link")
  )
  return float(np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee))


def height_scale(entity_cfg_fn: Callable[[], EntityCfg], knee_body: str) -> float:
  """Leg length relative to the G1's (0.641 m), by the same measure."""
  from morphmimic.robots.g1 import g1_leg_length

  return leg_length(entity_cfg_fn, knee_body) / g1_leg_length()
