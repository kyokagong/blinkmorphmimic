"""Static morphology: original 125 descriptors followed by the 29-joint mask.

The first116 entries are log(action scale, kp, kd, effort),29 each; geometry9
is thigh/shank/upper-arm/forearm length, hip width, torso height, total/leg/arm
mass. Descriptors are min/max scaled across q1/bumi/g1. Missing actuators use
positive placeholders before log and zero scaled features afterward. The appended
binary mask is never scaled. Without actor geometry the vector is116+29=145.
"""

from __future__ import annotations

import re
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import torch

from morphmimic.robots import ROBOTS, VARIANTS, get_robot

if TYPE_CHECKING:
  from collections.abc import Iterable

  from morphmimic.robots.spec import RobotSpec

NUM_CANON = 29
ACTUATOR_DIM = 4 * NUM_CANON
GEOMETRY_DIM = 9
DESCRIPTOR_DIM = ACTUATOR_DIM + GEOMETRY_DIM
MORPH_DIM = DESCRIPTOR_DIM + NUM_CANON

_ACTUATOR_KEYS = ("action_scale", "kp", "kd", "effort")
GEOMETRY_NAMES = (
  "thigh",
  "shank",
  "upper_arm",
  "forearm",
  "hip_width",
  "torso_height",
  "total_mass",
  "leg_mass",
  "arm_mass",
)


def _one_match(joint: str, exprs: Iterable[tuple[str, ...]], what: str) -> int:
  """Index of the single entry whose regexes fullmatch `joint`; ValueError otherwise."""
  hits = [
    i for i, group in enumerate(exprs) if any(re.fullmatch(e, joint) for e in group)
  ]
  if len(hits) != 1:
    raise ValueError(f"joint {joint!r} matches {len(hits)} {what} (expected 1)")
  return hits[0]


def _actuator_terms(spec: RobotSpec) -> dict[str, np.ndarray]:
  actuators = spec.entity_cfg_fn().articulation.actuators
  act_exprs = [tuple(a.target_names_expr) for a in actuators]
  scale_keys = list(spec.action_scale)
  out = {k: np.ones(NUM_CANON) for k in _ACTUATOR_KEYS}
  for slot, native in enumerate(spec.translator.perm):
    if native < 0:
      continue
    joint = spec.joint_names[native]
    a = actuators[_one_match(joint, act_exprs, f"{spec.name} actuator configs")]
    key = scale_keys[
      _one_match(joint, [(k,) for k in scale_keys], f"{spec.name} action-scale keys")
    ]
    vals = {
      "action_scale": spec.action_scale[key],
      "kp": a.stiffness,
      "kd": a.damping,
      "effort": a.effort_limit,
    }
    for k, v in vals.items():
      if v is None or not v > 0:
        raise ValueError(f"{spec.name} joint {joint!r}: {k} = {v}, expected > 0")
      out[k][slot] = float(v)
  return out


def _geometry(spec: RobotSpec) -> np.ndarray:
  import mujoco

  model = spec.entity_cfg_fn().spec_fn().compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  tb = spec.tracked_bodies

  def pos(body: str) -> np.ndarray:
    return data.body(body).xpos.copy()

  def dist(a: str, b: str) -> float:
    return float(np.linalg.norm(pos(b) - pos(a)))

  shoulder = model.body(tb[8]).parentid[0]
  if model.body(shoulder).parentid[0] != model.body(tb[7]).id:
    raise ValueError(
      f"{spec.name}: parent of {tb[8]!r} ({model.body(shoulder).name!r}) does not "
      f"hang off {tb[7]!r}"
    )
  return np.array(
    [
      dist(spec.hip_pitch_body, tb[2]),
      dist(tb[2], spec.ankle_pitch_body),
      dist(tb[8], tb[9]),
      dist(tb[9], tb[10]),
      dist(tb[1], tb[4]),
      abs(float(pos(tb[0])[2] - 0.5 * (pos(tb[8])[2] + pos(tb[11])[2]))),
      float(mujoco.mj_getTotalmass(model)),
      float(model.body_subtreemass[model.body(spec.hip_pitch_body).id]),
      float(model.body_subtreemass[shoulder]),
    ]
  )


def raw_descriptors(spec: RobotSpec) -> dict[str, np.ndarray]:
  """Unscaled descriptors: `action_scale`, `kp`, `kd`, `effort` [29] in canonical
  order, `geometry` [9] (see the module docstring). Uncached; see `_raw`."""
  out = _actuator_terms(spec)
  out["geometry"] = _geometry(spec)
  return out


@cache
def _raw(name: str) -> np.ndarray:
  """Unscaled [125] vector for a registry robot, log on the actuator part."""
  raw = raw_descriptors(get_robot(name))
  return np.concatenate([*(np.log(raw[k]) for k in _ACTUATOR_KEYS), raw["geometry"]])


@cache
def _bounds() -> tuple[np.ndarray, np.ndarray]:
  """Per-feature minimum and maximum of the raw vector over the training robots."""
  raw = np.stack([_raw(n) for n in ROBOTS])
  return raw.min(0), raw.max(0)


def _scale(raw: np.ndarray) -> np.ndarray:
  lo, hi = _bounds()
  span = hi - lo
  const = span <= 1e-12 * np.maximum(1.0, np.abs(hi))
  return np.where(const, 0.0, 2 * (raw - lo) / np.where(const, 1.0, span) - 1)


@cache
def _table() -> dict[str, np.ndarray]:
  """Per-feature min-max scaled [125] vector per training robot."""
  return {n: _scale(_raw(n)) for n in ROBOTS}


@cache
def _scaled(name: str) -> np.ndarray:
  """Scaled [125] vector of a training robot or a variant."""
  return _table()[name] if name in ROBOTS else _scale(_raw(name))


def morphology_vector(
  name: str, actor: bool = False, actor_geometry: bool = True
) -> torch.Tensor:
  """Scaled morphology vector of a training robot or a variant, float32 [154].

  With `actor=True` and `actor_geometry=False`: the actuator part and mask [145].
  """
  if name not in ROBOTS + VARIANTS:
    raise ValueError(f"unknown robot {name!r}; expected one of {ROBOTS + VARIANTS}")
  vec = _scaled(name)
  if actor and not actor_geometry:
    vec = vec[:ACTUATOR_DIM]
  mask = np.asarray(get_robot(name).translator.joint_mask, dtype=float)
  vec = vec.copy()
  vec[:ACTUATOR_DIM] *= np.tile(mask, 4)
  return torch.tensor(np.concatenate([vec, mask]), dtype=torch.float32)
