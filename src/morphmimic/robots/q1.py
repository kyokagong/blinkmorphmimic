"""Nominal Q1 geometry with hash-bound deployed PD gains and torque caps.

This is an ideal position-PD simulation, not the measured execution wrapper: motor
speed taper, slew, timing queue, prediction reuse and tilt guards are not modelled.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from morphmimic.constants import JOINT_NAMES, TRACKED_BODY_NAMES
from morphmimic.robots.spec import RobotSpec, Translator, height_scale

JOINTS = tuple(j for j in JOINT_NAMES if 'wrist' not in j and j != 'waist_pitch_joint')
# The XML puts waist roll before yaw; canonical order puts yaw before roll.
JOINTS = JOINTS[:12] + ('waist_roll_joint', 'waist_yaw_joint') + JOINTS[14:]
_PROFILE_HASHES = {
  'body_training_reference.json': 'f5c837889430345eef6d594e3615feaf5ced23121faf7b4db49f9c9cd52b9e27',
  'deployed_hardware.json': 'e33d406028847c3006779a26daf07c01ed70d3c70f957665526b116c07084057',
}


def asset_root() -> Path:
  return Path(os.environ.get('BLINKMORPH_DATA_ROOT', '/data/kk/data')) / 'blinkmimic_q1/20261107'


def profile() -> dict:
  root = asset_root() / 'hardware'
  loaded = {}
  for name, digest in _PROFILE_HASHES.items():
    raw = (root / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
      raise ValueError(f'Q1 profile identity mismatch: {root / name}')
    loaded[name] = json.loads(raw)
  ref, deployed = loaded.values()
  if ref['joint_order'] != list(JOINTS) or [j['name'] for j in ref['joints']] != list(JOINTS):
    raise ValueError('Q1 control profile native order mismatch')
  if ref['profile_sha256'] != _PROFILE_HASHES['deployed_hardware.json'] or ref['runtime']['policy_hz'] != 50:
    raise ValueError('Q1 control profile deployment mismatch')
  for key, target in [('kp', 'kp'), ('kd', 'kd')]:
    if [j[key] for j in ref['joints']] != deployed[target]:
      raise ValueError(f'Q1 deployed {key} mismatch')
  return ref


def get_spec():
  import mujoco
  path = asset_root() / 'robot/q1.xml'
  if hashlib.sha256(path.read_bytes()).hexdigest() != 'a10f1829fdf6b2a5a66f3a4bf8f8ce0cd5d067def8823d2286315c8eb758f4b6':
    raise ValueError(f'Q1 MJCF identity mismatch: {path}')
  return mujoco.MjSpec.from_file(str(path))


def get_robot_cfg():
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
  from mjlab.utils.spec_config import CollisionCfg
  joints = profile()['joints']
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0., 0., .42), joint_pos={j['name']: j['home_rad'] for j in joints}, joint_vel={'.*': 0.}),
    spec_fn=get_spec,
    collisions=(CollisionCfg(
      geom_names_expr=('.*_collision',), contype=1, conaffinity=1,
      condim={r'^(left|right)_foot[1-7]_collision$': 3, '.*_collision': 1},
      priority={r'^(left|right)_foot[1-7]_collision$': 1, '.*': 0},
      friction={r'^(left|right)_foot[1-7]_collision$': (.6,)},
    ),),
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(BuiltinPositionActuatorCfg(target_names_expr=(j['name'],), stiffness=j['kp'], damping=j['kd'], effort_limit=j['software_effort_limit_nm'], armature=.002) for j in joints),
      soft_joint_pos_limit_factor=.9,
    ),
  )


def q1_spec() -> RobotSpec:
  return RobotSpec(
    name='q1', entity_cfg_fn=get_robot_cfg, action_scale={'.*': .25}, joint_names=JOINTS,
    root_body='pelvis', anchor_body='torso_link', imu_body='pelvis', tracked_bodies=TRACKED_BODY_NAMES,
    foot_geom_regex=r'^(left|right)_foot[1-7]_collision$', com_body='torso_link',
    translator=Translator.from_names(JOINTS, {}, {}, {}), csv_joint_columns=JOINTS,
    height_scale=height_scale(get_robot_cfg, 'left_knee_link'), contact_budget=(100, 500),
    physics_timestep=.002, physics_decimation=10,
  )
