"""Noetix Bumi V3: native MJCF order, verified source PD values and rigid hand frames."""
from __future__ import annotations

import os
from pathlib import Path

from morphmimic.robots.spec import RobotSpec, Translator

JOINTS = ('waist_yaw_joint',) + tuple(f'{side}_{stem}_joint' for side in ('l', 'r') for stem in ('arm_pitch', 'arm_roll', 'arm_yaw', 'elbow_pitch')) + tuple(f'{side}_{stem}_joint' for side in ('l', 'r') for stem in ('leg_pitch', 'leg_roll', 'leg_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll'))
BODIES = ('base_link', 'l_leg_roll_link', 'l_knee_pitch_link', 'l_ankle_roll_link', 'r_leg_roll_link', 'r_knee_pitch_link', 'r_ankle_roll_link', 'waist_yaw_link', 'l_arm_roll_link', 'l_elbow_pitch_link', 'l_hand_frame', 'r_arm_roll_link', 'r_elbow_pitch_link', 'r_hand_frame')
# effort Nm, velocity rad/s, kp, kd, armature; supermimic/bumi.py pinned local controls.
MOTORS = {
  'leg_pitch': (60.,12.,60.,3.,0.), 'leg_roll': (27.,12.,60.,3.,0.),
  'leg_yaw': (15.,9.,60.,2.5,0.), 'knee_pitch': (60.,12.,45.,2.,0.),
  'ankle_pitch': (13.,12.,10.,.5,.01), 'ankle_roll': (13.,12.,10.,.5,.01),
  'waist_yaw': (27.,9.,53.,3.4,0.), 'arm_pitch': (5.,50.,12.,.4,0.),
  'arm_roll': (5.,50.,12.,.4,0.), 'arm_yaw': (5.,50.,12.,.4,0.), 'elbow_pitch': (5.,50.,12.,.4,0.),
}


def motor(name):
  return MOTORS[name.removeprefix('l_').removeprefix('r_').removesuffix('_joint')]


def get_spec():
  import mujoco
  root = Path(os.environ.get('BLINKMORPH_DATA_ROOT', '/data/kk/data'))
  spec = mujoco.MjSpec.from_file(str(root / 'supermimic_bumi/robot/bumi.xml'))
  for side in ('l', 'r'):
    # Both elbow STL bounds end at z=-0.18702218m: rigid distal endpoint, no new DoF/inertia.
    spec.body(f'{side}_elbow_pitch_link').add_body(name=f'{side}_hand_frame', pos=(0., 0., -.18702218))
    for i, geom in enumerate(spec.body(f'{side}_ankle_roll_link').geoms):
      if geom.contype != 0:
        geom.name = f'{side}_foot_collision_{i}'
  return spec


def get_robot_cfg():
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
  from mjlab.utils.spec_config import CollisionCfg
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0., 0., .65), joint_pos={'.*_leg_pitch_joint': -.1495, '.*_knee_pitch_joint': .3215, '.*_ankle_pitch_joint': -.172, 'l_arm_roll_joint': .3, 'r_arm_roll_joint': -.3}, joint_vel={'.*': 0.}),
    spec_fn=get_spec,
    collisions=(CollisionCfg(geom_names_expr=('(l|r)_foot_collision_.*',), contype=1, conaffinity=1, condim=3, priority=1, friction=(.6,), disable_other_geoms=False),),
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(BuiltinPositionActuatorCfg(target_names_expr=(name,), stiffness=motor(name)[2], damping=motor(name)[3], effort_limit=motor(name)[0], armature=motor(name)[4]) for name in JOINTS), soft_joint_pos_limit_factor=.9,
    ),
  )


def bumi_spec() -> RobotSpec:
  import mujoco
  import numpy as np
  from morphmimic.robots.g1 import g1_leg_length
  model = get_spec().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  hip, knee, ankle = (data.body(b).xpos for b in ('l_leg_pitch_link', 'l_knee_pitch_link', 'l_ankle_pitch_link'))
  length = float(np.linalg.norm(knee-hip) + np.linalg.norm(ankle-knee))
  mapping = {'waist_yaw_joint': 'waist_yaw_joint'}
  for side, native_side in [('left', 'l'), ('right', 'r')]:
    for canon, native in [('hip_pitch','leg_pitch'), ('hip_roll','leg_roll'), ('hip_yaw','leg_yaw'), ('knee','knee_pitch'), ('ankle_pitch','ankle_pitch'), ('ankle_roll','ankle_roll'), ('shoulder_pitch','arm_pitch'), ('shoulder_roll','arm_roll'), ('shoulder_yaw','arm_yaw'), ('elbow','elbow_pitch')]:
      mapping[f'{side}_{canon}_joint'] = f'{native_side}_{native}_joint'
  return RobotSpec(
    name='bumi', entity_cfg_fn=get_robot_cfg, action_scale={name: .25*motor(name)[0]/motor(name)[2] for name in JOINTS}, joint_names=JOINTS,
    root_body='base_link', anchor_body='waist_yaw_link', imu_body='base_link', tracked_bodies=BODIES,
    foot_geom_regex=r'^(l|r)_foot_collision_.*$', com_body='waist_yaw_link',
    translator=Translator.from_names(JOINTS, mapping, {'left_elbow_joint': -1., 'right_elbow_joint': -1.}, {}),
    csv_joint_columns=JOINTS, height_scale=length/g1_leg_length(), contact_budget=(100, 500),
    hip_pitch_body='l_leg_pitch_link', ankle_pitch_body='l_ankle_pitch_link', physics_timestep=.002, physics_decimation=10,
  )
