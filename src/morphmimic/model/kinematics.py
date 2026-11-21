"""Differentiable native MuJoCo hinge-tree FK with the external canonical240 contract.

MuJoCo is used only to read/validate a compiled asset at construction. Runtime FK is
batched torch and preserves the body-origin (not body-CoM) velocity convention.
"""
from __future__ import annotations

import re
import numpy as np
import torch
from torch import Tensor, nn
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from morphmimic.model.state import StateParts, join_state


class RobotKinematics(nn.Module):
  def __init__(self, parents, joints, tracked, root, *, body_pos, body_quat,
               joint_pos, joint_axis, joint_ref, joint_sign, joint_offset, joint_mask):
    super().__init__()
    self.parents, self.joints = tuple(parents), tuple(joints)
    self.tracked, self.root = tuple(tracked), root
    for name, values in locals().copy().items():
      if name in ('body_pos','body_quat','joint_pos','joint_axis','joint_ref',
                  'joint_sign','joint_offset','joint_mask'):
        self.register_buffer(name, torch.as_tensor(values, dtype=torch.float32).clone())

  @classmethod
  def from_mujoco(cls, model, spec):
    import mujoco
    root = model.body(spec.root_body).id
    if root != 1 or model.body_jntnum[root] != 1:
      raise ValueError('compact FK requires a single free root immediately below world')
    root_joint = model.body_jntadr[root]
    if model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError('compact FK root must have a free joint')
    canonical = {spec.joint_names[n]: c for c,n in enumerate(spec.translator.perm) if n >= 0}
    if spec.translator.untracked:
      raise ValueError('compact FK cannot represent untracked native degrees of freedom')
    joints = [-1] * model.nbody
    jpos = np.zeros((model.nbody,3)); axis = np.zeros_like(jpos)
    ref = np.zeros(model.nbody); sign=np.ones(model.nbody); offset=np.zeros(model.nbody)
    for b in range(root+1, model.nbody):
      count = model.body_jntnum[b]
      if count > 1:
        raise ValueError('compact FK requires at most one hinge per body')
      if count:
        j=int(model.body_jntadr[b]); name=model.joint(j).name
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or name not in canonical:
          raise ValueError(f'compact FK cannot represent joint {name!r}')
        c=canonical[name]; joints[b]=c
        jpos[b]=model.jnt_pos[j]; axis[b]=model.jnt_axis[j]
        ref[b]=model.qpos0[model.jnt_qposadr[j]]
        sign[b]=spec.translator.sign[c]; offset[b]=spec.translator.offset[c]
    return cls(model.body_parentid.tolist(),joints,
               [model.body(name).id for name in spec.tracked_bodies],root,
               body_pos=model.body_pos,body_quat=model.body_quat,joint_pos=jpos,
               joint_axis=axis,joint_ref=ref,joint_sign=sign,joint_offset=offset,
               joint_mask=spec.translator.joint_mask)

  @classmethod
  def from_spec(cls, spec):
    return cls.from_mujoco(spec.entity_cfg_fn().spec_fn().compile(),spec)

  def forward(self, q: Tensor, qd: Tensor, root_pos: Tensor, root_quat: Tensor,
              root_lin_vel: Tensor, root_ang_vel: Tensor) -> Tensor:
    q, qd = q * self.joint_mask, qd * self.joint_mask
    root_quat=nn.functional.normalize(root_quat,dim=-1)
    zero=torch.zeros_like(root_pos)
    identity=torch.cat([torch.ones_like(root_pos[...,:1]),zero],-1)
    positions=[zero]; rotations=[identity]; linear=[zero]; angular=[zero]
    for b,parent in enumerate(self.parents[1:],1):
      if b==self.root:
        p,r,v,w=root_pos,root_quat,root_lin_vel,root_ang_vel
      else:
        pp,pr,pv,pw=positions[parent],rotations[parent],linear[parent],angular[parent]
        p0=pp+quat_apply(pr,self.body_pos[b].expand_as(pp))
        r0=quat_mul(pr,self.body_quat[b].expand_as(pr))
        c=self.joints[b]
        if c>=0:
          angle=(q[...,c]*self.joint_sign[b]+self.joint_offset[b]-self.joint_ref[b])*.5
          rotation=torch.cat([angle.cos().unsqueeze(-1),angle.sin().unsqueeze(-1)*self.joint_axis[b]],-1)
          r=quat_mul(r0,rotation)
          before=quat_apply(r0,self.joint_pos[b].expand_as(pp))
          after=quat_apply(r,self.joint_pos[b].expand_as(pp))
          p=p0+before-after
          axis=quat_apply(r0,self.joint_axis[b].expand_as(pp))
          spin=axis*(qd[...,c]*self.joint_sign[b]).unsqueeze(-1)
          w=pw+spin
          v=pv+torch.cross(pw,p-pp,dim=-1)-torch.cross(spin,after,dim=-1)
        else:
          p,r=p0,r0; w=pw; v=pv+torch.cross(pw,p-pp,dim=-1)
      positions.append(p); rotations.append(r); linear.append(v); angular.append(w)
    def tracked(values): return torch.stack([values[b] for b in self.tracked],-2)
    return join_state(StateParts(q,qd,tracked(positions),tracked(rotations),tracked(linear),tracked(angular)))


def calibrate_foot_heights(model, spec) -> tuple[float,float]:
  """Body-origin height at flat-ground collision contact in the asset's neutral pose.

  Uses the actual foot collision geometry, including sphere/capsule/box/cylinder/
  ellipsoid/mesh support, rather than body-origin heuristics or robot size guesses.
  """
  import mujoco
  data=mujoco.MjData(model); mujoco.mj_kinematics(model,data)
  results=[]
  for name in (spec.tracked_bodies[3],spec.tracked_bodies[6]):
    body=model.body(name).id; support=[]
    for g in range(model.ngeom):
      geom_name=model.geom(g).name
      if not re.fullmatch(spec.foot_geom_regex,geom_name or ''):
        continue
      ancestor=int(model.geom_bodyid[g])
      while ancestor not in (0,body): ancestor=int(model.body_parentid[ancestor])
      if ancestor!=body: continue
      z=data.geom_xmat[g].reshape(3,3)[2]; size=model.geom_size[g]; kind=model.geom_type[g]
      if kind==mujoco.mjtGeom.mjGEOM_SPHERE: radius=size[0]
      elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE: radius=size[0]+abs(z[2])*size[1]
      elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER: radius=size[0]*np.linalg.norm(z[:2])+abs(z[2])*size[1]
      elif kind==mujoco.mjtGeom.mjGEOM_BOX: radius=np.abs(z)@size
      elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID: radius=np.linalg.norm(z*size)
      elif kind==mujoco.mjtGeom.mjGEOM_MESH:
        mesh=int(model.geom_dataid[g]); start=model.mesh_vertadr[mesh]; count=model.mesh_vertnum[mesh]
        radius=-np.min(model.mesh_vert[start:start+count]@z)
      else: raise ValueError(f'unsupported foot contact geometry {kind}')
      support.append(float(data.geom_xpos[g,2]-radius))
    if not support: raise ValueError(f'{spec.name}: no foot collision geometry for {name}')
    height=float(data.xpos[body,2])-min(support)
    if not np.isfinite(height) or height<=0: raise ValueError(f'{spec.name}: invalid foot height {height}')
    results.append(height)
  return tuple(results)
