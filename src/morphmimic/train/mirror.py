"""Sagittal full-window augmentation, gated by exact robot symmetry checks.

Frame ids remain clip identities. Mirrored reference *states* are returned separately
and must be passed into WorldModel.fit_loss(refs=...), including held clip-end frames.
"""
from __future__ import annotations

import re
import numpy as np
import torch
from torch import nn

from morphmimic.constants import JOINT_NAMES
from morphmimic.model.state import StateParts, split_state, join_state


class AsymmetricRobotError(ValueError):
  """The real asset/controls do not support the claimed reflection symmetry."""


def _swap(name):
  for a,b in (('left_','right_'),('l_','r_')):
    if name.startswith(a): return b+name[len(a):]
    if name.startswith(b): return a+name[len(b):]
  return name


class RobotMirror(nn.Module):
  """Tensor transform; production construction must use from_spec for validation."""
  def __init__(self, permutation, signs, joint_mask):
    super().__init__()
    permutation=torch.as_tensor(permutation,dtype=torch.long)
    signs=torch.as_tensor(signs,dtype=torch.float32)
    mask=torch.as_tensor(joint_mask,dtype=torch.float32)
    if permutation.shape!=(29,) or signs.shape!=(29,) or mask.shape!=(29,):
      raise ValueError('mirror requires canonical29 permutation, signs and mask')
    if not torch.equal(permutation[permutation],torch.arange(29)) or not torch.equal(signs*signs[permutation],torch.ones(29)):
      raise ValueError('joint reflection must be an involution')
    if not torch.equal(mask,mask[permutation]): raise ValueError('asymmetric joint availability')
    self.register_buffer('permutation',permutation); self.register_buffer('signs',signs)
    self.register_buffer('joint_mask',mask)
    self.register_buffer('bodies',torch.tensor([0,4,5,6,1,2,3,7,11,12,13,8,9,10]))
    self.register_buffer('polar',torch.tensor([1.,-1.,1.]))
    self.register_buffer('axial',torch.tensor([-1.,1.,-1.]))
    self.register_buffer('quaternion',torch.tensor([1.,-1.,1.,-1.]))
    self.validation={'validated':False}

  @classmethod
  def from_mujoco(cls,model,spec,atol=2e-5):
    """Validate tree, inertia, passive joints, limits and collision geometry.

    Mesh collisions are conservatively rejected: a mesh id or AABB is not a proof
    that mirrored contact surfaces coincide. from_spec additionally checks controls.
    """
    import mujoco
    from morphmimic.model.kinematics import RobotKinematics
    permutation=[JOINT_NAMES.index(_swap(n)) for n in JOINT_NAMES]
    signs=[-1. if 'roll' in n or 'yaw' in n else 1. for n in JOINT_NAMES]
    mirror=cls(permutation,signs,spec.translator.joint_mask)
    def fail(what): raise AsymmetricRobotError(f'{spec.name}: asymmetry in {what}; mirror disabled')
    def check(a,b,what):
      if not np.allclose(a,b,atol=atol,rtol=0): fail(what)
    data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    S=np.diag([1.,-1.,1.]); root=model.body(spec.root_body).id
    root_pos=data.xpos[root].copy()
    ids={model.body(i).name:i for i in range(model.nbody)}
    partners=[]
    for b in range(model.nbody):
      name=model.body(b).name; other=ids.get(_swap(name))
      if other is None: fail(f'body pairing {name}')
      partners.append(other)
      if b==0: continue
      check(data.xpos[b]-root_pos,S@(data.xpos[other]-root_pos),f'body position {name}')
      check(data.xmat[b].reshape(3,3),S@data.xmat[other].reshape(3,3)@S,f'body orientation {name}')
      check(model.body_mass[b],model.body_mass[other],f'body mass {name}')
      check(data.xipos[b]-root_pos,S@(data.xipos[other]-root_pos),f'center of mass {name}')
      R=data.ximat[b].reshape(3,3); O=data.ximat[other].reshape(3,3)
      check(R@np.diag(model.body_inertia[b])@R.T,S@O@np.diag(model.body_inertia[other])@O.T@S,f'inertia tensor {name}')
    for b in range(1,model.nbody):
      if partners[model.body_parentid[b]]!=model.body_parentid[partners[b]]: fail('parent tree')
    for c,native in enumerate(spec.translator.perm):
      if native<0: continue
      other_native=spec.translator.perm[permutation[c]]
      j=model.joint(spec.joint_names[native]).id; k=model.joint(spec.joint_names[other_native]).id
      if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE or model.jnt_type[k]!=mujoco.mjtJoint.mjJNT_HINGE: fail('nonhinge joints')
      native_sign=signs[c]*spec.translator.sign[c]*spec.translator.sign[permutation[c]]
      check(data.xaxis[j],-native_sign*S@data.xaxis[k],f'joint axis {spec.joint_names[native]}')
      check(data.xanchor[j]-root_pos,S@(data.xanchor[k]-root_pos),'joint anchor')
      for field in ('dof_damping','dof_armature','dof_frictionloss'):
        check(getattr(model,field)[model.jnt_dofadr[j]],getattr(model,field)[model.jnt_dofadr[k]],field)
      check(model.jnt_limited[j],model.jnt_limited[k],'joint limit enable')
      if model.jnt_limited[j]:
        lo=spec.translator.sign[c]*(model.jnt_range[j]-spec.translator.offset[c])
        hi=signs[c]*spec.translator.sign[permutation[c]]*(model.jnt_range[k]-spec.translator.offset[permutation[c]])
        check(np.sort(lo),np.sort(hi),'joint limits')
    for b in range(1,model.nbody):
      geom=[i for i in range(model.ngeom) if model.geom_bodyid[i]==b and (model.geom_contype[i] or model.geom_conaffinity[i])]
      other=[i for i in range(model.ngeom) if model.geom_bodyid[i]==partners[b] and (model.geom_contype[i] or model.geom_conaffinity[i])]
      if len(geom)!=len(other): fail('collision count')
      for g in geom:
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH: fail('unverified mesh collision surface')
        matching=[]
        for h in other:
          if model.geom_type[g]!=model.geom_type[h]: continue
          if not np.allclose(data.geom_xpos[g]-root_pos,S@(data.geom_xpos[h]-root_pos),atol=atol,rtol=0): continue
          if not np.allclose(model.geom_size[g],model.geom_size[h],atol=atol,rtol=0): continue
          # Spherical collisions are orientation-free; for other primitives require
          # matching reflected frames (conservative for equivalent rotated boxes).
          if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_SPHERE and not np.allclose(data.geom_xmat[g].reshape(3,3),S@data.geom_xmat[h].reshape(3,3)@S,atol=atol,rtol=0): continue
          matching.append(h)
        if not matching: fail('collision geometry')
        h=matching[0]; other.remove(h)
        for field in ('geom_contype','geom_conaffinity','geom_condim','geom_priority','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap'):
          check(getattr(model,field)[g],getattr(model,field)[h],field)
    # Validate the proposed canonical signs through the actual complete tree at
    # nonzero configurations, including the rigid hand and absent-joint mappings.
    fk=RobotKinematics.from_mujoco(model,spec)
    gen=torch.Generator().manual_seed(1729)
    q=torch.randn(4,29,generator=gen)*.2*mirror.joint_mask
    qd=torch.randn(4,29,generator=gen)*.3*mirror.joint_mask
    pos=torch.randn(4,3,generator=gen)*.2; rot=nn.functional.normalize(torch.randn(4,4,generator=gen),dim=-1)
    v=torch.randn(4,3,generator=gen)*.2; w=torch.randn(4,3,generator=gen)*.2
    actual=split_state(fk(mirror.joints(q),mirror.joints(qd),pos*mirror.polar,rot*mirror.quaternion,v*mirror.polar,w*mirror.axial))
    expected=split_state(mirror.state(fk(q,qd,pos,rot,v,w)))
    for field,a,e in zip(StateParts._fields,actual,expected):
      if field=='body_quat':
        a=a.unsqueeze(-1)*a.unsqueeze(-2); e=e.unsqueeze(-1)*e.unsqueeze(-2)
      check(a.detach().numpy(),e.detach().numpy(),f'nonzero FK {field}')
    mirror.validation={'validated':True,'robot':spec.name,'scope':'rigid_tree','atol':atol}
    return mirror

  @classmethod
  def from_spec(cls,spec,atol=2e-5):
    cfg=spec.entity_cfg_fn(); model=cfg.spec_fn().compile()
    mirror=cls.from_mujoco(model,spec,atol)
    def resolve(mapping,name,default):
      if isinstance(mapping,(int,float)): return float(mapping)
      values=[value for pattern,value in mapping.items() if re.fullmatch(pattern,name)]
      if len(values)>1 and len(set(values))>1:
        raise AsymmetricRobotError(f'{spec.name}: asymmetry validation cannot resolve overlapping control patterns')
      return values[0] if values else default
    def check(values,sign=False,label='controls'):
      native=torch.tensor(values,dtype=torch.float32)
      canonical=spec.translator.to_canonical(native) if sign else spec.translator.to_canonical_delta(native).abs()
      expected=mirror.joints(canonical) if sign else canonical[mirror.permutation]
      if not torch.allclose(canonical,expected,atol=atol,rtol=0):
        raise AsymmetricRobotError(f'{spec.name}: asymmetry in {label}; mirror disabled')
    check([resolve(cfg.init_state.joint_pos,n,0.) for n in spec.joint_names],True,'default joint positions')
    check([resolve(spec.action_scale,n,0.) for n in spec.joint_names],label='action scale')
    for field in ('stiffness','damping','effort_limit','armature'):
      values=[]
      for name in spec.joint_names:
        matched=[a for a in cfg.articulation.actuators if any(re.fullmatch(pattern,name) for pattern in a.target_names_expr)]
        if len(matched)!=1:
          raise AsymmetricRobotError(f'{spec.name}: asymmetry validation needs one actuator per native joint')
        value=getattr(matched[0],field,None)
        values.append(resolve(value,name,0.) if value is not None else 0.)
      check(values,label=field)
    mirror.validation['scope']='rigid_tree_and_controls'
    return mirror

  def joints(self,value): return value[...,self.permutation]*self.signs*self.joint_mask

  def state(self,value):
    p=split_state(value)
    return join_state(StateParts(self.joints(p.joint_pos),self.joints(p.joint_vel),
      p.body_pos[...,self.bodies,:]*self.polar,p.body_quat[...,self.bodies,:]*self.quaternion,
      p.body_lin_vel[...,self.bodies,:]*self.polar,p.body_ang_vel[...,self.bodies,:]*self.axial))

  def window(self,batch,refs,select=None):
    if refs.shape!=batch.state[:,1:].shape: raise ValueError('refs must contain actual transition reference states')
    ctx=torch.cat([batch.ctx[...,:1],self.joints(batch.ctx[...,1:30]),batch.ctx[...,30:]*self.polar],-1)
    result=batch._replace(state=self.state(batch.state),action=self.joints(batch.action),ctx=ctx)
    if batch.controller is not None:
      record=batch.controller
      mirrored=record._replace(before=record.before._replace(queue=self.joints(record.before.queue)),
        after=record.after._replace(queue=self.joints(record.after.queue)),targets=self.joints(record.targets))
      if select is not None:
        def controller_choose(a,b): return torch.where(select.reshape(-1,*([1]*(a.ndim-1))),a,b)
        mirrored=mirrored._replace(
          before=mirrored.before._replace(queue=controller_choose(mirrored.before.queue,record.before.queue)),
          after=mirrored.after._replace(queue=controller_choose(mirrored.after.queue,record.after.queue)),
          targets=controller_choose(mirrored.targets,record.targets))
      result=result._replace(controller=mirrored)
    transformed=self.state(refs)
    if select is not None:
      if select.shape!=(len(batch.state),) or select.dtype!=torch.bool: raise ValueError('select must be bool [B]')
      def choose(a,b): return torch.where(select.reshape(-1,*([1]*(a.ndim-1))),a,b)
      result=result._replace(state=choose(result.state,batch.state),action=choose(result.action,batch.action),ctx=choose(result.ctx,batch.ctx))
      transformed=choose(transformed,refs)
    return result,transformed
