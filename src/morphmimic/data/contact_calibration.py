"""Conservative per-clip reference-floor estimates from actual collision geometry.

A stationary low sole is kinematic support evidence, not a measured contact label.
Unknown/airborne clips remain explicitly uncalibrated; no percentile-only fallback.
Original motion trajectories and simulator floor/contacts are never changed.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
import math
import re
import numpy as np


def _rotation(quaternion):
  q=np.asarray(quaternion,dtype=np.float64)
  if q.shape[-1]!=4 or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q,axis=-1),1,atol=1e-4,rtol=0):
    raise ValueError('Reference foot orientations must be finite unit wxyz quaternions')
  w,x,y,z=np.moveaxis(q/np.linalg.norm(q,axis=-1,keepdims=True),-1,0)
  return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                   2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                   2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)


class FootContactGeometry:
  """Each foot's fixed collision surfaces in its tracked ankle body frame."""
  def __init__(self,feet):
    if len(feet)!=2 or any(not surfaces for surfaces in feet):
      raise ValueError('Both feet require actual collision surfaces')
    self.feet=feet

  @classmethod
  def from_mujoco(cls,model,spec):
    import mujoco
    data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
    types={int(getattr(mujoco.mjtGeom,'mjGEOM_'+name.upper())):name for name in
           ('sphere','capsule','cylinder','box','ellipsoid','mesh')}
    feet=[]
    for slot in (3,6):
      body=model.body(spec.tracked_bodies[slot]).id;R=data.xmat[body].reshape(3,3)
      surfaces=[]
      for g in range(model.ngeom):
        if not re.fullmatch(spec.foot_geom_regex,model.geom(g).name or ''):continue
        ancestor=int(model.geom_bodyid[g]);articulated=False
        while ancestor not in (0,body):
          articulated |= bool(model.body_jntnum[ancestor]);ancestor=int(model.body_parentid[ancestor])
        if ancestor!=body:continue
        if articulated:raise ValueError('Reference contact geometry cannot assume an articulated distal foot is fixed')
        kind=types.get(int(model.geom_type[g]))
        if kind is None:raise ValueError('Unsupported foot collision primitive')
        surface=dict(kind=kind,center=R.T@(data.geom_xpos[g]-data.xpos[body]),
                     rotation=R.T@data.geom_xmat[g].reshape(3,3),size=model.geom_size[g].copy())
        if kind=='mesh':
          mesh=int(model.geom_dataid[g]);start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
          surface['vertices']=model.mesh_vert[start:start+count].copy()
        surfaces.append(surface)
      feet.append(surfaces)
    return cls(feet)

  @classmethod
  def from_spec(cls,spec):return cls.from_mujoco(spec.entity_cfg_fn().spec_fn().compile(),spec)

  def sole_heights(self,positions,quaternions):
    positions=np.asarray(positions,dtype=np.float64)
    if positions.ndim!=3 or positions.shape[1:]!=(2,3) or not np.isfinite(positions).all():
      raise ValueError('Foot positions must be finite [T,2,3]')
    if np.shape(quaternions)!=(len(positions),2,4):raise ValueError('Foot orientations must be [T,2,4]')
    rotation=_rotation(quaternions);heights=[]
    for foot,surfaces in enumerate(self.feet):
      lower=[]
      for surface in surfaces:
        z=(rotation[:,foot]@surface['rotation'])[:,2]
        center=positions[:,foot,2]+(rotation[:,foot]@surface['center'])[:,2]
        size=surface['size'];kind=surface['kind']
        if kind=='sphere':radius=np.full(len(positions),size[0])
        elif kind=='capsule':radius=size[0]+np.abs(z[:,2])*size[1]
        elif kind=='cylinder':radius=size[0]*np.linalg.norm(z[:,:2],axis=-1)+np.abs(z[:,2])*size[1]
        elif kind=='box':radius=np.abs(z)@size
        elif kind=='ellipsoid':radius=np.linalg.norm(z*size,axis=-1)
        elif kind=='mesh':radius=-np.min(z@surface['vertices'].T,axis=-1)
        else:raise ValueError('Unsupported foot collision primitive')
        lower.append(center-radius)
      heights.append(np.min(lower,axis=0))
    return np.stack(heights,-1)


@dataclass(frozen=True)
class ReferenceFloorCalibration:
  offset_m: float
  valid: bool
  reason: str
  support_samples: int
  spread_m: float | None
  method: str = 'stationary-collision-sole-kinematic-v1'


def calibrate_reference_floor(foot_pos,foot_quat,foot_lin_vel,foot_ang_vel,geometry,*,
                              fps=50.,declared_floor=0.,max_offset=.12,
                              stationary_speed=.08,stationary_angular_speed=.35,
                              minimum_support_seconds=.2,support_band=.015,max_spread=.02):
  """Estimate the clip's reference plane from sustained stationary lowest soles.

  Bounds assume source coordinates are already near their declared flat ground.
  Inherent ambiguity remains: a stationary suspended foot is observationally
  indistinguishable from a source floor offset. This is never a force/contact label.
  """
  if not all(math.isfinite(x) for x in (fps,declared_floor,max_offset,stationary_speed,
      stationary_angular_speed,minimum_support_seconds,support_band,max_spread)) or min(
      fps,max_offset,stationary_speed,stationary_angular_speed,minimum_support_seconds,support_band,max_spread)<=0:
    raise ValueError('Floor calibration bounds must be finite and scales positive')
  sole=geometry.sole_heights(foot_pos,foot_quat)
  v,w=np.asarray(foot_lin_vel),np.asarray(foot_ang_vel)
  if v.shape!=np.shape(foot_pos) or w.shape!=v.shape or not np.isfinite(v).all() or not np.isfinite(w).all() or len(v)==0:
    raise ValueError('Foot velocities must be nonempty finite [T,2,3]')
  low=sole<=sole.min()+support_band
  bounded=np.abs(sole-declared_floor)<=max_offset
  candidates=low&bounded&(np.linalg.norm(v,axis=-1)<=stationary_speed)&(np.linalg.norm(w,axis=-1)<=stationary_angular_speed)
  sustained=np.zeros_like(candidates);minimum=math.ceil(minimum_support_seconds*fps)
  for foot in range(2):
    changes=np.diff(np.pad(candidates[:,foot].astype(np.int8),(1,1)))
    for start,stop in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
      if stop-start>=minimum:sustained[start:stop,foot]=True
  values=sole[sustained];count=len(values)
  if count<minimum:
    return ReferenceFloorCalibration(0.,False,'no sustained stationary low-sole support within declared-floor bounds',count,None)
  offset=float(np.median(values));spread=float(np.quantile(values,.95)-np.quantile(values,.05))
  if spread>max_spread:
    return ReferenceFloorCalibration(0.,False,'stationary support heights disagree',count,spread)
  return ReferenceFloorCalibration(offset,True,'bounded sustained kinematic support; not measured contact',count,spread)


def calibrate_reference_file(path,geometry,spec,*,expected_sha256=None,**kwargs):
  """Standalone source-bound receipt; preserves the original prepared NPZ."""
  from pathlib import Path
  import hashlib,io
  path=Path(path).resolve(strict=True);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
  if expected_sha256 is not None and digest!=expected_sha256:
    raise ValueError(f'Reference SHA256 differs from prepared manifest: {path}')
  with np.load(io.BytesIO(raw),allow_pickle=False) as data:
    names=list(data['body_names']);indices=[names.index(spec.tracked_bodies[i]) for i in (3,6)]
    result=calibrate_reference_floor(data['body_pos_w'][:,indices],data['body_quat_w'][:,indices],
      data['body_lin_vel_w'][:,indices],data['body_ang_vel_w'][:,indices],geometry,fps=float(data['fps']),**kwargs)
  return dict(path=str(path),sha256=digest,robot=spec.name,**asdict(result))
