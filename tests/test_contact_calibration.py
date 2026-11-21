"""Reference-floor evidence, exact sole geometry and structured target calibration."""
import unittest
import numpy as np
import torch


class ReferenceFloorTests(unittest.TestCase):
  def fixture(self,frames=50):
    from morphmimic.data.contact_calibration import FootContactGeometry
    geometry=FootContactGeometry([[dict(kind='sphere',center=np.array([0.,0.,-.02]),rotation=np.eye(3),size=np.array([.02,0.,0.]))]]*2)
    p=np.zeros((frames,2,3));p[:,:,2]=.04+.07
    q=np.zeros((frames,2,4));q[:,:,0]=1
    return geometry,p,q,np.zeros_like(p),np.zeros_like(p)

  def test_stationary_collision_surface_recovers_reference_offset(self):
    from morphmimic.data.contact_calibration import calibrate_reference_floor
    geometry,p,q,v,w=self.fixture()
    result=calibrate_reference_floor(p,q,v,w,geometry)
    self.assertTrue(result.valid);self.assertAlmostEqual(result.offset_m,.07,places=8)
    self.assertEqual(result.support_samples,100)
    self.assertIn('kinematic',result.method)

  def test_airborne_and_insufficient_evidence_never_get_a_quantile_contact_label(self):
    from morphmimic.data.contact_calibration import calibrate_reference_floor
    geometry,p,q,v,w=self.fixture()
    p[:,:,2]=.4
    self.assertFalse(calibrate_reference_floor(p,q,v,w,geometry).valid)
    p[:,:,2]=.11;v[:,:,2]=.2
    result=calibrate_reference_floor(p,q,v,w,geometry)
    self.assertFalse(result.valid);self.assertEqual(result.offset_m,0)
    self.assertFalse(calibrate_reference_floor(p[:5],q[:5],v[:5]*0,w[:5],geometry).valid)
    # A jump's near-zero apex velocity must not redefine its floor.
    t=np.linspace(-1,1,len(p));p[:,:,2]=(.04+.3*(1-t*t))[:,None]
    v[:,:,2]=(-.6*t)[:,None]
    self.assertFalse(calibrate_reference_floor(p,q,v,w,geometry).valid)

  def test_rotated_box_sole_uses_real_geometry_and_invalid_poses_fail(self):
    from morphmimic.data.contact_calibration import FootContactGeometry,calibrate_reference_floor
    geometry=FootContactGeometry([[dict(kind='box',center=np.zeros(3),rotation=np.eye(3),size=np.array([.1,.02,.03]))]]*2)
    p=np.zeros((20,2,3));p[:,:,2]=.1
    q=np.zeros((20,2,4));q[:,:,0]=2**-.5;q[:,:,2]=2**-.5
    np.testing.assert_allclose(geometry.sole_heights(p,q),0,atol=1e-12)
    q[:,:,0]=0
    with self.assertRaises(ValueError):calibrate_reference_floor(p,q,np.zeros_like(p),np.zeros_like(p),geometry)

  def test_reward_offset_affects_only_reference_and_unknown_contact_is_masked(self):
    from morphmimic.model.state import StateParts,join_state
    from morphmimic.model.rewards_torch import RewardCfg,contact_reward_terms
    q=torch.zeros(2,14,4);q[...,0]=1
    p=torch.zeros(2,14,3);p[:,3,2]=.04;p[:,6,2]=.04
    state=join_state(StateParts(torch.zeros(2,29),torch.zeros(2,29),p,q,torch.zeros_like(p),torch.zeros_like(p)))
    ref=state.clone();ref[:,58:100].reshape(2,14,3)[:,:,2]+=.07
    cfg=RewardCfg(w_contact_agreement=.4,w_foot_slip=-.2,foot_height=(.04,.04))
    expected=contact_reward_terms(state,state,cfg)
    actual=contact_reward_terms(state,ref,cfg,reference_floor=torch.tensor([.07,.07]),reference_floor_valid=torch.tensor([True,False]))
    torch.testing.assert_close(actual['contact_agreement'][0],expected['contact_agreement'][0])
    self.assertEqual(actual['contact_agreement'][1].item(),0)
    torch.testing.assert_close(actual['foot_slip'],expected['foot_slip'])
    # Supplying floor metadata must preserve finite differentiability and compile.
    state.requires_grad_()
    fn=torch.compile(lambda s,r,f,valid:contact_reward_terms(s,r,cfg,reference_floor=f,reference_floor_valid=valid)['contact_agreement'],backend='eager',fullgraph=True)
    fn(state,ref,torch.full((2,),.07),torch.ones(2,dtype=torch.bool)).sum().backward()
    self.assertTrue(torch.isfinite(state.grad).all())

  def test_real_asset_geometry_matches_neutral_collision_height(self):
    import mujoco
    from morphmimic.robots import get_robot
    from morphmimic.model.kinematics import calibrate_foot_heights
    from morphmimic.data.contact_calibration import FootContactGeometry
    for name in ('q1','bumi','g1'):
      with self.subTest(robot=name):
        spec=get_robot(name);model=spec.entity_cfg_fn().spec_fn().compile();data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
        ids=[model.body(spec.tracked_bodies[i]).id for i in (3,6)]
        geometry=FootContactGeometry.from_mujoco(model,spec)
        sole=geometry.sole_heights(data.xpos[ids][None],data.xquat[ids][None])[0]
        np.testing.assert_allclose(data.xpos[ids,2]-sole,calibrate_foot_heights(model,spec),atol=1e-8)


class CorrectionScaleTests(unittest.TestCase):
  def test_correction_target_rms_uses_next_velocity_and_checkpointed_scale(self):
    from morphmimic.model.world_model import WorldModel,WorldModelCfg,joint_correction_target,joint_correction_rms
    from morphmimic.model.state import StateParts,join_state,StateNormalizer,split_state
    quat=torch.zeros(4,14,4);quat[...,0]=1
    state=join_state(StateParts(torch.zeros(4,29),torch.ones(4,29),torch.zeros(4,14,3),quat,torch.zeros(4,14,3),torch.zeros(4,14,3)))
    next_state=state.clone();next_state[:,:29]=.045;next_state[:,29:58]=2
    mask=torch.ones(29);mask[14]=0
    target=joint_correction_target(state,next_state,.02,mask)
    torch.testing.assert_close(target,torch.full((4,29),.005)*mask)
    scale=joint_correction_rms(state,next_state,.02,mask)
    torch.testing.assert_close(scale,torch.full((29,),.005)*mask)
    model=WorldModel(WorldModelCfg(hidden=(8,),joint_dynamics='semi_implicit'),StateNormalizer(),torch.tensor([[-3.,3.]]).expand(29,2),torch.zeros(29),joint_mask=mask)
    before=model.delta_scale().clone();model.set_joint_correction_scale(scale)
    torch.testing.assert_close(model.joint_correction_scale,scale.clamp_min(1e-5))
    torch.testing.assert_close(model.delta_scale(),before)
    with torch.no_grad():model.dynamics_mlp[-1].bias[:29]=1
    out=split_state(model.step(state,torch.zeros(4,29),torch.zeros(4,33)))
    torch.testing.assert_close(out.joint_pos,torch.full((4,29),.025)*mask)
    restored=WorldModel(model.cfg,StateNormalizer(),model.joint_limits,model.default_joint_pos,joint_mask=mask)
    restored.load_state_dict(model.state_dict());torch.testing.assert_close(restored.joint_correction_scale,model.joint_correction_scale)
    with self.assertRaises(ValueError):model.set_joint_correction_scale(torch.full((29,),float('nan')))


class NativeJacobianTests(unittest.TestCase):
  def test_differentiable_body_origin_jacobians_match_native_mujoco(self):
    import mujoco
    from morphmimic.robots import get_robot
    from morphmimic.model.kinematics import RobotKinematics
    from morphmimic.model.state import split_state
    for name in ('q1','bumi','g1'):
      with self.subTest(robot=name):
        spec=get_robot(name);model=spec.entity_cfg_fn().spec_fn().compile();data=mujoco.MjData(model)
        fk=RobotKinematics.from_mujoco(model,spec).double()
        q=torch.linspace(-.1,.1,29,dtype=torch.float64)*torch.tensor(spec.translator.joint_mask)
        zero=torch.zeros(1,3,dtype=torch.float64);root=torch.tensor([[.2,-.1,.8]],dtype=torch.float64)
        orientation=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
        def positions(value):
          return split_state(fk(value[None],torch.zeros(1,29,dtype=torch.float64),root,orientation,zero,zero)).body_pos.reshape(-1)
        actual=torch.autograd.functional.jacobian(positions,q).numpy()
        data.qpos[:3]=root[0];data.qpos[3:7]=orientation[0]
        native=spec.translator.to_native(q,torch.zeros(len(spec.joint_names),dtype=torch.float64))
        for j,joint in enumerate(spec.joint_names):data.qpos[model.joint(joint).qposadr]=native[j]
        mujoco.mj_forward(model,data);expected=np.zeros((42,29))
        for body,name_body in enumerate(spec.tracked_bodies):
          jac=np.zeros((3,model.nv));angular=np.zeros_like(jac)
          mujoco.mj_jacBody(model,data,jac,angular,model.body(name_body).id)
          for c,n in enumerate(spec.translator.perm):
            if n>=0:expected[body*3:body*3+3,c]=jac[:,model.joint(spec.joint_names[n]).dofadr[0]]*spec.translator.sign[c]
        np.testing.assert_allclose(actual,expected,atol=3e-7,rtol=2e-6)


class BulkFloorReceiptTests(unittest.TestCase):
  def test_bulk_receipt_pins_manifest_source_and_refuses_overwrite(self):
    from pathlib import Path
    import tempfile,json,hashlib
    from unittest.mock import patch
    from morphmimic.scripts.calibrate_contact_floor import calibrate_manifest
    from morphmimic.constants import TRACKED_BODY_NAMES
    geometry,p,q,v,w=ReferenceFloorTests().fixture(frames=20)
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory);source=root/'motion.npz';manifest=root/'manifest.json';output=root/'contact-floor.json'
      pos=np.zeros((20,14,3));quat=np.zeros((20,14,4));quat[...,0]=1
      pos[:,[3,6]]=p;quat[:,[3,6]]=q
      np.savez(source,body_pos_w=pos,body_quat_w=quat,body_lin_vel_w=np.zeros_like(pos),body_ang_vel_w=np.zeros_like(pos),body_names=np.array(TRACKED_BODY_NAMES),fps=np.array(50))
      digest=hashlib.sha256(source.read_bytes()).hexdigest()
      row=dict(robot='q1',name='fixture',dataset='seed',split='train',output=str(source),output_sha256=digest)
      manifest.write_text(json.dumps(dict(schema_version=1,complete=True,clips=[row])))
      from types import SimpleNamespace
      spec=SimpleNamespace(name='q1',tracked_bodies=TRACKED_BODY_NAMES)
      with patch('morphmimic.scripts.calibrate_contact_floor.get_robot',return_value=spec),patch('morphmimic.scripts.calibrate_contact_floor.FootContactGeometry.from_spec',return_value=geometry):
        result=calibrate_manifest(manifest,output,workers=2)
        self.assertEqual(result['summary']['q1'],{'valid':1,'unknown':0})
        self.assertEqual(result['clips'][0]['sha256'],digest)
        self.assertEqual(result['prepared_manifest_sha256'],hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertGreaterEqual(result['elapsed_seconds'],0)
        with self.assertRaises(FileExistsError):calibrate_manifest(manifest,output,workers=2)
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):calibrate_manifest(manifest,root/'second.json',workers=2)
        self.assertFalse((root/'second.json').exists())
