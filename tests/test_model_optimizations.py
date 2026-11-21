"""Physical invariants for optional world-model, mirror and reward modes."""
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from morphmimic.model.state import StateParts, StateNormalizer, join_state, split_state
from morphmimic.model.world_model import WorldModel, WorldModelCfg
from morphmimic.model.rewards_torch import RewardCfg, tracking_reward


def state(batch=2):
  quat = torch.zeros(batch, 14, 4); quat[..., 0] = 1
  return join_state(StateParts(torch.zeros(batch, 29), torch.ones(batch, 29),
                              torch.zeros(batch, 14, 3), quat,
                              torch.zeros(batch, 14, 3), torch.zeros(batch, 14, 3)))


def wm(**kwargs):
  mask = torch.ones(29); mask[14] = 0
  return WorldModel(WorldModelCfg(hidden=(16,), **kwargs), StateNormalizer(),
                    torch.tensor([[-3., 3.]]).expand(29, 2), torch.zeros(29), joint_mask=mask)


class StructuredDynamicsTests(unittest.TestCase):
  def test_semi_implicit_uses_next_velocity_and_separate_correction_scale(self):
    model = wm(joint_dynamics='semi_implicit', dt=.02, joint_position_correction_scale=.003)
    with torch.no_grad():
      model.dynamics_mlp[-1].bias[:29] = 2
      model.dynamics_mlp[-1].bias[29:58] = 3
      model.set_residual_scale(torch.full((268,), .2))
    s = state()
    out = split_state(model.step(s, torch.zeros(2,29), torch.zeros(2,33)))
    expected_v = torch.full((2,29), 1.6) * model.joint_mask
    torch.testing.assert_close(out.joint_vel, expected_v)
    torch.testing.assert_close(out.joint_pos, (.02 * expected_v + .006) * model.joint_mask)
    # Integration correction has its own units; the state-fitting scale stays data calibrated.
    torch.testing.assert_close(model.delta_scale()[:29], torch.full((29,), .2))

  def test_legacy_identity_and_differentiable_structured_rollout(self):
    s = state(); s[:,14] = 0; s[:,43] = 0
    torch.testing.assert_close(wm().step(s, torch.zeros(2,29), torch.zeros(2,33)), s)
    model = wm(joint_dynamics='semi_implicit')
    with torch.no_grad(): model.dynamics_mlp[-1].weight.normal_(std=.01)
    action = torch.randn(2,3,29,requires_grad=True)
    model.rollout(s,action,torch.zeros(2,33)).square().mean().backward()
    self.assertTrue(torch.isfinite(action.grad).all())
    self.assertGreater(action.grad.abs().sum().item(),0)
    torch.testing.assert_close(action.grad[:,:,14],torch.zeros(2,3))

  def test_invalid_mode_and_unattached_compact_are_rejected(self):
    with self.assertRaises(ValueError): wm(joint_dynamics='magic')
    with self.assertRaises(ValueError): wm(dt=0)
    with self.assertRaises(ValueError): wm(joint_position_correction_scale=0)
    with self.assertRaises(ValueError): wm(state_mode='compact_fk')


class ContactRewardTests(unittest.TestCase):
  def test_opt_in_preserves_default_and_exposes_valid_reward_bound(self):
    cfg = RewardCfg()
    self.assertAlmostEqual(cfg.reward_upper_bound, .1)
    enabled = replace(cfg, w_contact_agreement=.4, w_foot_slip=-.2,
                      foot_height=(.08,.09), contact_height_margin=.015,
                      contact_height_temperature=.005, contact_speed_temperature=.1)
    self.assertAlmostEqual(enabled.reward_upper_bound,.108)
    s=state(); parts=split_state(s); parts.body_pos[:,3,2]=.08; parts.body_pos[:,6,2]=.09
    a=torch.zeros(2,29); limits=torch.tensor([[-3.,3.]]).expand(29,2)
    legacy, terms=tracking_reward(s,s,a,a,limits,cfg)
    self.assertNotIn('foot_slip',terms)
    good, terms=tracking_reward(s,s,a,a,limits,enabled)
    self.assertTrue(torch.all(good <= enabled.reward_upper_bound + 1e-7))
    self.assertTrue(torch.all(terms['foot_slip']==0))
    slip=s.clone(); split_state(slip).body_lin_vel[:,[3,6],0]=1
    slipped, slip_terms=tracking_reward(slip,s,a,a,limits,enabled)
    self.assertTrue(torch.all(slip_terms['foot_slip'] < 0))
    self.assertTrue(torch.all(slipped < good))
    self.assertTrue(torch.all(legacy <= cfg.reward_upper_bound + 1e-7))
    slip.requires_grad_(); tracking_reward(slip,s,a,a,limits,enabled)[0].sum().backward()
    self.assertTrue(torch.isfinite(slip.grad).all())

  def test_contact_terms_require_valid_calibration(self):
    with self.assertRaises(ValueError): RewardCfg(w_contact_agreement=.4)
    with self.assertRaises(ValueError): RewardCfg(w_foot_slip=.2,foot_height=(.08,.08))
    with self.assertRaises(ValueError): RewardCfg(contact_height_temperature=0)


class RealAssetKinematicsTests(unittest.TestCase):
  def test_native_q1_bumi_g1_fk_pose_velocity_and_gradient(self):
    import mujoco
    import numpy as np
    from morphmimic.robots import get_robot
    from morphmimic.model.kinematics import RobotKinematics
    for name in ('q1','bumi','g1'):
      with self.subTest(robot=name):
        spec=get_robot(name); model=spec.entity_cfg_fn().spec_fn().compile()
        fk=RobotKinematics.from_mujoco(model,spec)
        data=mujoco.MjData(model)
        q=torch.randn(3,29)*.12; qd=torch.randn(3,29)*.15
        q*=torch.tensor(spec.translator.joint_mask); qd*=torch.tensor(spec.translator.joint_mask)
        q.requires_grad_()
        root_pos=torch.tensor([[.2,-.1,.8]]).expand(3,3)
        root_quat=torch.tensor([[.9659258,.258819,0,0]]).expand(3,4)
        root_v=torch.tensor([[.1,.2,.3]]).expand(3,3)
        root_w=torch.tensor([[.2,-.3,.1]]).expand(3,3)
        result=split_state(fk(q,qd,root_pos,root_quat,root_v,root_w))
        for b in range(3):
          data.qpos[:3]=root_pos[b].numpy(); data.qpos[3:7]=root_quat[b].numpy()
          native=spec.translator.to_native(q[b].detach(),torch.zeros(len(spec.joint_names)))
          velocity=spec.translator.to_native_delta(qd[b])
          for k,j in enumerate(spec.joint_names):
            data.qpos[model.joint(j).qposadr]=native[k].item()
            data.qvel[model.joint(j).dofadr]=velocity[k].item()
          data.qvel[:3]=root_v[b].numpy()
          # MuJoCo free-joint angular qvel is local; external state is world frame.
          from mjlab.utils.lab_api.math import quat_apply_inverse
          data.qvel[3:6]=quat_apply_inverse(root_quat[b:b+1],root_w[b:b+1])[0].numpy()
          mujoco.mj_forward(model,data)
          for j,body in enumerate(spec.tracked_bodies):
            bid=model.body(body).id
            np.testing.assert_allclose(result.body_pos[b,j].detach(),data.xpos[bid],atol=3e-6)
            actual=result.body_quat[b,j].detach().numpy(); expected=data.xquat[bid]
            np.testing.assert_allclose(np.outer(actual,actual),np.outer(expected,expected),atol=3e-6)
            vel=np.zeros(6); mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,vel,0)
            # mj_objectVelocity returns body-CoM linear velocity; shift to body origin.
            origin_v=vel[3:]+np.cross(vel[:3],data.xpos[bid]-data.xipos[bid])
            np.testing.assert_allclose(result.body_lin_vel[b,j].detach(),origin_v,atol=4e-6)
            np.testing.assert_allclose(result.body_ang_vel[b,j].detach(),vel[:3],atol=4e-6)
        result.body_pos.square().sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(q.grad.abs().sum().item(),0)
        compact=WorldModel(WorldModelCfg(hidden=(16,),state_mode='compact_fk',joint_dynamics='semi_implicit'),
                           StateNormalizer(),torch.tensor([[-3.,3.]]).expand(29,2),torch.zeros(29),
                           joint_mask=torch.tensor(spec.translator.joint_mask),kinematics=fk)
        pred=compact.step(join_state(result),torch.zeros(3,29),torch.zeros(3,33))
        self.assertEqual(pred.shape,(3,240))
        self.assertEqual(compact.dynamics_mlp[-1].out_features,73)
        self.assertTrue(torch.isfinite(pred).all())





class MirrorTests(unittest.TestCase):
  def test_full_window_transforms_reference_state_context_and_axial_vectors(self):
    from morphmimic.train.mirror import RobotMirror
    from morphmimic.constants import JOINT_NAMES
    from morphmimic.train.replay import WindowBatch
    permutation=[JOINT_NAMES.index(n.replace('left_','right_') if n.startswith('left_') else n.replace('right_','left_') if n.startswith('right_') else n) for n in JOINT_NAMES]
    signs=[-1. if ('roll' in n or 'yaw' in n) else 1. for n in JOINT_NAMES]
    mirror=RobotMirror(permutation,signs,torch.ones(29))
    s=state(6).reshape(2,3,240)
    s[...,:58]=torch.randn(2,3,58)
    p=split_state(s); p.body_pos.copy_(torch.randn_like(p.body_pos))
    p.body_lin_vel.copy_(torch.randn_like(p.body_lin_vel)); p.body_ang_vel.copy_(torch.randn_like(p.body_ang_vel))
    p.body_quat.copy_(torch.nn.functional.normalize(torch.randn_like(p.body_quat),dim=-1))
    batch=WindowBatch(s,torch.randn(2,2,29),torch.randn(2,2),torch.tensor([[False,True],[False,False]]),torch.randn(2,33),torch.tensor([[2,2,2],[3,4,5]]))
    refs=s[:,:-1].clone()+.01; refs[...,100:156]=s[:,:-1,100:156]
    mirrored, reflected_refs=mirror.window(batch,refs)
    back, back_refs=mirror.window(mirrored,reflected_refs)
    for actual,expected in zip(back,batch): torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(back_refs,refs)
    torch.testing.assert_close(reflected_refs,mirror.state(refs))
    self.assertFalse(torch.equal(reflected_refs,refs))
    torch.testing.assert_close(mirrored.frame,batch.frame)
    torch.testing.assert_close(mirrored.ctx[:,0],batch.ctx[:,0])
    torch.testing.assert_close(mirrored.ctx[:,-3:],batch.ctx[:,-3:]*torch.tensor([1.,-1.,1.]))
    torch.testing.assert_close(split_state(mirrored.state).body_ang_vel[...,3,:],p.body_ang_vel[...,6,:]*torch.tensor([-1.,1.,-1.]))
    selected, selected_refs=mirror.window(batch,refs,select=torch.tensor([True,False]))
    torch.testing.assert_close(selected.state[1],batch.state[1]); torch.testing.assert_close(selected_refs[1],refs[1])
    torch.testing.assert_close(selected.state[0],mirrored.state[0])
    from morphmimic.model.rewards_torch import termination_features
    reward_cfg=RewardCfg(w_contact_agreement=.3,w_foot_slip=-.2,foot_height=(.04,.04))
    limits=torch.tensor([[-3.,3.]]).expand(29,2)
    reward,_=tracking_reward(batch.state[:,1:],refs,batch.action,batch.action*.5,limits,reward_cfg)
    mirrored_reward,_=tracking_reward(mirrored.state[:,1:],reflected_refs,mirrored.action,mirrored.action*.5,limits,reward_cfg)
    torch.testing.assert_close(reward,mirrored_reward)
    for a,b in zip(termination_features(batch.state[:,1:],refs),termination_features(mirrored.state[:,1:],reflected_refs)):
      torch.testing.assert_close(a,b)
    model=wm(); model.fit_loss(selected,SimpleNamespace(state=lambda _: (_ for _ in ()).throw(AssertionError('must use actual refs'))),refs=selected_refs)[0].backward()

  def test_real_assets_are_validated_and_asymmetry_is_explicit(self):
    from morphmimic.train.mirror import RobotMirror, AsymmetricRobotError
    from morphmimic.robots import get_robot
    from morphmimic.model.kinematics import calibrate_foot_heights
    for name in ('q1','bumi','g1'):
      with self.subTest(robot=name):
        spec=get_robot(name); model=spec.entity_cfg_fn().spec_fn().compile()
        heights=calibrate_foot_heights(model,spec)
        self.assertTrue(all(0<h<.25 for h in heights))
        try:
          mirror=RobotMirror.from_spec(spec)
        except AsymmetricRobotError as e:
          self.assertIn(name,str(e)); self.assertIn('asymmetr',str(e).lower())
        else:
          self.assertTrue(mirror.validation['validated'])
          self.assertEqual(mirror.validation['robot'],name)

  def test_symmetric_native_tree_passes_and_inertia_asymmetry_fails(self):
    import mujoco
    from morphmimic.constants import JOINT_NAMES,TRACKED_BODY_NAMES
    from morphmimic.robots.spec import Translator
    from morphmimic.train.mirror import RobotMirror,AsymmetricRobotError
    def chain(names):
      out=''
      for i,n in enumerate(names):
        body=n.replace('_joint','_link')
        if n=='waist_pitch_joint': body='torso_link'
        axis='1 0 0' if 'roll' in n else '0 0 1' if 'yaw' in n else '0 1 0'
        y=.2 if i==0 and n.startswith('left_') else -.2 if i==0 and n.startswith('right_') else 0
        out+=f'<body name="{body}" pos="0 {y} -.1"><joint name="{n}" axis="{axis}" range="-2 2"/><geom type="sphere" size=".02" mass=".1"/>'
      return out+'</body>'*len(names)
    legs=chain(JOINT_NAMES[:6])+chain(JOINT_NAMES[6:12])
    waist=chain(JOINT_NAMES[12:15])
    arms=chain(JOINT_NAMES[15:22])+chain(JOINT_NAMES[22:])
    xml='<mujoco><compiler angle="radian"/><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".1" mass="1"/>'+legs+waist+arms+'</body></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml)
    spec=SimpleNamespace(name='symmetric_fixture',root_body='pelvis',joint_names=JOINT_NAMES,tracked_bodies=TRACKED_BODY_NAMES,
                         translator=Translator.identity(29),action_scale={'.*':.25})
    mirror=RobotMirror.from_mujoco(model,spec)
    self.assertTrue(mirror.validation['validated'])
    motors=[SimpleNamespace(target_names_expr=(n,),stiffness=5.,damping=1.,effort_limit=10.,armature=.1) for n in JOINT_NAMES]
    cfg=SimpleNamespace(spec_fn=lambda:SimpleNamespace(compile=lambda:model),
                        init_state=SimpleNamespace(joint_pos={'.*':0.}),
                        articulation=SimpleNamespace(actuators=motors))
    spec.entity_cfg_fn=lambda:cfg
    self.assertEqual(RobotMirror.from_spec(spec).validation['scope'],'rigid_tree_and_controls')
    motors[3].stiffness=6.
    with self.assertRaisesRegex(AsymmetricRobotError,'stiffness'):
      RobotMirror.from_spec(spec)
    motors[3].stiffness=5.
    spec.action_scale={n:(.3 if n=='left_knee_joint' else .25) for n in JOINT_NAMES}
    with self.assertRaisesRegex(AsymmetricRobotError,'action scale'):
      RobotMirror.from_spec(spec)
    model.body_mass[model.body('left_knee_link').id]*=1.1
    with self.assertRaisesRegex(AsymmetricRobotError,'asymmetr'):
      RobotMirror.from_mujoco(model,spec)


class ReferenceSkipTests(unittest.TestCase):
  def test_common_latent_reconstruction_skip_and_robot_controls(self):
    from morphmimic.model.policy import Policy,PolicyCfg
    policy=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,hist_len=1,human=True,
                            morph_dim=154,ref_reconstruction_skip=True))
    morph=torch.zeros(3,154); morph[:,0]=torch.tensor([-1.,0.,1.]); morph[:,-29:]=1
    morph[1,-29+14]=0
    q0=torch.stack([torch.full((29,),.1),torch.full((29,),.2),torch.full((29,),.3)])
    scales=torch.stack([torch.full((29,),.25),torch.full((29,),.5),torch.full((29,),.75)])
    policy.set_reference_controls(morph,q0,scales)
    policy.set_reference_stats(torch.full((670,),.4),torch.full((670,),.2))
    with torch.no_grad():
      policy.decoder.reference_skip_gain.fill_(1.)
      policy.decoder.net[-1].weight.zero_(); policy.decoder.net[-1].bias.fill_(.01)
      policy.decoder.reference_head.net[-1].weight.zero_(); policy.decoder.reference_head.net[-1].bias.fill_(2)
    z=torch.randn(3,4)
    out=policy.decoder(z,torch.zeros(3,93),morph)
    expected=((.8-q0)/scales+.01)*morph[:,-29:]
    torch.testing.assert_close(out,expected)
    self.assertEqual(policy.reconstruct(z,morph).shape,(3,670))
    # Both interface encoders feed the exact same reconstruction/decoder path.
    policy.set_morphology(morph[0]); policy.set_joint_mask(morph[0,-29:])
    goal=torch.randn(2,670); human=torch.randn(2,2254); hist=torch.zeros(2,93)
    torch.testing.assert_close(policy(hist,goal),policy.decoder(policy.encode(goal),hist))
    torch.testing.assert_close(policy.forward_human(hist,human),policy.decoder(policy.encode_human(human),hist))
    clone=Policy(policy.cfg); clone.load_state_dict(policy.state_dict())
    torch.testing.assert_close(clone.decoder(z,torch.zeros(3,93),morph),out)
    unknown=morph[0].clone(); unknown[0]=.7
    self.assertFalse(torch.isfinite(policy.decoder(z[:1],torch.zeros(1,93),unknown)).all())

  def test_reconstruction_losses_train_both_encoders_and_shared_head(self):
    from morphmimic.model.policy import Policy,PolicyCfg
    policy=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,hist_len=1,human=True,
                            ref_reconstruction_skip=True))
    policy.set_reference_controls(torch.empty(1,0),torch.zeros(1,29),torch.ones(1,29))
    policy.set_reference_stats(torch.zeros(670),torch.ones(670))
    target=torch.randn(3,670)
    a=policy.reconstruct(policy.encode(target)); b=policy.reconstruct(policy.encode_human(torch.randn(3,2254)))
    ((a-target).square().mean()+(b-target).square().mean()).backward()
    for module in (policy.encoder,policy.human_encoder,policy.decoder.reference_head):
      self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()),0)
    self.assertEqual(len(list(policy.parameters())),len({id(p) for p in policy.parameters()}))

  def test_reference_calibration_rejects_bad_shapes_and_missing_scales(self):
    from morphmimic.model.policy import Policy,PolicyCfg
    policy=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,hist_len=1,ref_reconstruction_skip=True))
    with self.assertRaises(ValueError): policy.set_reference_controls(torch.empty(1,0),torch.zeros(1,29),torch.zeros(1,29))
    with self.assertRaises(ValueError): policy.set_reference_stats(torch.zeros(670),torch.zeros(670))


class ReferenceSupervisionTests(unittest.TestCase):
  def fixture(self):
    from morphmimic.model.policy import Policy,PolicyCfg
    from morphmimic.train.replay import StartBatch
    policy=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,hist_len=1,human=True,ref_reconstruction_skip=True))
    mask=torch.ones(29); mask[14]=0; policy.set_joint_mask(mask)
    refs=state(3).unsqueeze(1).expand(3,10,240).clone(); refs[:,:,:29]=torch.randn(3,10,29)
    poses=torch.randn(3,1,24,7); poses[...,3:]=torch.nn.functional.normalize(poses[...,3:],dim=-1)
    valid=torch.ones(3,24,dtype=torch.bool); valid[2,20]=False
    calls=[]
    def human_future(frames,K): calls.append(K); return poses[frames,:K]
    library=SimpleNamespace(future=lambda frames,K:refs[frames,:K],human_future=human_future,
                            human_valid_at=lambda frames:valid[frames],has_human=True)
    trainer=SimpleNamespace(policy=policy,library=library,cfg=SimpleNamespace(goal_anchor='reference'),
                            goal_norm=torch.nn.Identity(),human_norm=torch.nn.Identity(),heading_norm=torch.nn.Identity())
    start=StartBatch(state(3).requires_grad_(),torch.zeros(3,1,93),torch.zeros(3,33),torch.arange(3))
    return trainer,start,refs,poses,calls

  def test_motion_and_human_loss_gradients_do_not_train_decoder_or_feedback_state(self):
    from morphmimic.train.reference_reconstruction import reference_reconstruction_loss
    for human in (False,True):
      trainer,start,refs,poses,calls=self.fixture()
      loss=reference_reconstruction_loss(trainer,start,human=human); loss.backward()
      active=trainer.policy.human_encoder if human else trainer.policy.encoder
      inactive=trainer.policy.encoder if human else trainer.policy.human_encoder
      self.assertGreater(sum(p.grad.abs().sum().item() for p in active.parameters()),0)
      self.assertTrue(all(p.grad is None for p in inactive.parameters()))
      self.assertTrue(all(p.grad is None for p in trainer.policy.decoder.net.parameters()))
      self.assertIsNone(start.state.grad)
      self.assertGreater(sum(p.grad.abs().sum().item() for p in trainer.policy.decoder.reference_head.parameters()),0)
      if human:self.assertEqual(calls,[1])

  def test_absent_joints_and_missing_human_rows_do_not_enter_loss(self):
    from morphmimic.train.reference_reconstruction import reference_reconstruction_loss
    trainer,start,refs,poses,calls=self.fixture()
    before=reference_reconstruction_loss(trainer,start,human=True)
    refs[2]=state(1)[0]*100
    poses[2]=float('nan')
    refs[:2,:,14]=10000; refs[:2,:,43]=-10000
    after=reference_reconstruction_loss(trainer,start,human=True)
    torch.testing.assert_close(before,after)
    self.assertTrue(torch.isfinite(after))

  def test_visible_future_human_is_rejected_and_empty_paired_rows_return_zero(self):
    from morphmimic.train.reference_reconstruction import reference_reconstruction_loss
    trainer,start,refs,poses,calls=self.fixture()
    with self.assertRaises(ValueError):
      reference_reconstruction_loss(trainer,start,human=True,frame_mask=torch.ones(3,10,dtype=torch.bool))
    trainer.library.human_valid_at=lambda frames:torch.zeros(len(frames),24,dtype=torch.bool)
    loss=reference_reconstruction_loss(trainer,start,human=True)
    self.assertEqual(loss.item(),0); loss.backward()
    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()==0 for p in trainer.policy.human_encoder.parameters()))


class ReferenceWarmStartTests(unittest.TestCase):
  def test_zero_gain_preserves_existing_motion_human_outputs_and_learns_gate(self):
    from morphmimic.model.policy import Policy,PolicyCfg
    legacy=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,hist_len=1,human=True))
    policy=Policy(replace(legacy.cfg,ref_reconstruction_skip=True))
    policy.load_state_dict(legacy.state_dict(),strict=False)
    policy.set_reference_controls(torch.empty(1,0),torch.zeros(1,29),torch.ones(1,29))
    policy.set_reference_stats(torch.ones(670),torch.ones(670))
    hist=torch.randn(2,93);goal=torch.randn(2,670);human=torch.randn(2,2254)
    self.assertEqual(policy.decoder.reference_skip_gain.item(),0)
    torch.testing.assert_close(policy(hist,goal),legacy(hist,goal),atol=0,rtol=0)
    torch.testing.assert_close(policy.forward_human(hist,human),legacy.forward_human(hist,human),atol=0,rtol=0)
    policy(hist,goal).square().mean().backward()
    self.assertTrue(torch.isfinite(policy.decoder.reference_skip_gain.grad))
    self.assertNotEqual(policy.decoder.reference_skip_gain.grad.item(),0)
