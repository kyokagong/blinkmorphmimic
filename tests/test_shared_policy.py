"""Shared inference must preserve each robot's conditioning and pinned identity."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from morphmimic.constants import GOAL_VERSION
from morphmimic.model.policy import Policy, PolicyCfg
from morphmimic.eval.policy_loader import load_policy
from morphmimic.train.normalizers import RunningNormalizer
from preview.server import FixedCheckpoint

ROBOTS = ('q1', 'bumi', 'g1')

class SharedPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        policy = Policy(PolicyCfg(enc_hidden=(16, 8), dec_hidden=(16, 8), latent=4, hist_len=2, morph_dim=154, film=True))
        self.checkpoints = {}
        for i, robot in enumerate(ROBOTS):
            morph = torch.zeros(154)
            morph[i] = 1
            mask = torch.zeros(29)
            mask[:(22, 21, 29)[i]] = 1
            morph[-29:] = mask
            mask = morph[-29:]
            policy.set_morphology(morph)
            policy.set_joint_mask(mask)
            cfg = dict(robot=robot, share='all', policy_hidden=(16, 8), latent=4, hist_len=2, act='elu', goal_anchor='reference')
            folder = Path(self.tmp.name)/robot;folder.mkdir()
            path = folder/'ckpt_20.pt'
            torch.save(dict(cfg=cfg, iteration=20, goal_version=GOAL_VERSION, policy=policy.state_dict(), ema_policy=policy.state_dict(), morph=dict(actor=morph, film=True), obs_norm=RunningNormalizer(93).state_dict(), goal_norm=RunningNormalizer(670).state_dict()), path)
            self.checkpoints[robot] = FixedCheckpoint.pin(path, robot)
        self.sessions = {}
        for robot in ROBOTS:
            h, g = torch.randn(1, 186), torch.randn(1, 670)
            self.sessions[robot] = SimpleNamespace(history=SimpleNamespace(flat=lambda h=h: h), goals={'goal':g}, env=SimpleNamespace(device='cpu'))

    def make(self):
        from preview.shared_policy import SharedPolicy
        return SharedPolicy(self.checkpoints, 'cpu')

    def test_batched_actions_match_original_loader_for_each_body(self):
        shared = self.make()
        actions = shared(self.sessions)
        self.assertEqual(set(actions), set(ROBOTS))
        for robot in ROBOTS:
            fn,_ = load_policy(self.checkpoints[robot].path, 'cpu')
            child = self.sessions[robot]
            expected = fn(child.history.flat(), child.goals)
            torch.testing.assert_close(actions[robot], expected, atol=1e-5, rtol=1e-5)
            self.assertEqual(tuple(actions[robot].shape), (1,29))
        self.assertEqual(shared.calls, 1)
        self.assertEqual(shared.entries['bumi']['meta'].robot, 'bumi')

    def test_human_batch_matches_official_five_point_loader(self):
        from morphmimic.constants import KP_GOAL_VERSION
        from morphmimic.model.human import pack_human_goal
        from morphmimic.train.human_adapt import PARENT_SHA256
        from preview.shared_policy import SharedPolicy
        human=Policy(PolicyCfg(enc_hidden=(16,8),dec_hidden=(16,8),latent=4,hist_len=2,morph_dim=154,film=True,human=True))
        hs=human.human_encoder.state_dict()
        for robot in ROBOTS:
            ck=torch.load(self.checkpoints[robot].path,weights_only=False)
            for group in ('policy','ema_policy'):
                ck[group].update({'human_encoder.'+k:v for k,v in hs.items()})
            ck.update(kp_goal_version=KP_GOAL_VERSION,human_norm=RunningNormalizer(2160).state_dict(),heading_norm=RunningNormalizer(60).state_dict())
            ck['cfg'].update(kp_interface=True,kp_fixed_horizon=1)
            ck['human_adaptation']=dict(parent=dict(iteration=30000,sha256=PARENT_SHA256[robot]),human_updates=20,paired_manifest_sha256='a'*64)
            torch.save(ck,self.checkpoints[robot].path)
            self.checkpoints[robot]=FixedCheckpoint.pin(self.checkpoints[robot].path,robot)
            pos=torch.randn(1,10,24,3);quat=torch.zeros(1,10,24,4);quat[...,0]=1
            self.sessions[robot].goals.update(goal_kp=pack_human_goal(pos,quat,torch.tensor([[1.,0,0,0]])),human_joint_valid=torch.ones(1,24,dtype=torch.bool))
        for points in ('five','full'):
            self.sessions['g1'].goals['human_joint_valid'][:,22:]=False
            shared=SharedPolicy(self.checkpoints,'cpu',interface='human',human_points=points,paired_manifest_sha256='a'*64)
            actions=shared(self.sessions)
            for robot in ROBOTS:
                fn,_=load_policy(self.checkpoints[robot].path,'cpu',interface='human',human_points=points,kp_horizon=1)
                child=self.sessions[robot]
                torch.testing.assert_close(actions[robot],fn(child.history.flat(),child.goals),atol=1e-5,rtol=1e-5)
            self.assertEqual(shared.calls,1)

    def test_changed_network_is_refused_instead_of_silently_using_q1(self):
        pinned = self.checkpoints['bumi']
        ck = torch.load(pinned.path, weights_only=False)
        ck['ema_policy']['encoder.net.0.weight'][0,0] += .1
        torch.save(ck,pinned.path)
        self.checkpoints['bumi'] = FixedCheckpoint.pin(pinned.path,'bumi')
        with self.assertRaisesRegex(ValueError, 'shared.*policy|policy.*differ'):
            self.make()

    def test_different_normalization_is_refused(self):
        pinned = self.checkpoints['g1']
        ck = torch.load(pinned.path, weights_only=False)
        ck['obs_norm']['mean'][0] += 1
        torch.save(ck,pinned.path)
        self.checkpoints['g1'] = FixedCheckpoint.pin(pinned.path,'g1')
        with self.assertRaisesRegex(ValueError, 'normaliz'):
            self.make()

    def test_wrong_robot_label_is_refused(self):
        pinned = self.checkpoints['bumi']
        ck = torch.load(pinned.path,weights_only=False);ck['cfg']['robot']='g1'
        torch.save(ck,pinned.path)
        self.checkpoints['bumi'] = FixedCheckpoint.pin(pinned.path,'bumi')
        with self.assertRaises(ValueError):
            self.make()

    def test_nonfinite_input_is_not_sent_to_physics(self):
        shared=self.make()
        self.sessions['g1'].goals['goal'].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            shared(self.sessions)

    def test_same_weights_with_different_activation_are_refused(self):
        pinned=self.checkpoints['bumi']
        ck=torch.load(pinned.path,weights_only=False);ck['cfg']['act']='relu'
        torch.save(ck,pinned.path)
        self.checkpoints['bumi']=FixedCheckpoint.pin(pinned.path,'bumi')
        with self.assertRaisesRegex(ValueError,'config'):
            self.make()

    def test_swapped_robot_files_cannot_relabel_pinned_provenance(self):
        q1,g1=self.checkpoints['q1'],self.checkpoints['g1']
        a,b=q1.path.read_bytes(),g1.path.read_bytes()
        q1.path.write_bytes(b);g1.path.write_bytes(a)
        self.checkpoints['q1']=FixedCheckpoint.pin(q1.path,'q1')
        self.checkpoints['g1']=FixedCheckpoint.pin(g1.path,'g1')
        with self.assertRaisesRegex(ValueError,'identity|path|robot'):
            self.make()
