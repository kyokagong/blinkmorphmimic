"""One motion actor with explicit Q1/Bumi/G1 conditioning per batch row."""
from __future__ import annotations

import torch

from deploy.export_onnx import DeployPolicy, DeployHumanOnlyPolicy, _build_policy, _load, has_human
from morphmimic.model.human import fixed_human_masks, human_features_from_goal, heading_from_goal
from morphmimic.eval.policy_loader import PolicyMeta
from morphmimic.train.normalizers import RunningNormalizer
from preview.release import verify_human_identity

ROBOTS = ('q1', 'bumi', 'g1')
BODY_BUFFERS = {'decoder.morph_actor', 'decoder.joint_mask'}


class SharedPolicy:
    def __init__(self, checkpoints, device, *, interface='motion', human_points='five',paired_manifest_sha256=None,training_manifest_sha256=None,control_graph=False):
        if interface not in ('motion','human'):
            raise ValueError('Unknown shared policy interface')
        self.interface = interface
        if human_points not in ('five','full'):raise ValueError('Human points must be five or full')
        self.human_points = human_points
        for robot in ROBOTS:
            checkpoints[robot].verify()
        saved = _load([checkpoints[r].path for r in ROBOTS])
        first = saved['q1']
        state = first.get('ema_policy', first['policy'])
        self.entries = {}
        for robot in ROBOTS:
            ck = saved[robot]
            if any(ck['cfg'].get(k) != first['cfg'].get(k) for k in
                   ('policy_hidden', 'act', 'hist_len', 'latent')) or ck['morph']['film'] != first['morph']['film']:
                raise ValueError(f'{robot}: shared policy configuration differs')
            if (ck['_path'] != str(checkpoints[robot].path) or ck['cfg']['robot'] != robot
                    or ck['iteration'] != checkpoints[robot].iteration):
                raise ValueError('Shared checkpoint robot/iteration differs from its pinned identity')
            other = ck.get('ema_policy', ck['policy'])
            if other.keys() != state.keys() or any(
                not torch.equal(v, other[k]) for k, v in state.items() if k not in BODY_BUFFERS
            ):
                raise ValueError(f'{robot}: shared policy weights differ')
            if interface == 'human' and not has_human(ck):
                raise ValueError('Checkpoint has no trained human interface')
            if interface=='human' or training_manifest_sha256 is not None:
                verify_human_identity(ck,robot=robot,update=checkpoints[robot].iteration,data_sha256=paired_manifest_sha256,training_sha256=training_manifest_sha256)
            for group in (('obs_norm','goal_norm','human_norm','heading_norm') if interface == 'human' else ('obs_norm', 'goal_norm')):
                if ck[group].keys() != first[group].keys() or any(
                    not torch.equal(v, ck[group][k]) for k, v in first[group].items()
                ):
                    raise ValueError(f'{robot}: shared normalization differs')
            morph, mask = other['decoder.morph_actor'], other['decoder.joint_mask']
            if morph.shape != (154,) or mask.shape != (29,) or not torch.equal(morph[-29:], mask):
                raise ValueError(f'{robot}: incompatible morphology/joint mask')
            if not bool(torch.isfinite(morph).all()) or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError(f'{robot}: invalid morphology/joint mask')
            cfg = ck['cfg']
            if cfg.get('policy_goal', 'motion') != 'motion':
                raise ValueError('Shared preview requires a motion policy')
            self.entries[robot] = dict(cfg=dict(cfg), training_physics=ck.get('physics'),
                meta=PolicyMeta(path=str(checkpoints[robot].path), iteration=ck['iteration'],
                    hist_len=int(cfg['hist_len']), train_cfg=dict(cfg), robot=robot,
                    interface=interface, mode=f'human-{human_points}-h1' if interface=='human' else 'motion'))
            checkpoints[robot].verify()
        if len({e['cfg'].get('goal_anchor', 'reference') for e in self.entries.values()}) != 1:
            raise ValueError('Shared policies differ in goal anchor')
        self.device = device
        model_type = DeployHumanOnlyPolicy if interface=='human' else DeployPolicy
        self.model = model_type(_build_policy(first), RunningNormalizer(93).eps).to(device).eval()
        if control_graph:
            from preview.control_graph import ActorGraph
            self.infer = ActorGraph(self.model, device)
        else:
            self.infer = self.model
        groups = ('obs_norm','human_norm','heading_norm') if interface=='human' else ('obs_norm','goal_norm')
        self.stats = tuple(first[group][key].to(device) for group in groups for key in ('mean','var'))
        self.morph = torch.stack([saved[r].get('ema_policy', saved[r]['policy'])['decoder.morph_actor']
                                  for r in ROBOTS]).to(device)
        self.calls = 0

    @torch.no_grad()
    def __call__(self, sessions):
        history = torch.cat([sessions[r].history.flat().to(self.device) for r in ROBOTS])
        if self.interface=='human':
            goal = torch.cat([sessions[r].goals['goal_kp'].to(self.device) for r in ROBOTS])
            points,frames,heading = fixed_human_masks(3,self.human_points,1,history.device)
            valid = torch.cat([sessions[r].goals['human_joint_valid'].to(self.device) for r in ROBOTS])
            five,_,_ = fixed_human_masks(3,'five',1,history.device)
            if not bool((five & valid).eq(five).all()):
                raise ValueError('Human preview requires all five tracked joints')
            points &= valid
            action = self.infer(history,human_features_from_goal(goal,points),points,frames,
                heading_from_goal(goal,points),heading,*self.stats,self.morph)
        else:
            goal = torch.cat([sessions[r].goals['goal'].to(self.device) for r in ROBOTS])
            action = self.infer(history, goal, *self.stats, self.morph)
        if not bool(torch.isfinite(action).all()):
            raise ValueError('Policy produced a nonfinite shared action; simulation stopped')
        self.calls += 1
        return {r: action[i:i + 1].to(sessions[r].env.device) for i, r in enumerate(ROBOTS)}
