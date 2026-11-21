from types import SimpleNamespace

import pytest
import torch
from torch import nn


def fixture():
    from morphmimic.model.policy import Policy, PolicyCfg
    from morphmimic.train.normalizers import RunningNormalizer
    policy=Policy(PolicyCfg(enc_hidden=(8,8),dec_hidden=(8,8),human=True,ref_reconstruction_skip=True))
    return SimpleNamespace(cfg=SimpleNamespace(robot='q1',initial_checkpoint_iteration=10000,wm_joint_dynamics='semi_implicit'),
        policy=policy,wm=nn.Linear(2,2),critic=nn.Linear(2,1),critic_kp=nn.Linear(2,1),
        obs_norm=RunningNormalizer(93),goal_norm=RunningNormalizer(670),human_norm=RunningNormalizer(2160),
        heading_norm=RunningNormalizer(60),target_critic=None,target_critic_kp=None,it=0)


def test_joint_initialization_keeps_new_counters_and_old_inputs(tmp_path):
    from morphmimic.train.joint_initialization import initialize_joint
    from morphmimic.model.policy import Policy,PolicyCfg
    original=Policy(PolicyCfg(enc_hidden=(8,8),dec_hidden=(8,8),human=True))
    t=fixture();path=tmp_path/'q1/ckpt_10000.pt';path.parent.mkdir()
    ck=dict(iteration=10000,cfg=dict(robot='q1'),ema_policy=original.state_dict(),policy=original.state_dict(),
        wm=t.wm.state_dict(),critic=t.critic.state_dict(),critic_kp=t.critic_kp.state_dict(),
        **{key:getattr(t,key).state_dict() for key in ('obs_norm','goal_norm','human_norm','heading_norm')})
    torch.save(ck,path)
    initialize_joint(t,path)
    assert t.it==0 and t.initialized_from['iteration']==10000
    assert t.initialized_from['optimizer_state_restored'] is False
    for key,value in original.state_dict().items():torch.testing.assert_close(t.policy.state_dict()[key],value,atol=0,rtol=0)
    assert t.policy.decoder.reference_skip_gain.item()==0
    t.cfg.robot='g1'
    with pytest.raises(ValueError):initialize_joint(t,path)

