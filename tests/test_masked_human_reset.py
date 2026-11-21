from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('mask', [[False]*4, [True,False,True,False], [True]*4])
def test_human_reset_mask_preserves_other_worlds_and_motion_only_fallback(mask):
    from morphmimic.train.trainer import Trainer, TrainCfg
    trainer=Trainer.__new__(Trainer)
    trainer.cfg=TrainCfg(num_envs=4,kp_interface=True,kp_masking='five',kp_fixed_horizon=1)
    trainer.device='cpu'
    trainer.command=SimpleNamespace(global_frame=torch.arange(4))
    valid=torch.ones(4,24,dtype=torch.bool);valid[2]=False
    trainer.library=SimpleNamespace(human_valid_at=lambda frames:valid[frames])
    trainer.kp_env=torch.ones(4,dtype=torch.bool)
    trainer.kp_point_mask=torch.ones(4,24,dtype=torch.bool)
    trainer.kp_frame_mask=torch.ones(4,10,dtype=torch.bool)
    trainer.kp_heading_mask=torch.ones(4,dtype=torch.bool)
    trainer.kp_drift=torch.full((4,15),9.)
    trainer.kp_drift_step=torch.full((4,),7.)
    fields=('kp_env','kp_point_mask','kp_frame_mask','kp_heading_mask','kp_drift','kp_drift_step')
    before={key:getattr(trainer,key).clone() for key in fields}
    selected=torch.tensor(mask)
    trainer._on_reset_masked(selected)
    for key in fields:
        torch.testing.assert_close(getattr(trainer,key)[~selected],before[key][~selected])
    assert (trainer.kp_drift_step[selected]==0).all()
    if selected[0]:
        assert trainer.kp_env[0]
        assert trainer.kp_point_mask[0].nonzero().flatten().tolist()==[0,7,8,20,21]
        assert trainer.kp_frame_mask[0].tolist()==[True]+[False]*9
    if selected[2]:
        assert not trainer.kp_env[2] and not trainer.kp_point_mask[2].any()
