from dataclasses import asdict

import pytest
import torch


@pytest.mark.parametrize('field,value', [('wm_joint_dynamics','semi_implicit'),('wm_state_mode','compact_fk'),('ref_reconstruction_skip',True)])
def test_resume_rejects_changed_model_semantics_before_loading_weights(tmp_path,field,value):
    from morphmimic.constants import GOAL_VERSION,KP_GOAL_VERSION
    from morphmimic.train.trainer import Trainer,TrainCfg
    source=asdict(TrainCfg(robot='q1'))
    source[field]=value
    checkpoint=dict(cfg=source,goal_version=GOAL_VERSION,kp_goal_version=KP_GOAL_VERSION,
        imu_body='pelvis',policy={},iteration=1)
    path=tmp_path/'ckpt_1.pt';torch.save(checkpoint,path)
    trainer=Trainer.__new__(Trainer);trainer.cfg=TrainCfg(robot='q1');trainer.multi=False
    with pytest.raises(ValueError,match=field):trainer.load(path)


def test_checkpoint_zero_resume_keeps_calibration_and_existing_baseline(tmp_path):
    from types import SimpleNamespace
    from morphmimic.constants import LOCAL_DIM
    from morphmimic.train.trainer import Trainer, TrainCfg
    checkpoint = tmp_path/'ckpt_0.pt'
    checkpoint.write_bytes(b'immutable initialized checkpoint')
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainCfg(iterations=0, optimization_schema=1, residual_scale='step')
    trainer.it=0; trainer.multi=False; trainer.world_size=1; trainer.rank=0
    trainer.is_main=False; trainer._writes=False; trainer.device='cpu'
    trainer.run_dir=tmp_path; trainer.resumed_from=checkpoint
    trainer.collector=SimpleNamespace(reset_all=lambda: None, step=lambda n: {})
    trainer.wm=SimpleNamespace(residual_scale=torch.full((LOCAL_DIM,), .05))
    trainer._close_loggers=lambda: None
    def unexpected():
        raise AssertionError('Resume must retain its initialized calibration and baseline')
    trainer.estimate_residual_scale=unexpected
    trainer.save_checkpoint=unexpected
    trainer.train()
    assert checkpoint.read_bytes()==b'immutable initialized checkpoint'


def test_shared_rank_divergence_prevents_checkpoint_publication(tmp_path):
    from morphmimic.train.trainer import Trainer
    trainer=Trainer.__new__(Trainer)
    trainer.run_dir=tmp_path; trainer.it=2; trainer.world_size=3; trainer.multi=True
    trainer._check_ranks_identical=lambda: False
    with pytest.raises(RuntimeError, match='shared'):
        trainer.save_checkpoint()
    assert not list(tmp_path.iterdir())

