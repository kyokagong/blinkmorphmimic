import threading

import pytest
import torch

from morphmimic.train.optimizations import AsyncCheckpointWriter, ActorLRSchedule, WarmupGate


def test_adaptive_warmup_refuses_a_world_model_only_final_run():
    gate = WarmupGate(minimum=2, threshold=.2, maximum=4)
    assert not gate.observe(2, [.3])
    with pytest.raises(RuntimeError, match='warmup'):
        gate.observe(4, [.3])


def test_async_checkpoint_owns_an_immutable_snapshot(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    save = torch.save
    def slow_save(value, path):
        started.set()
        assert release.wait(5)
        save(value, path)
    monkeypatch.setattr(torch, 'save', slow_save)
    writer = AsyncCheckpointWriter()
    tensor = torch.tensor([1.])
    path = tmp_path / 'ckpt_1.pt'
    writer.submit(path, {'policy': {'weight': tensor}, 'iteration': 1, 'optim': {'step': tensor}})
    assert started.wait(5)
    tensor.fill_(9)
    release.set()
    writer.close()
    assert torch.load(path, weights_only=True)['policy']['weight'].item() == 1
    assert torch.load(tmp_path/'latest.pt', weights_only=True)['iteration'] == 1
    light = torch.load(tmp_path/'policy_1.pt', weights_only=True)
    assert light['inference_only'] is True and 'optim' not in light


def test_async_checkpoint_propagates_write_failure(tmp_path):
    blocker = tmp_path/'file'; blocker.write_text('original')
    writer = AsyncCheckpointWriter()
    writer.submit(blocker/'ckpt_1.pt', {'iteration': 1})
    with pytest.raises(FileExistsError):
        writer.close()
    assert blocker.read_text() == 'original'


def test_actor_lr_trigger_preserves_resume_and_ignores_nan():
    schedule = ActorLRSchedule(multiplier=4, threshold=.75, decay_steps=300)
    schedule.observe(100, float('nan'))
    assert schedule.factor(100) == 4
    schedule.observe(200, .8)
    assert schedule.factor(200) == 4
    assert schedule.factor(350) == 2.5
    restored = ActorLRSchedule(**schedule.state_dict())
    assert restored.factor(500) == 1
    restored.observe(550, .9)
    assert restored.factor(550) == 1


def test_warmup_waits_for_worst_robot_and_minimum_samples():
    gate = WarmupGate(minimum=100, threshold=.2)
    assert not gate.observe(99, [.01, .02, .03])
    assert not gate.observe(100, [.01, .3, .03])
    assert not gate.observe(150, [.01, float('nan'), .03])
    assert gate.observe(200, [.01, .2, .03])
    assert gate.ready_at == 200
    assert gate.observe(250, [float('nan')])


def test_replay_training_and_validation_worlds_do_not_overlap():
    from morphmimic.train.replay import ReplayBuffer, ReplayCfg
    replay=ReplayBuffer(ReplayCfg(num_envs=2,capacity=8))
    replay.state[0].fill_(1);replay.state[1].fill_(2)
    replay.size.fill_(8);replay.ep_step[:]=torch.arange(8);replay.valid.fill_(True)
    assert replay.sample_windows(32,2,env_range=(0,1)).state.eq(1).all()
    assert replay.sample_windows(32,2,env_range=(1,2)).state.eq(2).all()
    with pytest.raises(ValueError):replay.sample_windows(1,2,env_range=(2,2))

