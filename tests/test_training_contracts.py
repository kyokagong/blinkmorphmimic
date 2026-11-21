import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from torch import nn

path = Path(__file__).parents[1] / 'src/morphmimic/train/optim.py'

def helpers():
    assert path.exists(), 'optimization helpers are not implemented'
    spec = importlib.util.spec_from_file_location('training_optim', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ema_averages_parameters_but_copies_robot_buffers():
    m = nn.Linear(2, 2)
    m.register_buffer('morph', torch.tensor([1., 0.]))
    with torch.no_grad():
        m.weight.fill_(2); m.bias.fill_(4)
    ema = helpers().PolicyEMA(m, .5)
    with torch.no_grad():
        m.weight.fill_(6); m.bias.fill_(8); m.morph.copy_(torch.tensor([-1., 1.]))
    ema.update(m)
    torch.testing.assert_close(ema.module.weight, torch.full((2,2), 4.))
    torch.testing.assert_close(ema.module.bias, torch.full((2,), 6.))
    torch.testing.assert_close(ema.module.morph, torch.tensor([-1.,1.]))
    assert ema.updates == 1
    assert all(not p.requires_grad for p in ema.module.parameters())


def test_muon_partitions_only_internal_weights_and_roundtrips_state():
    if not hasattr(torch.optim, 'Muon'):
        pytest.skip('Muon requires the pinned training runtime')
    m = nn.Sequential(nn.Linear(3,8), nn.ELU(), nn.Linear(8,8), nn.ELU(), nn.Linear(8,2))
    h = helpers()
    opt = h.make_optimizer(m.parameters(), [m], kind='muon', lr=.001, fused=False)
    assert {id(p) for g in opt.muon.param_groups for p in g['params']} == {id(m[2].weight)}
    m(torch.ones(4,3)).square().mean().backward(); opt.step()
    restored = h.make_optimizer(m.parameters(), [m], kind='muon', lr=.001, fused=False)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    opt.zero_grad(); m(torch.ones(4,3)).sum().backward()
    expected = {n:p.detach().clone() for n,p in m.named_parameters()}
    opt.step(); after = {n:p.detach().clone() for n,p in m.named_parameters()}
    with torch.no_grad():
        for n,p in m.named_parameters(): p.copy_(expected[n])
    restored.step()
    for n,p in m.named_parameters(): torch.testing.assert_close(p, after[n], rtol=0, atol=0)


def test_ema_rejects_invalid_decay():
    with pytest.raises(ValueError): helpers().PolicyEMA(nn.Linear(2,2), 1.)


def test_motion_coverage_counts_actual_valid_frames_by_clip():
    from morphmimic.data.motion_library import MotionLibrary
    lib = MotionLibrary.__new__(MotionLibrary)
    lib.frame_clip = torch.tensor([0,0,1,1,2])
    lib.usage = {name:torch.zeros(3,dtype=torch.long) for name in ('collected','world_model','policy')}
    lib.source_usage = {name: torch.zeros_like(value) for name, value in lib.usage.items()}
    lib.clip_start = torch.tensor([0, 2, 4]); lib.clip_len = torch.tensor([2, 2, 1])
    lib.source_start = torch.tensor([0, 0, 0]); lib.source_stop = torch.tensor([2, 2, 1])
    assert hasattr(lib, 'record_usage'), 'actual training coverage is not recorded'
    lib.record_usage('world_model', torch.tensor([[0,2],[3,4]]), torch.tensor([[1,1],[0,1]],dtype=torch.bool))
    torch.testing.assert_close(lib.usage['world_model'],torch.tensor([1,1,1]))
    torch.testing.assert_close(lib.usage['policy'],torch.tensor([0,0,0]))


def test_trainer_installs_only_its_robot_training_probe(tmp_path):
    import hashlib, json
    from types import SimpleNamespace
    from morphmimic.train.trainer import Trainer
    from morphmimic.env.commands import MultiMotionCommand
    subset = tmp_path/'train.yaml'; subset.write_text('root: .\nclips: [clip.npz]\n')
    lib = SimpleNamespace(fps=50,clip_names=['clip'],num_bins=2,bin_start=torch.tensor([0,50]),
        bin_len=torch.tensor([50,50]),bin_clip=torch.tensor([0,0]),clip_start=torch.tensor([0]),clip_len=torch.tensor([101]))
    command = MultiMotionCommand.__new__(MultiMotionCommand);command.library=lib;command.G_b=torch.zeros(2)
    trainer = Trainer.__new__(Trainer);trainer.library=lib;trainer.command=command
    refreshed=[]
    trainer.collector=SimpleNamespace(refresh_reset_sampler=lambda:refreshed.append(True))
    trainer.cfg=SimpleNamespace(probe_focus_dir=str(tmp_path),probe_reset_share=.15,robot='q1',subset_yaml=str(subset))
    trainer._probe_mtime=None
    assert hasattr(trainer,'_reload_probe_focus'), 'training cannot install the validated probe focus'
    trainer._reload_probe_focus()
    doc=dict(schema_version=1,robot='q1',interface='motion',fps=50,checkpoint_sha256='a'*64,
        training_subset_sha256=hashlib.sha256(subset.read_bytes()).hexdigest(),
        failures=[dict(robot='q1',source_id='seed/clip',clip='clip',fail_frame=100)])
    (tmp_path/'q1.json').write_text(json.dumps(doc))
    trainer._reload_probe_focus()
    torch.testing.assert_close(command.probe_probabilities,torch.tensor([.5,.5]))
    assert command.probe_rho==.15
    assert refreshed==[True]
    trainer._reload_probe_focus()
    assert refreshed==[True]
    doc['robot']='g1';(tmp_path/'q1.json').write_text(json.dumps(doc))
    trainer._probe_mtime=None
    with pytest.raises(ValueError,match='robot'):trainer._reload_probe_focus()
    assert refreshed==[True]


def test_resume_reloads_cached_current_probe_after_saved_sampler(tmp_path):
    import hashlib, json
    from dataclasses import asdict
    from types import SimpleNamespace
    from morphmimic.constants import GOAL_VERSION, LOCAL_DIM
    from morphmimic.env.commands import MultiMotionCommand
    from morphmimic.model.policy import Policy, PolicyCfg
    from morphmimic.train.trainer import Trainer, TrainCfg, rng_state

    subset = tmp_path / 'train.yaml'
    subset.write_text('root: .\nclips: [clip.npz]\n')
    lib = SimpleNamespace(fps=50, device='cpu', clip_names=['clip'], num_bins=2,
        bin_start=torch.tensor([0, 50]), bin_len=torch.tensor([50, 50]),
        bin_clip=torch.tensor([0, 0]), clip_start=torch.tensor([0]), clip_len=torch.tensor([101]))
    command = MultiMotionCommand.__new__(MultiMotionCommand)
    command.library, command.G_b, command.past_softmax = lib, torch.zeros(2), torch.ones(2)
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainCfg(robot='q1', subset_yaml=str(subset), policy_hidden=(4,),
        policy_ema=0, probe_focus_dir=str(tmp_path))
    from morphmimic.data.motion_library import MotionLibrary
    lib.source_usage = {stage: torch.zeros(1, dtype=torch.long) for stage in ('collected', 'world_model', 'policy')}
    lib.restore_source_coverage = lambda state, iteration: MotionLibrary.restore_source_coverage(lib, state, iteration)
    trainer.library, trainer.command, trainer._probe_mtime = lib, command, None
    trainer.multi, trainer.world_size, trainer.rank, trainer._fused_adam = False, 1, 0, False
    trainer.policy = Policy(PolicyCfg(enc_hidden=(4,), dec_hidden=(4,)))
    trainer.wm, trainer.critic = nn.Linear(2, 2), nn.Linear(2, 1)
    trainer.wm.register_buffer('residual_scale', torch.ones(LOCAL_DIM))
    trainer.obs_norm, trainer.goal_norm = nn.Linear(2, 2), nn.Linear(2, 2)
    trainer.policy_avg = trainer.target_critic = None
    trainer.wm_opt = torch.optim.Adam(trainer.wm.parameters())
    trainer.policy_opt = torch.optim.Adam(trainer.policy.parameters())
    trainer.critic_opt = torch.optim.Adam(trainer.critic.parameters())
    doc = dict(schema_version=1, robot='q1', interface='motion', fps=50,
        checkpoint_sha256='a' * 64, training_subset_sha256=hashlib.sha256(subset.read_bytes()).hexdigest(),
        failures=[dict(robot='q1', source_id='seed/clip', clip='clip', fail_frame=100)])
    (tmp_path / 'q1.json').write_text(json.dumps(doc))
    trainer._reload_probe_focus()
    torch.testing.assert_close(command.probe_probabilities, torch.tensor([.5, .5]))
    parent=dict(path='/original/parent/q1/ckpt_10000.pt',sha256='b'*64,iteration=10000)
    trainer.initialized_from=dict(parent,sha256='c'*64)
    checkpoint = dict(cfg=asdict(trainer.cfg), goal_version=GOAL_VERSION, iteration=6,initialized_from=parent,
        sampler=dict(G_b=torch.ones(2), past_softmax=torch.ones(2),
            probe_probabilities=torch.tensor([1., 0.]), probe_rho=.3), rng=rng_state(),
        optim={name: getattr(trainer, name + '_opt').state_dict() for name in ('wm', 'policy', 'critic')},
        **{name: getattr(trainer, name).state_dict() for name in ('policy', 'wm', 'critic', 'obs_norm', 'goal_norm')})
    path = tmp_path / 'ckpt_6.pt'
    torch.save(checkpoint, path)
    trainer.load(path)
    assert trainer.initialized_from==parent
    torch.testing.assert_close(command.G_b, torch.ones(2))
    torch.testing.assert_close(command.probe_probabilities, torch.tensor([.5, .5]))
    assert command.probe_rho == .15
