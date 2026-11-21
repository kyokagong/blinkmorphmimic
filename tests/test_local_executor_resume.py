"""Exercise the real coordinator checkpoint/load with native-free Trainer fixtures.

Physics/replay are deliberately absent here; real native rewarm is a GPU qualification.
"""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn


def make_executor(tmp_path):
    from morphmimic.constants import LOCAL_DIM, PROPRIO_DIM, GOAL_DIM,HUMAN_FEAT_DIM,HUMAN_HEADING_DIM
    from morphmimic.model.policy import Policy,PolicyCfg
    from morphmimic.robots import get_robot
    from morphmimic.train.trainer import Trainer,TrainCfg
    from morphmimic.train.single_gpu import SingleGpuMultiRobotExecutor
    from morphmimic.train.normalizers import RunningNormalizer
    from morphmimic.train.optim import PolicyEMA
    from morphmimic.train.optimizations import ActorLRSchedule,WarmupGate
    from morphmimic.train.update_phases import share_parameters
    ex=SingleGpuMultiRobotExecutor.__new__(SingleGpuMultiRobotExecutor)
    ex.cfg=TrainCfg(executor='single_gpu_sequential',robots=('q1','bumi','g1'),subset_yamls=('a','b','c'),
        share='all',pool_norms=True,kp_interface=True,kp_student='return',latent_align_weight=0.,compile='none',
        policy_hidden=(4,),wm_hidden=(4,),critic_hidden=(4,),target_critic_alpha=.9,policy_ema=.9,
        log_dir=str(tmp_path),run_name='run',optimization_schema=0,async_checkpoint=False)
    ex.subset_hashes=dict.fromkeys(('q1','bumi','g1'),'a'*64)
    ex.device='cpu';ex.run_dir=tmp_path/'run';ex.trainers=[];ex.accumulators={};ex.writer=None;ex.stage_updates={}
    for index,robot in enumerate(ex.cfg.robots):
        t=Trainer.__new__(Trainer);t.cfg=replace(ex.cfg,robot=robot,subset_yaml=ex.cfg.subset_yamls[index],morphology=False,decoder_film=False)
        t.rank=0;t.world_size=1;t.multi=True;t.is_main=True;t.device='cpu';t._cuda=False;t._fused_adam=False
        t.policy=Policy(PolicyCfg(enc_hidden=(4,),dec_hidden=(4,),human=True))
        t.wm=nn.Linear(2,2);t.wm.register_buffer('residual_scale',torch.ones(LOCAL_DIM)*(index+1))
        t.critic=nn.Linear(2,1);t.critic_kp=nn.Linear(2,1)
        t.target_critic=copy.deepcopy(t.critic).requires_grad_(False)
        t.target_critic_kp=copy.deepcopy(t.critic_kp).requires_grad_(False)
        if index:
            for name in ('policy','wm','critic','critic_kp','target_critic','target_critic_kp'):share_parameters(getattr(t,name),getattr(ex.owner,name))
        for name,dim in [('obs_norm',PROPRIO_DIM),('goal_norm',GOAL_DIM),('human_norm',HUMAN_FEAT_DIM),('heading_norm',HUMAN_HEADING_DIM)]:
            setattr(t,name,getattr(ex.owner,name) if index else RunningNormalizer(dim))
        for name,params in [('wm_opt',t.wm.parameters()),('policy_opt',t.policy.motion_parameters()),('kp_opt',t.policy.human_parameters()),('critic_opt',t.critic.parameters()),('critic_kp_opt',t.critic_kp.parameters())]:
            setattr(t,name,getattr(ex.owner,name) if index else torch.optim.Adam(params,lr=.001,fused=False,capturable=False))
        t.policy_avg=PolicyEMA(t.policy,.9)
        if index:share_parameters(t.policy_avg.module,ex.owner.policy_avg.module)
        t._target_pairs=((t.target_critic,t.critic),(t.target_critic_kp,t.critic_kp))
        t.kp_loss_scales=None;t._probe_mtime=None;t._collect_acc=.25;t.it=0;t.resumed_from=None
        t._actor_lr=ex.owner._actor_lr if index else ActorLRSchedule(4,.75,3)
        t._warmup_gate=ex.owner._warmup_gate if index else WarmupGate(1,.2,ready_at=1)
        t._optimization_updates=dict(world_model=0,motion=0,human=0)
        t.command=SimpleNamespace(G_b=torch.tensor([float(index)]),past_softmax=torch.ones(1),probe_probabilities=None,spec=get_robot(robot))
        t.library=SimpleNamespace(device='cpu',clip_names=['clip'],usage={k:torch.tensor([index]) for k in ('collected','world_model','policy')},
            source_coverage_state=lambda:{},restore_source_coverage=lambda *a:None)
        from morphmimic.train.dataset_transition import human_coverage_state
        t._human_coverage=human_coverage_state(None,t.library,0)
        t.env=SimpleNamespace(physics_diagnostics=SimpleNamespace(report=lambda:{'overflow_bits':0}))
        ex.trainers.append(t)
    return ex


def step(ex):
    from morphmimic.train.update_phases import run_equal_phases,UpdatePhase
    def phases(t,index):
        for name,opt_name,params in [('wm','wm_opt',list(t.wm.parameters())),('policy','policy_opt',list(t.policy.motion_parameters())),
                ('critic','critic_opt',list(t.critic.parameters())),('human','kp_opt',list(t.policy.human_parameters())),
                ('critic_kp','critic_kp_opt',list(t.critic_kp.parameters()))]:
            opt=getattr(t,opt_name);opt.zero_grad()
            sum(p.square().mean()*(index+1) for p in params).backward()
            yield UpdatePhase(name,tuple(params),opt,opt.step)
        t.it+=1
        for k in t._optimization_updates:t._optimization_updates[k]+=1
    run_equal_phases([phases(t,i) for i,t in enumerate(ex.trainers)],on_step=ex._record_step)
    ex.owner._update_target();ex.owner.policy_avg.update_parameters(ex.owner.policy)
    for t in ex.trainers:t.policy_avg.copy_buffers(t.policy);t.policy_avg.updates=ex.owner.policy_avg.updates


def test_real_coordinator_restore_once_and_split_run_tensor_parity(tmp_path):
    from morphmimic.train.optimizations import cpu_snapshot
    from morphmimic.train.single_gpu import _same
    ex=make_executor(tmp_path);step(ex);ex.save_checkpoint()
    expected_random=torch.rand(5)
    step(ex)
    expected=cpu_snapshot({r.cfg.robot:r.checkpoint() for r in ex.trainers})
    counts={};patches=[]
    for name in ('wm_opt','policy_opt','kp_opt','critic_opt','critic_kp_opt'):
        opt=getattr(ex.owner,name);original=opt.load_state_dict
        def load(state,name=name,original=original):counts[name]=counts.get(name,0)+1;return original(state)
        patches.append(patch.object(opt,'load_state_dict',load));patches[-1].start()
    try:ex.load(ex.run_dir)
    finally:
        for p in patches:p.stop()
    assert set(counts.values())=={1} and len(counts)==5
    torch.testing.assert_close(torch.rand(5),expected_random)
    assert ex.it==1 and ex.stage_updates==dict.fromkeys(('wm','policy','critic','human','critic_kp'),1)
    assert [float(t.wm.residual_scale[0]) for t in ex.trainers]==[1.,2.,3.]
    assert [float(t.command.G_b[0]) for t in ex.trainers]==[0.,1.,2.]
    step(ex)
    for t in ex.trainers:
        actual=t.checkpoint();old=expected[t.cfg.robot]
        for name in ('wm','policy','critic','critic_kp','target_critic','target_critic_kp','ema_policy','optim','optimization_state','ema_updates','coverage','sampler'):
            assert _same(actual[name],old[name]),name


def test_all_robot_semantics_are_validated_before_mutating_owner(tmp_path):
    import pytest
    ex=make_executor(tmp_path)
    last=ex.trainers[-1];original=last.checkpoint
    def invalid():
        state=original();state['goal_version']=-1;return state
    with patch.object(last,'checkpoint',invalid):ex.save_checkpoint()
    step(ex)
    before=[p.detach().clone() for p in ex.owner.policy.parameters()]
    with pytest.raises(ValueError,match='goal'):ex.load(ex.run_dir)
    for a,b in zip(ex.owner.policy.parameters(),before):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_resume_refuses_another_run_root_before_reading_files(tmp_path):
    import pytest
    ex=make_executor(tmp_path)
    with pytest.raises(ValueError,match='root'):ex.load(tmp_path/'different-root')


def test_last_robot_invalid_probe_focus_rejects_before_any_state_change(tmp_path):
    import pytest
    from morphmimic.train.optimizations import cpu_snapshot
    from morphmimic.train.single_gpu import _same
    from morphmimic.train.probe_focus import normalize_probe_focus
    ex=make_executor(tmp_path)
    for t in ex.trainers:
        t.library.num_bins=t.command.G_b.numel()
        t.command.set_probe_focus=lambda probabilities,rho:normalize_probe_focus(probabilities,1,rho)
    step(ex)
    last=ex.trainers[-1];original=last.checkpoint
    def invalid():
        state=original();state['sampler']['probe_probabilities']=torch.tensor([-1.]);state['sampler']['probe_rho']=.1;return state
    with patch.object(last,'checkpoint',invalid):ex.save_checkpoint()
    step(ex)
    before=cpu_snapshot({t.cfg.robot:t.checkpoint() for t in ex.trainers})
    import random,numpy as np
    rng=torch.get_rng_state().clone();python_rng=random.getstate();numpy_rng=np.random.get_state()
    with pytest.raises(ValueError,match='probe'):ex.load(ex.run_dir)
    after=cpu_snapshot({t.cfg.robot:t.checkpoint() for t in ex.trainers})
    for robot in before:
        for name in ('policy','wm','optim','sampler','ema_policy','coverage','optimization_state','iteration'):
            assert _same(before[robot][name],after[robot][name]),(robot,name)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert random.getstate()==python_rng
    now=np.random.get_state();assert now[0]==numpy_rng[0] and np.array_equal(now[1],numpy_rng[1]) and now[2:]==numpy_rng[2:]


def test_valid_probe_focus_roundtrip_uses_actual_command_installer(tmp_path):
    from types import MethodType
    from morphmimic.env.commands import MultiMotionCommand
    ex=make_executor(tmp_path)
    for index,t in enumerate(ex.trainers):
        t.library.num_bins=1;t.command.library=t.library
        t.command.set_probe_focus=MethodType(MultiMotionCommand.set_probe_focus,t.command)
        t.command.set_probe_focus(torch.tensor([float(index+1)]),.1*(index+1))
    step(ex);ex.save_checkpoint();step(ex)
    for t in ex.trainers:t.command.set_probe_focus(torch.zeros(1),0.)
    ex.load(ex.run_dir)
    for index,t in enumerate(ex.trainers):
        torch.testing.assert_close(t.command.probe_probabilities,torch.ones(1))
        assert t.command.probe_rho==.1*(index+1)


def test_last_robot_controller_identity_rejects_before_owner_changes(tmp_path):
    import pytest
    from dataclasses import asdict
    from morphmimic.env.controller_recording import controller_from_config
    from morphmimic.train.optimizations import cpu_snapshot
    from morphmimic.train.single_gpu import _same
    ex=make_executor(tmp_path)
    ex.cfg=replace(ex.cfg,controller_mode='native_delay_v1')
    for t in ex.trainers:
        t.cfg=replace(t.cfg,controller_mode='native_delay_v1')
        # Native plant identity on the existing network-only coordinator fixture.
        t.wm.controller=controller_from_config(asdict(t.cfg))
        t.wm.cfg=SimpleNamespace(controller_mode='native_delay_v1')
    step(ex)
    last=ex.trainers[-1];original=last.checkpoint
    def invalid():
        state=original()
        state['controller_identity']['mapping']['scale'][0]+=.1
        return state
    with patch.object(last,'checkpoint',invalid):ex.save_checkpoint()
    step(ex)
    before=cpu_snapshot({t.cfg.robot:t.checkpoint() for t in ex.trainers})
    with pytest.raises(ValueError,match='controller identity'):ex.load(ex.run_dir)
    after=cpu_snapshot({t.cfg.robot:t.checkpoint() for t in ex.trainers})
    for robot in before:
        for name in ('policy','wm','optim','sampler','ema_policy','iteration'):
            assert _same(before[robot][name],after[robot][name]),(robot,name)


def test_last_robot_invalid_human_coverage_rejects_before_owner_changes(tmp_path):
    import pytest
    ex=make_executor(tmp_path)
    last=ex.trainers[-1]; original=last.checkpoint
    def invalid():
        state=original();state['human_coverage']['since_iteration']=-1;return state
    with patch.object(last,'checkpoint',invalid):ex.save_checkpoint()
    step(ex)
    before=[p.detach().clone() for p in ex.owner.policy.parameters()]
    with pytest.raises(ValueError,match='Human coverage'):ex.load(ex.run_dir)
    for a,b in zip(ex.owner.policy.parameters(),before):torch.testing.assert_close(a,b,rtol=0,atol=0)
