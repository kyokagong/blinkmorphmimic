"""CPU contracts around the real executor orchestration and existing critic loops."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from morphmimic.train.single_gpu import SingleGpuMultiRobotExecutor,validate_config
from morphmimic.train.normalizers import RunningNormalizer
from morphmimic.train.update_phases import run_equal_phases,share_parameters


def test_all_collectors_see_merged_norms_before_any_step():
    ex=SingleGpuMultiRobotExecutor.__new__(SingleGpuMultiRobotExecutor)
    shared=RunningNormalizer(1);seen=[];trainers=[]
    for number,n in enumerate((2,3,7)):
        values=torch.full((n,1),float(number*3))
        t=SimpleNamespace(obs_norm=shared)
        t.collection_normalizer_inputs=lambda hist,obs,v=values:{'obs_norm':v}
        def step(count,t=t,n=n,number=number):
            assert t._collection_norms_preupdated
            seen.append((number,int(shared.count),float(shared.mean)))
            return dict(steps=n,episodes_finished=0,failures=0,mean_reward=2.,mean_norm_return=float('nan'),mean_episode_length=float('nan'))
        t.collector=SimpleNamespace(history=SimpleNamespace(flat=lambda:None),obs={},step=step)
        trainers.append(t)
    ex.trainers=trainers;stats=ex.collect(2)
    expected=(3*3+7*6)/12
    assert [n for _,n,_ in seen]==[12]*3+[24]*3
    assert all(abs(mean-expected)<1e-6 for _,_,mean in seen)
    assert [x['steps'] for x in stats]==[4,6,14]
    assert all(not t._collection_norms_preupdated for t in trainers)


def test_actual_critic_minibatches_share_one_step_and_match_joint_gradient():
    from morphmimic.train.imagination import fit_critic_phases,ImaginationCfg
    torch.manual_seed(8)
    model=nn.Linear(2,1);reference=copy.deepcopy(model)
    others=[model,copy.deepcopy(model),copy.deepcopy(model)]
    for m in others[1:]:share_parameters(m,model)
    opt=torch.optim.Adam(model.parameters(),lr=.01);refopt=torch.optim.Adam(reference.parameters(),lr=.01)
    batches=[torch.randn(8,2) for _ in range(3)];targets=[torch.randn(8) for _ in range(3)]
    cfg=ImaginationCfg(critic_epochs=2,critic_minibatches=2)
    indices=[[] for _ in range(3)]
    def loss_fn(robot):
        def loss(ro,target,idx):
            indices[robot].append(idx.clone())
            return (others[robot](batches[robot][idx]).flatten()-target[idx]).square().mean()
        return loss
    phases=[fit_critic_phases(m,opt,None,cfg,targets=y,loss_fn=loss_fn(i),max_grad_norm=1e8) for i,(m,y) in enumerate(zip(others,targets))]
    counts=[];run_equal_phases(phases,on_step=counts.append)
    assert counts==['critic']*4
    for k in range(4):
        refopt.zero_grad()
        sum((reference(x[indices[i][k]]).flatten()-targets[i][indices[i][k]]).square().mean()/3 for i,x in enumerate(batches)).backward();refopt.step()
    for a,b in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)
        assert opt.state[a]['step']==4
        torch.testing.assert_close(opt.state[a]['exp_avg'],refopt.state[b]['exp_avg'])


def test_cli_config_contract_rejects_multi_device_and_unsupported_semantics():
    from morphmimic.train.trainer import TrainCfg
    cfg=TrainCfg(executor='single_gpu_sequential',robots=('q1','bumi','g1'),subset_yamls=('a','b','c'),
                 share='all',pool_norms=True,kp_interface=True,kp_student='return',latent_align_weight=0.,compile='none')
    validate_config(cfg)
    for change in [dict(gpu_ids=(0,1)),dict(share='wm'),dict(pool_norms=False),dict(latent_align_weight=None),
                   dict(compile='reduce-overhead'),dict(initial_checkpoint_root='/old')]:
        with pytest.raises(ValueError):validate_config(replace(cfg,**change))
    assert TrainCfg().executor=='distributed'


def test_local_logging_gate_uses_worst_robot_and_weighted_return():
    from morphmimic.train.optimizations import WarmupGate,ActorLRSchedule
    from morphmimic.train.trainer import _Window
    ex=SingleGpuMultiRobotExecutor.__new__(SingleGpuMultiRobotExecutor)
    ex.cfg=SimpleNamespace(adaptive_warmup=True);ex.trainers=[]
    gate=WarmupGate(1,.2,maximum=10);schedule=ActorLRSchedule(4.,.75,3)
    for error in (.1,.15,.3):
        t=SimpleNamespace(it=2,_warmup_gate=gate,_actor_lr=schedule,cfg=SimpleNamespace(num_envs=16,wm_validation_envs=2,window_k=8))
        t.replay=SimpleNamespace(sample_windows=lambda *a,**k:None);t.library=None
        t.wm=SimpleNamespace(fit_loss=lambda *a,error=error:(None,{'pred_err_8step':torch.tensor(error)}))
        t._log=lambda window,observe_schedules:None
        ex.trainers.append(t)
    windows=[_Window() for _ in range(3)]
    for w,ret,n in zip(windows,(.9,.1,.7),(1,10,2)):
        w.policy_iterations=1;w.mean('collect/mean_norm_return',ret,n)
    ex._log(windows)
    assert gate.ready_at is None and schedule.triggered_at is None
    ex.trainers[-1].wm.fit_loss=lambda *a:(None,{'pred_err_8step':torch.tensor(.19)})
    ex._log(windows);assert gate.ready_at==2


def test_cli_uses_local_runner_and_refuses_multiple_gpus_before_selection(monkeypatch):
    from morphmimic.scripts import train
    called=[]
    monkeypatch.setattr(train,'select_gpus',lambda *a:called.append(('gpu',a)))
    monkeypatch.setattr(train,'run',lambda cfg,*a,**kw:called.append(('run',cfg.executor)))
    monkeypatch.setattr(sys,'argv',['train','--executor','single_gpu_sequential','--gpu','1'])
    train.main()
    assert called[-1]==('run','single_gpu_sequential')
    called.clear()
    monkeypatch.setattr(sys,'argv',['train','--executor','single_gpu_sequential','--gpu-ids','0','2'])
    with pytest.raises(ValueError,match='requires one'):train.main()
    assert not called


def test_frozen_subset_hash_is_required_and_checked_before_native_allocation(tmp_path,monkeypatch):
    import hashlib
    from morphmimic.train import trainer
    cfg=trainer.TrainCfg(executor='single_gpu_sequential',robots=('q1','bumi','g1'),subset_yamls=tuple(str(tmp_path/r) for r in ('q1','bumi','g1')),
        share='all',pool_norms=True,kp_interface=True,kp_student='return',latent_align_weight=0.,compile='none',freeze_receipt='frozen.json')
    with pytest.raises(ValueError,match='executor_subset_sha256'):validate_config(cfg)
    for path in cfg.subset_yamls:Path(path).write_text('original training subset')
    digest=hashlib.sha256(b'original training subset').hexdigest()
    cfg=replace(cfg,executor_subset_sha256=(digest,)*3)
    validate_config(cfg)
    Path(cfg.subset_yamls[-1]).write_text('changed subset')
    called=[]
    monkeypatch.setattr(trainer,'Trainer',lambda *a,**kw:called.append(True))
    with pytest.raises(ValueError,match='subset SHA'):SingleGpuMultiRobotExecutor(cfg,'cuda:0')
    assert not called
