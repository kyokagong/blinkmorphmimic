import pytest
import torch


def test_fixed_horizon_is_causal_for_both_full_and_five_masks():
    from morphmimic.model.human import sample_human_masks
    for mode in ('full','five','mixed'):
        points,frames,heading=sample_human_masks(32,mode,torch.device('cpu'),fixed_horizon=1)
        assert frames[:,0].all() and not frames[:,1:].any()
        if mode=='five':assert points.sum(-1).eq(5).all()
    with pytest.raises(ValueError):sample_human_masks(1,'five',torch.device('cpu'),fixed_horizon=0)


def test_library_joint_availability_follows_clip_identity():
    from morphmimic.data.motion_library import MotionLibrary
    lib=MotionLibrary.__new__(MotionLibrary)
    lib.frame_clip=torch.tensor([0,0,1,1])
    lib.human_joint_valid=torch.ones(2,24,dtype=torch.bool);lib.human_joint_valid[1,22:]=False
    mask=lib.human_valid_at(torch.tensor([0,2,3,1]))
    assert mask[:,22].tolist()==[True,False,False,True]
    assert mask[:,[0,7,8,20,21]].all()


def _teacher_fixture(tmp_path):
    import copy,hashlib
    from types import SimpleNamespace
    from torch import nn
    from morphmimic.model.policy import Policy,PolicyCfg
    from morphmimic.train.normalizers import RunningNormalizer
    cfg=PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4)
    teacher=Policy(cfg);ema=copy.deepcopy(teacher)
    with torch.no_grad():
        for p in ema.parameters():p.add_(.02)
    obs=RunningNormalizer(93);obs.update(torch.randn(16,93));goal=RunningNormalizer(670);goal.update(torch.randn(16,670))
    wm=nn.Linear(2,2);critic=nn.Linear(2,1)
    checkpoint=dict(iteration=30000,cfg=dict(robot='q1',policy_goal='motion'),policy=teacher.state_dict(),ema_policy=ema.state_dict(),wm=wm.state_dict(),critic=critic.state_dict(),obs_norm=obs.state_dict(),goal_norm=goal.state_dict())
    path=tmp_path/'q1/ckpt_30000.pt';path.parent.mkdir();torch.save(checkpoint,path)
    student=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,human=True))
    trainer=SimpleNamespace(cfg=SimpleNamespace(robot='q1',policy_ema=.995),policy=student,wm=nn.Linear(2,2),critic=nn.Linear(2,1),obs_norm=RunningNormalizer(93),goal_norm=RunningNormalizer(670),device='cpu',policy_avg=None)
    return trainer,path,hashlib.sha256(path.read_bytes()).hexdigest(),ema,checkpoint


def test_warm_start_uses_ema_and_preserves_motion_while_human_learns(tmp_path):
    from morphmimic.train.human_adapt import initialize_teacher,teacher_fingerprint
    from morphmimic.constants import HUMAN_GOAL_DIM
    t,path,sha,ema,source=_teacher_fixture(tmp_path)
    human_before={k:v.clone() for k,v in t.policy.human_encoder.state_dict().items()}
    parent=initialize_teacher(t,path,sha)
    assert parent['iteration']==30000 and parent['sha256']==sha and t.it==0
    for k,v in human_before.items():torch.testing.assert_close(t.policy.human_encoder.state_dict()[k],v,rtol=0,atol=0)
    obs=torch.randn(4,930);goal=torch.randn(4,670)
    torch.testing.assert_close(t.policy(obs,goal),ema(obs,goal),rtol=0,atol=0)
    before=teacher_fingerprint(t)
    count=t.obs_norm.count.clone();t.obs_norm.update(torch.randn(9,93));assert t.obs_norm.count==count
    opt=torch.optim.Adam(t.policy.human_parameters(),lr=.01)
    out=t.policy.forward_human(obs,torch.randn(4,HUMAN_GOAL_DIM));out.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in t.policy.human_parameters())
    opt.step();t.policy_avg.update(t.policy)
    assert teacher_fingerprint(t)==before
    torch.testing.assert_close(t.policy(obs,goal),ema(obs,goal),rtol=0,atol=0)
    torch.testing.assert_close(t.policy_avg.module(obs,goal),ema(obs,goal),rtol=0,atol=0)


def test_warm_start_rejects_changed_or_wrong_robot_teacher(tmp_path):
    from morphmimic.train.human_adapt import initialize_teacher
    t,path,sha,_,source=_teacher_fixture(tmp_path)
    with pytest.raises(ValueError,match='SHA'):initialize_teacher(t,path,'0'*64)
    t.cfg.robot='g1'
    with pytest.raises(ValueError,match='identity'):initialize_teacher(t,path,sha)


def test_student_collection_fits_human_statistics_only_on_calibrated_sources():
    from types import SimpleNamespace
    from morphmimic.train.human_adapt import HumanAdaptTrainer
    from morphmimic.train.normalizers import RunningNormalizer
    from morphmimic.data.motion_library import MotionLibrary
    from morphmimic.model.policy import Policy,PolicyCfg
    from morphmimic.model.human import pack_human_goal
    t=HumanAdaptTrainer.__new__(HumanAdaptTrainer)
    t.cfg=SimpleNamespace(kp_interface=True,policy_goal='motion',kp_drift=False)
    t.library=MotionLibrary.__new__(MotionLibrary);t.library.frame_clip=torch.tensor([0,1]);t.library.human_joint_valid=torch.ones(2,24,dtype=torch.bool);t.library.human_joint_valid[1,22:]=False
    t.command=SimpleNamespace(global_frame=torch.tensor([0,1]));t.kp_point_mask=t.library.human_joint_valid.clone()
    t.kp_frame_mask=torch.zeros(2,10,dtype=torch.bool);t.kp_frame_mask[:,0]=True
    t.kp_heading_mask=torch.ones(2,dtype=torch.bool);t.kp_env=torch.ones(2,dtype=torch.bool)
    t.policy=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,human=True))
    t.obs_norm=RunningNormalizer(93).eval();t.goal_norm=RunningNormalizer(670).eval()
    t.human_norm=RunningNormalizer(2160);t.heading_norm=RunningNormalizer(60)
    pos=torch.zeros(2,10,24,3);pos[...,2]=1;pos[1,:,22:]=100000
    quat=torch.zeros(2,10,24,4);quat[...,0]=1
    raw=pack_human_goal(pos,quat,torch.tensor([[1.,0,0,0],[1.,0,0,0]]))
    action=t._policy_fn(torch.zeros(2,930),{'goal':torch.zeros(2,670),'goal_kp':raw})
    assert torch.isfinite(action).all()
    assert t.human_norm.count==1 and t.heading_norm.count==2
    assert t.obs_norm.count==t.goal_norm.count==0
    assert not t.human_norm.training


def test_human_iteration_never_calls_motion_or_world_model_updates():
    from types import SimpleNamespace
    from morphmimic.train.human_adapt import HumanAdaptTrainer
    calls=[];t=HumanAdaptTrainer.__new__(HumanAdaptTrainer)
    t.cfg=SimpleNamespace(lr_decay_iters=10000,lr_final_frac=.1,target_critic_alpha=None)
    t.steps=SimpleNamespace(cudagraphs=False);t._base_lr=[];t._cuda=False;t.it=0
    t._collect_steps=lambda:1;t.collector=SimpleNamespace(step=lambda n:calls.append('collect'))
    t.improve_policy_kp=lambda:(None,{'kp/loss':torch.tensor(1.)})
    t.fit_world_model=lambda:(_ for _ in ()).throw(AssertionError('teacher WM changed'))
    t.improve_policy=lambda *a:(_ for _ in ()).throw(AssertionError('teacher motion changed'))
    t.critic_kp=None;t.target_critic_kp=None;t.policy_avg=None
    t.save_checkpoint=lambda:calls.append('checkpoint')
    result=t.iteration()
    assert t.it==1 and 'kp/loss' in result and calls==['checkpoint','collect']


def test_training_recipe_keeps_teacher_architecture_and_uses_new_run(tmp_path):
    from types import SimpleNamespace
    from morphmimic.scripts.train_human import build_cfg
    args=SimpleNamespace(data_root=str(tmp_path/'paired'),teacher_root=str(tmp_path/'teacher'),
        run_root=str(tmp_path/'runs'),run_name='human-v1',updates=10000,num_envs=2048,
        gpu_ids=[0,2,3],compile='none',warmup=100)
    cfg=build_cfg(dict(policy_hidden=(16,8),latent=4,actor_geometry=True),args)
    assert cfg.policy_hidden==(16,8) and cfg.latent==4
    assert cfg.robots==('q1','bumi','g1') and cfg.share=='all'
    assert cfg.kp_interface and cfg.policy_goal=='motion' and cfg.kp_fixed_horizon==1
    assert cfg.iterations==cfg.lr_decay_iters==10000 and cfg.ckpt_every==1000
    assert cfg.num_envs==2048 and cfg.gpu_ids==[0,2,3]
    assert cfg.human_root==str(tmp_path/'paired/human50')
    assert cfg.subset_yamls[0]==str(tmp_path/'paired/q1/train.yaml')


def test_evaluation_intersects_available_joints_without_mutating_cached_mask(tmp_path,monkeypatch):
    from morphmimic.eval import policy_loader as loader
    from morphmimic.model.policy import Policy,PolicyCfg
    from morphmimic.train.normalizers import RunningNormalizer
    from morphmimic.constants import GOAL_VERSION,KP_GOAL_VERSION
    p=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,human=True))
    ckpt=dict(iteration=1,goal_version=GOAL_VERSION,kp_goal_version=KP_GOAL_VERSION,
        cfg=dict(robot='g1',policy_hidden=(8,),act='elu',hist_len=10,latent=4),policy=p.state_dict(),
        obs_norm=RunningNormalizer(93).state_dict(),goal_norm=RunningNormalizer(670).state_dict(),
        human_norm=RunningNormalizer(2160).state_dict(),heading_norm=RunningNormalizer(60).state_dict())
    path=tmp_path/'ckpt_1.pt';torch.save(ckpt,path)
    received=[]
    monkeypatch.setattr(loader,'act_human',lambda *args:received.append(args[-3].clone()))
    fn,_=loader.load_policy(path,device='cpu',interface='human',human_points='full',kp_horizon=1)
    valid=torch.ones(1,24,dtype=torch.bool);valid[:,22:]=False
    fn(torch.zeros(1,930),{'goal_kp':torch.zeros(1,1684),'human_joint_valid':valid})
    fn(torch.zeros(1,930),{'goal_kp':torch.zeros(1,1684),'human_joint_valid':torch.ones_like(valid)})
    assert received[0].sum()==22 and received[1].sum()==24


def test_human_logging_does_not_require_frozen_world_model_training_metrics(capsys):
    from morphmimic.train.human_adapt import HumanAdaptTrainer
    t=HumanAdaptTrainer.__new__(HumanAdaptTrainer);t.it=2
    t._print_line({'perf/it_per_s':1.,'perf/gpu_mem_gb':7.,'kp/return':.7,
                   'kp/action_err':.1,'kp/align_err':.2,'kp/nonfinite_steps':0})
    text=capsys.readouterr().out
    assert 'human update 2' in text and 'parent 30000' in text


def test_checkpoint_candidate_requires_three_robot_validation_and_ranks_failures_first():
    from morphmimic.scripts.evaluate_human import choose_candidate
    rows=[]
    for update,failed,value in ((0,0,.99),(1000,2,.9),(2000,1,.7),(3000,1,.8)):
        for robot in ('q1','bumi','g1'):
            rows.append(dict(human_updates=update,robot=robot,split='val',points='five',failures=failed,mean_norm_return=value))
    rows.append(dict(human_updates=4000,robot='q1',split='val',points='five',failures=0,mean_norm_return=1.))
    assert choose_candidate(rows)['human_updates']==3000


@pytest.mark.parametrize('change',['robot','iteration','parent','updates','data'])
def test_adaptation_identity_rejects_a_mislabeled_run(change):
    from morphmimic.train.human_adapt import verify_adaptation_identity,PARENT_SHA256
    ckpt=dict(iteration=1000,cfg=dict(robot='q1',kp_interface=True,kp_fixed_horizon=1,policy_goal='motion'),
        human_adaptation=dict(parent=dict(iteration=30000,sha256=PARENT_SHA256['q1']),human_updates=1000,paired_manifest_sha256='a'*64))
    verify_adaptation_identity(ckpt,robot='q1',update=1000,data_sha256='a'*64)
    if change=='robot':ckpt['cfg']['robot']='g1'
    elif change=='iteration':ckpt['iteration']=2000
    elif change=='parent':ckpt['human_adaptation']['parent']['sha256']='b'*64
    elif change=='updates':ckpt['human_adaptation']['human_updates']=2000
    else:ckpt['human_adaptation']['paired_manifest_sha256']='b'*64
    with pytest.raises(ValueError,match='identity'):verify_adaptation_identity(ckpt,robot='q1',update=1000,data_sha256='a'*64)


def test_evaluation_groups_datasets_and_preserves_every_validation_clip(tmp_path,monkeypatch):
    import hashlib,json,numpy as np,pandas as pd,yaml
    from morphmimic.scripts import evaluate_human as evaluator
    from morphmimic.train.human_adapt import PARENT_SHA256
    root=tmp_path/'paired';rows=[]
    for name,dataset in (('seed_val','seed'),('lafan_val','lafan1')):
        row=dict(name=name,robot='q1',dataset=dataset,split='val',output='q1/'+name+'.npz',human_sidecar='human50/q1/'+name+'.npz')
        for field,key in (('output','output_sha256'),('human_sidecar','human_sha256')):
            p=root/row[field];p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,x=np.array([1.]))
            row[key]=hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(row)
    manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(clips=rows)));digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    subset=root/'q1/val.yaml';subset.write_text(yaml.safe_dump(dict(root='.',clips=['seed_val.npz','lafan_val.npz'])))
    checkpoint=tmp_path/'q1/ckpt_1000.pt';checkpoint.parent.mkdir();torch.save(dict(iteration=1000,policy={},
        cfg=dict(robot='q1',kp_interface=True,kp_fixed_horizon=1),human_adaptation=dict(parent=dict(iteration=30000,sha256=PARENT_SHA256['q1']),human_updates=1000,paired_manifest_sha256=digest)),checkpoint)
    calls=[]
    def evaluate(path,yaml_path,device,**kwargs):
        names=[n.removesuffix('.npz') for n in yaml.safe_load(open(yaml_path))['clips']];calls.append(set(names))
        df=pd.DataFrame(dict(name=names,norm_return=[.5]*len(names),failed=[False]*len(names)))
        count=24 if 'seed_val' in names else 16
        df.attrs.update(mode='human-five-h1',physics=dict(overflow_bits=0),available_human_joints={str(count):len(names)})
        return df
    monkeypatch.setattr(evaluator,'evaluate',evaluate)
    monkeypatch.setattr(evaluator,'summarize',lambda df,_:dict(num_clips=len(df),mean_norm_return=float(df.norm_return.mean()),failures=int(df.failed.sum())))
    out=tmp_path/'evaluation';out.mkdir();categories=pd.DataFrame(dict(name=['seed_val','lafan_val'],category=['seed','lafan1']))
    summary=evaluator.evaluate_one(checkpoint,subset,digest,out,robot='q1',split='val',points='five',update=1000,num_envs=256,categories=categories,human_root=str(root/'human50'))
    assert calls==[{'lafan_val'},{'seed_val'}] or calls==[{'seed_val'},{'lafan_val'}]
    assert summary['num_clips']==2 and summary['available_human_joints']=={'16':1,'24':1}
