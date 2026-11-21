import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from torch import nn

pytest.importorskip('mjlab')
from morphmimic.constants import NUM_JOINTS,CRITIC_PRIV_DIM
from morphmimic.model.policy import Policy,PolicyCfg,Critic,CriticCfg
from morphmimic.model.world_model import WorldModel,WorldModelCfg
from morphmimic.model.state import StateNormalizer
from morphmimic.train.optim import MuonAdam

ROBOTS=('q1','bumi','g1')


def modules():
    policy=Policy(PolicyCfg(enc_hidden=(8,8),dec_hidden=(8,8),latent=4,human=True,morph_dim=3,film=True))
    wm=WorldModel(WorldModelCfg(hidden=(8,8),morph_dim=4),StateNormalizer(),torch.zeros(NUM_JOINTS,2),torch.zeros(NUM_JOINTS))
    critic=Critic(CriticCfg(hidden=(8,8),morph_dim=4,priv_dim=CRITIC_PRIV_DIM))
    return policy,wm,critic


def internal(module):
    layers=[layer for layer in module.modules() if isinstance(layer,nn.Linear)]
    return layers[1:-1]


def optimize_state(params,networks):
    opt=MuonAdam(params,networks,lr=.001,betas=(.9,.999),fused=False)
    for group in opt.muon.param_groups:
        for p in group['params']:opt.muon.state[p]['momentum_buffer']=torch.ones_like(p)
    return opt.state_dict()


def corpus(root):
    rows=[];yamls=[]
    for robot in ROBOTS:
        folder=root/robot;folder.mkdir();paths=[]
        for dataset in ('seed','lafan1','motiondecode'):
            path=str((folder/(dataset+'.npz')).resolve());paths.append(path)
            rows.append(dict(robot=robot,dataset=dataset,split='train',name=dataset,output=path))
        p=folder/'train.yaml';p.write_text(json.dumps(dict(root='/',clips=paths)));yamls.append(str(p))
    manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(complete=True,clips=rows)))
    return manifest,tuple(yamls),hashlib.sha256(manifest.read_bytes()).hexdigest()


def checkpoints(root,iteration):
    manifest,subset,digest=corpus(root)
    torch.manual_seed(10);policy,wm,critic=modules()
    baseline=dict(policy=copy.deepcopy(policy.state_dict()),wm=copy.deepcopy(wm.state_dict()),critic=copy.deepcopy(critic.state_dict()))
    zero=copy.deepcopy(baseline)
    for net in (wm.dynamics_mlp,policy.encoder.net,policy.decoder.net,policy.human_encoder.net):
        with torch.no_grad():internal(net)[0].weight.add_(.125)
    optim={'wm':optimize_state(wm.parameters(),[wm.dynamics_mlp]),
           'policy':optimize_state(policy.motion_parameters(),[policy.encoder.net,policy.decoder.net]),
           'kp':optimize_state(policy.human_encoder.parameters(),[policy.human_encoder.net])}
    cfg=dict(num_envs=4096,optimizer='muon',robots=ROBOTS,share='all',subset_yamls=subset,
             prepared_manifest=str(manifest),kp_interface=True,policy_goal='motion',kp_masking='five',kp_fixed_horizon=1,
             wm_hidden=(8,8),policy_hidden=(8,8),critic_hidden=(8,8),act='elu',hist_len=10,latent=4,bf16=False,
             wm_layer_norm=False,wm_frame='root',residual_scale='step',termination='hazard',wm_joint_dynamics='residual',
             wm_state_mode='full',critic_layer_norm=False,critic_anchor_pos=True,morphology=True,ref_reconstruction_skip=False,
             kp_student='return',target_critic_alpha=None)
    run=root/'run';run.mkdir()
    normalizers={n:{'mean':torch.zeros(2),'var':torch.ones(2),'count':torch.tensor(4)} for n in ('obs_norm','goal_norm','human_norm','heading_norm')}
    for robot in ROBOTS:
        folder=run/robot;folder.mkdir()
        for it in (0,iteration):
            counts=dict(world_model=0 if it==0 else 1,motion=0 if it==0 else 1,human=0 if it==0 else 1)
            state=zero if it==0 else dict(policy=policy.state_dict(),wm=wm.state_dict(),critic=critic.state_dict())
            c=dict(iteration=it,cfg=dict(cfg,robot=robot,subset_yaml=str(root/robot/'train.yaml')),
                optimized_training=dict(schema_version=1,iteration=it,optimizer='muon',num_envs=4096,
                    joint_motion_human=True,human_horizon=1,human_points='five',prepared_manifest_sha256=digest,
                    optimizer_steps=counts,freeze_receipt=None),
                optimization_state={'optimizer_steps':counts},physics=dict(overflow_bits=0,checked_physics_steps=0 if it==0 else 2),
                morph={'wm':torch.zeros(4),'actor':torch.zeros(3),'film':True},ema_updates=counts['motion'],
                optim=copy.deepcopy(optim),ema_policy=copy.deepcopy(state['policy']),critic_kp=copy.deepcopy(state['critic']),
                **copy.deepcopy(state),**copy.deepcopy(normalizers))
            c['policy']['decoder.joint_mask'][0]=0 if robot=='q1' else 1
            c['ema_policy']['decoder.joint_mask'][0]=0 if robot=='q1' else 1
            torch.save(c,folder/f'ckpt_{it}.pt')
    return manifest,run


@pytest.fixture
def run_data(tmp_path):return checkpoints(tmp_path,1000)


def test_valid_proof_and_robot_buffers_may_differ(run_data):
    from scripts.audit_optimized import audit
    manifest,run=run_data;report=audit(run,manifest,[0,1000],list(ROBOTS))
    assert len(report['checkpoints'])==6
    assert all(value['l2_delta']>0 for value in report['deltas']['1000']['q1'].values())


def mutate(run,robot,fn):
    path=run/robot/'ckpt_1000.pt';obj=torch.load(path,map_location='cpu',weights_only=False);fn(obj);torch.save(obj,path)


def test_rejects_nonfinite_and_divergent_shared_weights(run_data):
    from scripts.audit_optimized import audit
    manifest,run=run_data
    mutate(run,'bumi',lambda c:c['policy']['encoder.net.2.weight'].fill_(float('nan')))
    with pytest.raises(ValueError,match='nonfinite'):audit(run,manifest,[0,1000],list(ROBOTS))
    mutate(run,'bumi',lambda c:c['policy']['encoder.net.2.weight'].fill_(.7))
    with pytest.raises(ValueError,match='Shared'):audit(run,manifest,[0,1000],list(ROBOTS))


def test_rejects_empty_muon_and_dead_actor(run_data):
    from scripts.audit_optimized import audit
    manifest,run=run_data
    mutate(run,'q1',lambda c:c['optim']['policy']['muon']['state'].clear())
    with pytest.raises(ValueError,match='Muon'):audit(run,manifest,[0,1000],['q1'])
    # Restore checkpoint and remove motion learning despite positive claimed step counts.
    baseline=torch.load(run/'q1/ckpt_0.pt',map_location='cpu',weights_only=False)
    def kill(c):
        for key in c['policy']:
            if key.startswith('encoder.net.'):c['policy'][key]=baseline['policy'][key].clone()
        c['optim']['policy']=copy.deepcopy(baseline['optim']['policy'])
    mutate(run,'q1',kill)
    with pytest.raises(ValueError,match='delta'):audit(run,manifest,[0,1000],['q1'])


def test_rejects_wrong_env_overflow_and_optimizer_counter(run_data):
    from scripts.audit_optimized import audit
    manifest,run=run_data
    mutate(run,'q1',lambda c:c['optimized_training'].__setitem__('num_envs',32))
    with pytest.raises(ValueError,match='identity'):audit(run,manifest,[0,1000],['q1'])
    mutate(run,'q1',lambda c:(c['optimized_training'].__setitem__('num_envs',4096),c['physics'].__setitem__('overflow_bits',1)))
    with pytest.raises(ValueError,match='overflow'):audit(run,manifest,[0,1000],['q1'])
    mutate(run,'q1',lambda c:(c['physics'].__setitem__('overflow_bits',0),c['optimization_state']['optimizer_steps'].__setitem__('motion',0)))
    with pytest.raises(ValueError,match='optimizer'):audit(run,manifest,[0,1000],['q1'])


def test_rejects_divergent_ema_and_pooled_norms(run_data):
    from scripts.audit_optimized import audit
    manifest,run=run_data
    mutate(run,'g1',lambda c:c['ema_policy']['encoder.net.2.weight'].add_(1))
    with pytest.raises(ValueError,match='Shared'):audit(run,manifest,[0,1000],list(ROBOTS))
    baseline=torch.load(run/'q1/ckpt_1000.pt',map_location='cpu',weights_only=False)
    mutate(run,'g1',lambda c:(c.__setitem__('ema_policy',copy.deepcopy(baseline['ema_policy'])),c['goal_norm']['mean'].add_(1)))
    with pytest.raises(ValueError,match='pooled norm'):audit(run,manifest,[0,1000],list(ROBOTS))


def test_rejects_ema_counter_and_missing_corpus(run_data):
    from scripts.audit_optimized import audit
    manifest,run=run_data
    mutate(run,'q1',lambda c:c.__setitem__('ema_updates',0))
    with pytest.raises(ValueError,match='EMA'):audit(run,manifest,[0,1000],['q1'])
    data=json.loads(manifest.read_text());data['clips']=[r for r in data['clips'] if not (r['robot']=='g1' and r['dataset']=='motiondecode')]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='corpus'):audit(run,manifest,[0,1000],['q1'])


def test_rejects_zero_muon_momentum(run_data):
    from scripts.audit_optimized import audit
    manifest,run=run_data
    def clear(c):
        for state in c['optim']['kp']['muon']['state'].values():state['momentum_buffer'].zero_()
    mutate(run,'q1',clear)
    with pytest.raises(ValueError,match='momentum is entirely zero'):audit(run,manifest,[0,1000],['q1'])


def test_rejects_nested_nonfinite_momentum(run_data):
    from scripts.audit_optimized import audit
    manifest,run=run_data
    def poison(c):
        states=c['optim']['kp']['muon']['state']
        next(iter(states.values()))['momentum_buffer'].fill_(float('inf'))
    mutate(run,'q1',poison)
    with pytest.raises(ValueError,match='nonfinite'):audit(run,manifest,[0,1000],['q1'])


def test_cli_never_overwrites_report(run_data,tmp_path,monkeypatch):
    import sys
    from scripts.audit_optimized import main
    manifest,run=run_data;out=tmp_path/'audit.json'
    monkeypatch.setattr(sys,'argv',['audit','--run-root',str(run),'--data-manifest',str(manifest),'--out',str(out),'--iterations','1000','--robots','q1'])
    main();before=out.read_bytes()
    with pytest.raises(FileExistsError):main()
    assert out.read_bytes()==before
