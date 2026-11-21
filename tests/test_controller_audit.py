import copy
from dataclasses import asdict
import pytest
import torch
from morphmimic.train.trainer import TrainCfg
from scripts.audit_optimized import construct_models,finite_tree


@pytest.mark.parametrize('robot',['q1','bumi','g1'])
def test_audit_rebuilds_actual_cpu_compiled_controller_plant(robot):
    from morphmimic.env.controller_recording import controller_from_config
    cfg=asdict(TrainCfg(robot=robot,controller_mode='native_delay_v1',wm_hidden=(8,8),policy_hidden=(8,8),critic_hidden=(8,8)))
    c=controller_from_config(cfg)
    morph={'wm':torch.zeros(4),'actor':torch.zeros(3),'film':True}
    _,wm,_=construct_models(cfg,morph,controller_identity=c.identity())
    assert wm.dynamics_mlp[0].in_features==641
    assert wm.controller.identity()==c.identity()
    finite_tree(wm.state_dict())
    assert not c.clip_enabled
    # Disabled clipping is truly unbounded, rather than a finite saturation approximation.
    action=torch.full((1,29),1e35)
    torch.testing.assert_close(c.target(action,torch.zeros_like(action)),(c.q0+action*c.scale)*c.mask)
    from morphmimic.model.controller import validate_controller_checkpoint
    checkpoint=dict(cfg=cfg,wm=copy.deepcopy(wm.state_dict()),controller_identity=c.identity())
    validate_controller_checkpoint(wm,checkpoint)
    checkpoint['wm']['controller.scale'][0]+=.1
    with pytest.raises(ValueError,match='buffers'):validate_controller_checkpoint(wm,checkpoint)


@pytest.mark.parametrize('bad',['missing','mode','timing','plant','mapping'])
def test_audit_rejects_untrusted_mode_cfg_or_controller_identity(bad):
    from morphmimic.env.controller_recording import controller_from_config
    cfg=asdict(TrainCfg(robot='g1',controller_mode='native_delay_v1',wm_hidden=(8,8),policy_hidden=(8,8),critic_hidden=(8,8)))
    identity=controller_from_config(cfg).identity()
    if bad=='missing':identity=None
    elif bad=='mode':cfg['controller_mode']='legacy'
    elif bad=='timing':cfg['actuator_delay_max_ms']=10
    elif bad=='plant':identity['plant']['actuators']['actuator_gainprm'][0][0]+=.5
    else:identity['mapping']['scale'][0]+=.2
    with pytest.raises(ValueError,match='controller'):
        construct_models(cfg,{'wm':torch.zeros(4),'actor':torch.zeros(3),'film':True},controller_identity=identity)


def native_checkpoint_fixture(tmp_path):
    """CPU checkpoint-schema fixture with real models/Adam and CPU-compiled native plants.

    No simulator trajectories or claim of real Trainer/GPU qualification are made.
    """
    import json
    from test_audit_experiment_contract import frozen_run,ROBOTS
    from morphmimic.train.run_identity import create_receipt
    from morphmimic.env.controller_recording import controller_from_config
    from scripts.audit_fresh_initialization import parameter_hash
    manifest,run,config,receipt=frozen_run(tmp_path,'adam')
    frozen_cfg=json.loads(config.read_text());frozen_cfg['controller_mode']='native_delay_v1'
    config.write_text(json.dumps(frozen_cfg))
    old=json.loads(receipt.read_text())
    frozen=create_receipt(tmp_path/'source',old['runtime_image_id'],frozen_cfg,old['data_manifest_sha256'])
    receipt.write_text(json.dumps(frozen))
    proofs=tmp_path/'raw-proof';proofs.mkdir()
    for robot in ROBOTS:
        path=run/robot/'ckpt_0.pt';zero=torch.load(path,weights_only=False)
        cfg=dict(zero['cfg'],controller_mode='native_delay_v1')
        identity=controller_from_config(cfg).identity()
        torch.manual_seed(421)
        policy,wm,critic=construct_models(cfg,zero['morph'],controller_identity=identity)
        critic_kp=copy.deepcopy(critic)
        modules={'wm':wm,'policy':policy,'critic':critic,'critic_kp':critic_kp}
        optimizers={name:torch.optim.Adam(list(params)) for name,params in (
            ('wm',wm.parameters()),('policy',policy.motion_parameters()),('kp',policy.human_parameters()),
            ('critic',critic.parameters()),('critic_kp',critic_kp.parameters()))}
        for iteration in (0,5000):
            if iteration:
                for name,opt in optimizers.items():
                    for group in opt.param_groups:
                        for p in group['params']:p.grad=torch.ones_like(p)
                    for _ in range(20 if name.startswith('critic') else 1):opt.step()
            ckpt=copy.deepcopy(zero)
            ckpt.update(cfg=cfg,iteration=iteration,controller_identity=identity,
                        optim={k:copy.deepcopy(v.state_dict()) for k,v in optimizers.items()},
                        **{k:copy.deepcopy(v.state_dict()) for k,v in modules.items()})
            ckpt['ema_policy']=copy.deepcopy(ckpt['policy'])
            counts={k:1 if iteration else 0 for k in ('world_model','motion','human')}
            ckpt['ema_updates']=counts['motion'];ckpt['optimization_state']['optimizer_steps']=counts
            ckpt['optimized_training'].update(iteration=iteration,optimizer_steps=counts,freeze_receipt=frozen)
            ckpt['physics']['checked_physics_steps']=1 if iteration else 0
            torch.save(ckpt,run/robot/f'ckpt_{iteration}.pt')
            if iteration==0:
                hashes={k:parameter_hash(v,ckpt[k]) for k,v in modules.items()}
                proof=dict(robot=robot,iteration=0,random_initialization=True,previous_checkpoint_loaded=False,
                           optimizer_states_empty=True,parameter_sha256=hashes,
                           normalizer_counts_before_warmup={k:0 for k in ('obs_norm','goal_norm','human_norm','heading_norm')})
                (proofs/f'random-init-{robot}.json').write_text(json.dumps(proof))
    return manifest,run,config,receipt,proofs


def test_native_mode_zero_final_audit_and_fresh_hashes(tmp_path):
    from scripts.audit_optimized import audit
    from scripts.audit_fresh_initialization import audit_fresh
    manifest,run,config,receipt,proofs=native_checkpoint_fixture(tmp_path)
    result=audit(run,manifest,[0,5000],['q1','bumi','g1'],frozen_config=config,freeze_receipt=receipt)
    assert len(result['checkpoints'])==6
    assert all(row['adam']['world_model'] for row in result['checkpoints'] if row['iteration'])
    fresh=audit_fresh(run,manifest,proofs,config,receipt)
    assert fresh['random_initialization_verified'] and len(fresh['raw_initializations'])==3


@pytest.mark.parametrize('bad',['shape','buffer','extra_buffer','plant','cfg','mode'])
def test_full_native_audit_and_fresh_entrypoints_reject_tampering(tmp_path,bad):
    from scripts.audit_optimized import audit
    from scripts.audit_fresh_initialization import audit_fresh
    manifest,run,config,receipt,proofs=native_checkpoint_fixture(tmp_path)
    path=run/'q1/ckpt_0.pt';ckpt=torch.load(path,weights_only=False)
    if bad=='shape':ckpt['wm']['dynamics_mlp.0.weight']=ckpt['wm']['dynamics_mlp.0.weight'][:,:-290]
    elif bad=='buffer':ckpt['wm']['controller.scale'][0]+=.1
    elif bad=='extra_buffer':ckpt['wm']['controller.hidden_extra']=torch.ones(1)
    elif bad=='plant':ckpt['controller_identity']['plant']['actuators']['actuator_gainprm'][0][0]+=.5
    elif bad=='cfg':ckpt['cfg']['actuator_delay_max_ms']=10
    else:ckpt['cfg']['controller_mode']='legacy'
    torch.save(ckpt,path)
    with pytest.raises(ValueError):audit(run,manifest,[0,5000],['q1'],frozen_config=config,freeze_receipt=receipt)
    with pytest.raises(ValueError):audit_fresh(run,manifest,proofs,config,receipt)
