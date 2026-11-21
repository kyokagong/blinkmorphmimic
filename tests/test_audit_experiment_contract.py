"""Frozen experiment expectations cannot be relaxed by an audit CLI flag."""
import copy
from dataclasses import asdict
import json
import pytest
import torch

pytest.importorskip('mjlab')
from test_audit_optimized import checkpoints, ROBOTS


def frozen_run(tmp_path, optimizer='muon'):
    from morphmimic.train.trainer import TrainCfg
    from morphmimic.train.run_identity import create_receipt, write_receipt
    from scripts.audit_optimized import construct_models
    manifest, run = checkpoints(tmp_path, 1000)
    first = torch.load(run/'q1/ckpt_0.pt', weights_only=False)
    values = first['cfg'].copy()
    values.update(num_envs=2048, optimizer=optimizer, iterations=5000, lr_decay_iters=5000,
                  robot='g1', subset_yaml='', morphology=False, decoder_film=None,
                  latent_align_weight=0., initial_checkpoint_root=None, optimization_schema=1)
    cfg = asdict(TrainCfg(**values))
    config = tmp_path/'config.json'; receipt = tmp_path/'freeze.json'
    cfg['freeze_receipt'] = str(receipt)
    config.write_text(json.dumps(cfg))
    source = tmp_path/'source'; source.mkdir(); (source/'README.md').write_text('fixture')
    frozen = create_receipt(source, 'sha256:'+'a'*64, cfg, first['optimized_training']['prepared_manifest_sha256'])
    write_receipt(receipt, frozen)
    for robot in ROBOTS:
        for it in (0,1000):
            path = run/robot/f'ckpt_{it}.pt'; ckpt = torch.load(path, weights_only=False)
            ckpt['cfg'] = dict(cfg, robot=robot, subset_yaml=str(tmp_path/robot/'train.yaml'),
                               morphology=True, decoder_film=True)
            ckpt['optimized_training'].update(num_envs=2048, optimizer=optimizer, freeze_receipt=frozen)
            if optimizer == 'adam':
                policy, wm, critic = construct_models(ckpt['cfg'], ckpt['morph'])
                ckpt['optim'] = {}
                for name, params in [('wm', wm.parameters()), ('policy', policy.motion_parameters()),
                                     ('kp', policy.human_parameters()), ('critic', critic.parameters()),
                                     ('critic_kp', critic.parameters())]:
                    params = list(params); opt = torch.optim.Adam(params)
                    if it:
                        for p in params: p.grad = torch.ones_like(p)
                        # Production fit_critic performs 5 epochs x 4 minibatches.
                        for _ in range(20 if name.startswith('critic') else 1):opt.step()
                    ckpt['optim'][name] = copy.deepcopy(opt.state_dict())
            torch.save(ckpt,path)
    return manifest, run, config, receipt


def test_nonlegacy_expectations_require_frozen_files(tmp_path):
    from scripts.audit_optimized import audit
    manifest,run=checkpoints(tmp_path,1000)
    with pytest.raises(ValueError,match='frozen'):
        audit(run,manifest,[1000],['q1'],expected_num_envs=2048)


@pytest.mark.parametrize('optimizer',['muon','adam'])
def test_frozen_2048_contract_and_optimizer_evidence(tmp_path,optimizer):
    from scripts.audit_optimized import audit
    manifest,run,config,receipt=frozen_run(tmp_path,optimizer)
    report=audit(run,manifest,[1000],list(ROBOTS),frozen_config=config,freeze_receipt=receipt,
                 expected_num_envs=2048,expected_optimizer=optimizer)
    assert report['experiment']['num_envs']==2048 and report['experiment']['iterations']==5000
    row=next(r for r in report['checkpoints'] if r['iteration']==1000)
    assert row[optimizer]['motion']
    if optimizer=='adam':assert row['adam']['motion']['state_steps']==1
    with pytest.raises(ValueError,match='expected'):
        audit(run,manifest,[1000],['q1'],frozen_config=config,freeze_receipt=receipt,expected_num_envs=4096)
    value=json.loads(config.read_text());value['seed']=99;config.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Configuration'):
        audit(run,manifest,[1000],['q1'],frozen_config=config,freeze_receipt=receipt)


def test_frozen_checkpoint_cfg_cannot_drift(tmp_path):
    from scripts.audit_optimized import audit
    manifest,run,config,receipt=frozen_run(tmp_path)
    path=run/'q1/ckpt_1000.pt';ckpt=torch.load(path,weights_only=False)
    ckpt['cfg']['seed']=99;torch.save(ckpt,path)
    with pytest.raises(ValueError,match='configuration'):
        audit(run,manifest,[1000],['q1'],frozen_config=config,freeze_receipt=receipt)


@pytest.mark.parametrize('corrupt',['missing','step','zero','negative'])
def test_adam_requires_real_moments_and_exact_counters(tmp_path,corrupt):
    from scripts.audit_optimized import audit
    manifest,run,config,receipt=frozen_run(tmp_path,'adam')
    path=run/'q1/ckpt_1000.pt';ckpt=torch.load(path,weights_only=False)
    states=ckpt['optim']['policy']['state']
    if corrupt=='missing':states.pop(next(iter(states)))
    elif corrupt=='step':next(iter(states.values()))['step'].fill_(2)
    elif corrupt=='zero':
        for state in states.values():state['exp_avg'].zero_();state['exp_avg_sq'].zero_()
    else:next(iter(states.values()))['exp_avg_sq'].fill_(-1)
    torch.save(ckpt,path)
    with pytest.raises(ValueError,match='Adam'):
        audit(run,manifest,[1000],['q1'],frozen_config=config,freeze_receipt=receipt)


def test_adam_critic_counter_counts_all_minibatches(tmp_path):
    from scripts.audit_optimized import audit
    manifest,run,config,receipt=frozen_run(tmp_path,'adam')
    path=run/'q1/ckpt_1000.pt';ckpt=torch.load(path,weights_only=False)
    for state in ckpt['optim']['critic']['state'].values():state['step'].fill_(1)
    torch.save(ckpt,path)
    with pytest.raises(ValueError,match='critic: Adam step'):
        audit(run,manifest,[1000],['q1'],frozen_config=config,freeze_receipt=receipt)


@pytest.mark.parametrize('optimizer',['muon','adam'])
def test_legacy_frozen_zero_to_5000_contract_remains_supported(tmp_path,optimizer):
    from scripts.audit_optimized import audit
    manifest,run,config,receipt=frozen_run(tmp_path,optimizer)
    for robot in ROBOTS:
        ckpt=torch.load(run/robot/'ckpt_1000.pt',weights_only=False)
        ckpt['iteration']=5000;ckpt['optimized_training']['iteration']=5000
        torch.save(ckpt,run/robot/'ckpt_5000.pt')
    result=audit(run,manifest,[0,5000],list(ROBOTS),frozen_config=config,freeze_receipt=receipt)
    assert len(result['checkpoints'])==6
    assert result['experiment']['num_envs']==2048 and result['experiment']['iterations']==5000
