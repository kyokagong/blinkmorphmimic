"""The online sampler is part of checkpoint identity, never silently discarded."""
from types import SimpleNamespace
import pytest
from morphmimic.train.trainer import Trainer


def trainer(enabled=True):
    obj=Trainer.__new__(Trainer)
    obj.cfg=SimpleNamespace(importance_sampling=enabled)
    obj.command=SimpleNamespace(validate_importance_sampling_state_dict=lambda state: None)
    return obj


def test_importance_checkpoint_epoch_and_mode_are_strict():
    obj=trainer()
    checkpoint=dict(iteration=6250,cfg={'importance_sampling':True},
        importance_sampling={'since_iteration':6000,'state':{}})
    obj._validate_sampler_state(checkpoint)
    for bad in (-1,6251,True):
        with pytest.raises(ValueError):
            obj._validate_sampler_state(dict(checkpoint,importance_sampling={'since_iteration':bad,'state':{}}))
    with pytest.raises(ValueError):trainer(False)._validate_sampler_state(checkpoint)
    del checkpoint['importance_sampling']
    with pytest.raises(ValueError):obj._validate_sampler_state(checkpoint)


def test_legacy_parent_can_be_validated_before_explicit_transition():
    trainer()._validate_sampler_state({'iteration':6000,'cfg':{}})


def test_launcher_preserves_model_and_enables_only_adaptive_mixed_sampling(tmp_path):
    import json
    from scripts.train_importance import config_values
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text(json.dumps({'human_root':'/paired/human'}))
    for name in ('transition.json','windows.json'):(data/name).write_text('{}')
    run=tmp_path/'run';run.mkdir();(run/'parent.json').write_text('{}')
    parent={'optimizer':'muon','share':'all','kp_interface':True,'latent':96,'policy_ema':.995}
    cfg=config_values(parent,data,tmp_path,'run',[0,2,3],'/frozen/receipt.json')
    assert cfg['iterations']==7000 and cfg['num_envs']==2048
    assert cfg['importance_sampling'] and cfg['recovery_reset_share']==0 and cfg['failure_reset']==0
    assert cfg['optimizer']=='muon' and cfg['latent']==96 and cfg['policy_ema']==.995
    assert cfg['lr_decay_iters']==10000 and cfg['ckpt_every']==250
    assert cfg['human_root']=='/paired/human' and cfg['probe_focus_dir'] is None
    for iterations in (6000,7001):
        with pytest.raises(ValueError):config_values(parent,data,tmp_path,'run',[0,2,3],'/f',iterations=iterations)
