import hashlib
import json
import pytest
import torch

pytest.importorskip('mjlab')
from test_audit_experiment_contract import frozen_run, ROBOTS


def fixture(tmp_path):
    from scripts.audit_optimized import construct_models
    manifest,run,config,receipt=frozen_run(tmp_path,'adam')
    proofs=tmp_path/'proofs';proofs.mkdir()
    for robot in ROBOTS:
        path=run/robot/'ckpt_0.pt';ckpt=torch.load(path,weights_only=False)
        policy,wm,critic=construct_models(ckpt['cfg'],ckpt['morph'])
        hashes={}
        for name,module in [('policy',policy),('wm',wm),('critic',critic),('critic_kp',critic)]:
            h=hashlib.sha256()
            for key,param in module.named_parameters():
                value=ckpt[name][key]
                h.update(key.encode());h.update(str((tuple(value.shape),str(value.dtype))).encode())
                h.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            hashes[name]=h.hexdigest()
        proof=dict(robot=robot,iteration=0,random_initialization=True,previous_checkpoint_loaded=False,
                   optimizer_states_empty=True,parameter_sha256=hashes,
                   normalizer_counts_before_warmup={key:0 for key in ('obs_norm','goal_norm','human_norm','heading_norm')})
        (proofs/f'random-init-{robot}.json').write_text(json.dumps(proof))
    return manifest,run,config,receipt,proofs


def test_fresh_proof_matches_raw_parameters_but_allows_warm_norms(tmp_path):
    from scripts.audit_fresh_initialization import audit_fresh
    manifest,run,config,receipt,proofs=fixture(tmp_path)
    result=audit_fresh(run,manifest,proofs,config,receipt)
    assert result['random_initialization_verified'] is True
    assert len(result['raw_initializations'])==3


@pytest.mark.parametrize('bad',['provenance','parameter','optimizer','raw_norm','raw_loaded'])
def test_rejects_false_fresh_claim(tmp_path,bad):
    from scripts.audit_fresh_initialization import audit_fresh
    manifest,run,config,receipt,proofs=fixture(tmp_path)
    if bad.startswith('raw_'):
        path=proofs/'random-init-q1.json';obj=json.loads(path.read_text())
        if bad=='raw_norm':obj['normalizer_counts_before_warmup']['human_norm']=4
        else:obj['previous_checkpoint_loaded']=True
        path.write_text(json.dumps(obj))
    else:
        path=run/'q1/ckpt_0.pt';obj=torch.load(path,weights_only=False)
        if bad=='provenance':obj['initialized_from']={'path':'old.pt'}
        elif bad=='parameter':
            raw=proofs/'random-init-q1.json';data=json.loads(raw.read_text());data['parameter_sha256']['policy']='0'*64;raw.write_text(json.dumps(data))
        else:obj['optim']['policy']['state'][0]={'step':torch.tensor(1.)}
        torch.save(obj,path)
    with pytest.raises(ValueError):audit_fresh(run,manifest,proofs,config,receipt)
