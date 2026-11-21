import copy
from pathlib import Path
import pytest,torch
from morphmimic.constants import GOAL_VERSION,KP_GOAL_VERSION
from morphmimic.model.policy import Policy,PolicyCfg
from morphmimic.model.human import pack_human_goal
from morphmimic.train.normalizers import RunningNormalizer
from morphmimic.train.human_adapt import PARENT_SHA256


def checkpoints(tmp_path):
    torch.manual_seed(18)
    model=Policy(PolicyCfg(enc_hidden=(8,),dec_hidden=(8,),latent=4,hist_len=2,morph_dim=154,film=True,human=True))
    morph=torch.zeros(154);morph[-29:]=1;model.set_morphology(morph)
    paths=[]
    for update in (0,1000):
        saved=copy.deepcopy(model.state_dict())
        if update:
            for key in saved:
                if key.startswith('human_encoder.'):saved[key]+=.01
        norm=RunningNormalizer(2160);norm.update(torch.randn(20,2160)+update/1000)
        cfg=dict(robot='g1',policy_hidden=(8,),act='elu',hist_len=2,latent=4,kp_interface=True,kp_fixed_horizon=1)
        path=tmp_path/f'ckpt_{update}.pt'
        torch.save(dict(cfg=cfg,iteration=update,goal_version=GOAL_VERSION,kp_goal_version=KP_GOAL_VERSION,
            policy=saved,ema_policy=saved,morph=dict(actor=morph,film=True),obs_norm=RunningNormalizer(93).state_dict(),goal_norm=RunningNormalizer(670).state_dict(),human_norm=norm.state_dict(),heading_norm=RunningNormalizer(60).state_dict(),
            human_adaptation=dict(parent=dict(iteration=30000,sha256=PARENT_SHA256['g1']),human_updates=update,paired_manifest_sha256='a'*64)),path)
        paths.append(path)
    return paths


def test_checkpoint_routing_matches_official_loader_for_every_row(tmp_path):
    from morphmimic.eval.checkpoint_batch import BatchHumanPolicy
    from morphmimic.eval.policy_loader import load_policy
    paths=checkpoints(tmp_path);obs=torch.randn(5,186)
    p=torch.randn(5,10,24,3);q=torch.zeros(5,10,24,4);q[...,0]=1
    valid=torch.ones(5,24,dtype=torch.bool);valid[1,22:]=False
    goals=dict(goal_kp=pack_human_goal(p,q,torch.tensor([[1.,0,0,0]]).expand(5,-1)),human_joint_valid=valid)
    route=torch.tensor([1,0,1,1,0])
    for points in ('five','full'):
        batch=BatchHumanPolicy(paths,'g1','a'*64,'cpu',points=points)
        actual=batch(obs,goals,route)
        for i in range(5):
            fn,_=load_policy(paths[int(route[i])],'cpu',interface='human',human_points=points,kp_horizon=1)
            expected=fn(obs[i:i+1],{k:v[i:i+1] for k,v in goals.items()})
            torch.testing.assert_close(actual[i:i+1],expected,atol=1e-5,rtol=1e-5)


def test_checkpoint_batch_refuses_different_frozen_decoder(tmp_path):
    from morphmimic.eval.checkpoint_batch import BatchHumanPolicy
    paths=checkpoints(tmp_path);ckpt=torch.load(paths[1],weights_only=False)
    ckpt['ema_policy']['decoder.net.0.weight'][0,0]+=.1;torch.save(ckpt,paths[1])
    with pytest.raises(ValueError,match='frozen'):BatchHumanPolicy(paths,'g1','a'*64,'cpu')


def test_checkpoint_result_split_checks_every_case_name_and_frame_count():
    import pandas as pd
    from morphmimic.eval.checkpoint_batch import split_checkpoint_results
    frame=pd.DataFrame(dict(name=['a','a','b','b'],frames=[70,70,90,90],norm_return=[.1,.2,.3,.4]))
    rows=split_checkpoint_results(frame,['a','b'],[70,90],2)
    assert rows[1].norm_return.tolist()==[.2,.4]
    for broken in (frame.iloc[:-1],frame.assign(name=['a','b','b','b']),frame.assign(frames=[70,71,90,90])):
        with pytest.raises(ValueError,match='case'):split_checkpoint_results(broken,['a','b'],[70,90],2)


def test_checkpoint_batch_refuses_different_activation_configuration(tmp_path):
    from morphmimic.eval.checkpoint_batch import BatchHumanPolicy
    paths=checkpoints(tmp_path);ckpt=torch.load(paths[1],weights_only=False)
    ckpt['cfg']['act']='relu';torch.save(ckpt,paths[1])
    with pytest.raises(ValueError,match='configuration'):BatchHumanPolicy(paths,'g1','a'*64,'cpu')


@pytest.mark.parametrize('kind',('report','raw'))
def test_existing_report_is_rejected_before_policy_or_case_yaml_changes(tmp_path,monkeypatch,kind):
    import json
    from morphmimic.scripts import evaluate_human_batch as evaluator
    root=tmp_path/'data';root.mkdir();(root/'manifest.json').write_text(json.dumps(dict(complete=True,clips=[])))
    out=tmp_path/'out';out.mkdir();prior=out/'g1_val_five_1000.csv' if kind=='report' else out/'raw/g1_val_seed_five.csv';prior.parent.mkdir(exist_ok=True);prior.write_text('prior immutable result')
    subset=tmp_path/'val.yaml';subset.write_text('root: .\nclips: []\n')
    monkeypatch.setattr(evaluator,'verify_prepared_subset',lambda *a,**k:None)
    monkeypatch.setattr(evaluator,'BatchHumanPolicy',lambda *a,**k:(_ for _ in ()).throw(AssertionError('policy loaded too early')))
    with pytest.raises(FileExistsError):evaluator.evaluate_versions([tmp_path/'ckpt_1000.pt'],subset,root,out,robot='g1',split='val')
    assert prior.read_text()=='prior immutable result' and not (out/'subsets').exists()


def test_case_yaml_is_absolute_and_cannot_change_on_rerun(tmp_path):
    import yaml
    from morphmimic.scripts.evaluate_human_batch import write_case_yaml
    path=tmp_path/'cases.yaml';clips=[Path('paired/g1/a.npz'),Path('paired/g1/a.npz')]
    write_case_yaml(path,clips);content=path.read_bytes()
    spec=yaml.safe_load(content);assert Path(spec['root']).is_absolute()
    write_case_yaml(path,clips);assert path.read_bytes()==content
    with pytest.raises(FileExistsError):write_case_yaml(path,clips[:1])
    assert path.read_bytes()==content


def test_precision_report_handles_distinct_cudnn_operator_flags():
    from morphmimic.scripts.evaluate_human_batch import precision_report
    groups=(torch.backends.cuda.matmul,torch.backends.cudnn.conv,torch.backends.cudnn.rnn)
    original=[group.fp32_precision for group in groups]
    try:
        torch.backends.cudnn.conv.fp32_precision='tf32';torch.backends.cudnn.rnn.fp32_precision='ieee'
        with pytest.raises(RuntimeError):_ = torch.backends.cudnn.allow_tf32
        report=precision_report();assert report['cudnn_conv']=='tf32' and report['cudnn_rnn']=='ieee'
    finally:
        for group,value in zip(groups,original):group.fp32_precision=value


def test_group_result_survives_later_failure_and_refuses_overwrite(tmp_path):
    import json,hashlib,pandas as pd
    from morphmimic.scripts.evaluate_human_batch import save_group_result
    frame=pd.DataFrame(dict(name=['a','a'],frames=[70,70],norm_return=[.1,.2]))
    path=tmp_path/'raw/seed_five.csv';identity=dict(checkpoints=['fixed-sha-0','fixed-sha-1000'],subset_sha256='b'*64)
    save_group_result(path,frame,identity)
    saved=json.loads(path.with_suffix('.json').read_text());assert saved['checkpoints']==identity['checkpoints']
    assert saved['per_clip_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    before=path.read_bytes()
    with pytest.raises(FileExistsError):save_group_result(path,frame,identity)
    assert path.read_bytes()==before


def test_full_precision_initialization_sets_all_recorded_operators_to_ieee():
    from morphmimic.scripts.evaluate_human_batch import configure_full_precision,precision_report
    groups=(torch.backends.cuda.matmul,torch.backends.cudnn,torch.backends.cudnn.conv,torch.backends.cudnn.rnn)
    original=[group.fp32_precision for group in groups]
    try:
        for group in groups:group.fp32_precision='tf32'
        configure_full_precision()
        assert set(precision_report().values())=={'ieee'}
    finally:
        for group,value in zip(groups,original):group.fp32_precision=value
