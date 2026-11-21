import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest


def test_default_preset_is_fresh_muon_2048_by_5000():
    from scripts.train_optimized import config_values
    c=config_values('/data','/human','/runs','fresh',[0,2,3])
    assert (c['num_envs'],c['iterations'],c['lr_decay_iters'])==(2048,5000,5000)
    assert c['optimizer']=='muon' and c['initial_checkpoint_root'] is None


def test_actual_dry_run_cli_keeps_fresh_configuration(tmp_path,capsys):
    from scripts.train_optimized import main
    (tmp_path/'manifest.json').write_text(json.dumps({'complete':True,'human_root':'/human'}))
    argv=['train','--data-root',str(tmp_path),'--log-dir',str(tmp_path),'--run-name','new',
        '--gpu-ids','0','2','3','--dry-run','--freeze-receipt',str(tmp_path/'future-receipt.json')]
    with patch.object(sys,'argv',argv):main()
    c=json.loads(capsys.readouterr().out)
    assert c['initial_checkpoint_root'] is None and c['optimizer']=='muon'
    assert c['num_envs']==2048 and c['iterations']==c['lr_decay_iters']==5000
    assert not (tmp_path/'new').exists()

    # The documented dry-run -> freeze route uses the exact same future receipt path.
    from scripts.freeze_optimized import main as freeze
    from morphmimic.train.run_identity import ManifestIdentity,read_receipt
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    source=tmp_path/'source';source.mkdir();(source/'README.md').write_text('source fixture')
    image='sha256:'+'a'*64
    with patch.object(sys,'argv',['freeze','--data-manifest',str(tmp_path/'manifest.json'),
        '--source-root',str(source),'--config',str(config),'--image-id',image,
        '--output',c['freeze_receipt']]):freeze()
    read_receipt(c['freeze_receipt'],ManifestIdentity(tmp_path/'manifest.json').sha256,
        source_root=source,image_id=image,config=c)


def test_legacy_selection_remains_strict_and_explicit():
    from scripts.evaluate_optimized import validate_selection
    assert validate_selection(None,None)[0]==list(range(0,5001,1000))
    assert validate_selection([6000,10000],['q1'])==([6000,10000],['q1'])
    for bad in [[],[5000,5000],[5001],[11000],[True]]:
        with pytest.raises(ValueError):validate_selection(bad,None)


def test_default_watcher_stops_at_5000(tmp_path):
    import scripts.watch_evaluations as s
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text('{"complete":true,"clips":[]}')
    out=tmp_path/'out';seen=[]
    argv=['watch','--run-root',str(tmp_path/'run'),'--data-root',str(data),'--out-root',str(out)]
    def ready(root,it):
        assert it<=5000,'default watcher waited for a nonexistent6000 checkpoint'
        if it not in seen:seen.append(it)
        folder=out/f'iteration_{it:05d}';folder.mkdir(exist_ok=True);(folder/'results.json').write_text('[]')
        return {'q1':{'sha256':'a'*64}}
    with patch.object(sys,'argv',argv),patch.object(s,'ready_checkpoints',ready),patch.object(s,'verify_result',return_value=[]):s.main()
    assert seen==list(range(0,5001,1000))


def test_plot_default_rows_stop_at_5000_legacy_still_available():
    from scripts.plot_optimized import training_checkpoint_rows
    train={'q1':{'status':'ok','tags':{'loss':[]}}}
    assert {r['iteration'] for r in training_checkpoint_rows(train)}==set(range(0,5001,1000))
    assert {r['iteration'] for r in training_checkpoint_rows(train,[6000,10000])}=={6000,10000}


def probe_args(tmp_path,target=None):
    run=tmp_path/'run';out=tmp_path/'out';run.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    for r in ['q1','bumi','g1']:
        (run/r).mkdir(exist_ok=True);p=run/r/f'ckpt_{target or 5000}.pt';p.write_bytes(b'fixed')
    argv=['probe','--run-root',str(run),'--data-root',str(tmp_path),'--output',str(out),
        '--training-start-epoch','100']
    if target:argv+=['--training-target',str(target)]
    return argv,out


@pytest.mark.parametrize('target',[None,10000])
def test_probe_final_checkpoint_without_budget_exits(tmp_path,target):
    import scripts.probe_optimized as s
    argv,out=probe_args(tmp_path,target)
    with patch.object(sys,'argv',argv),patch('morphmimic.scripts.gpu_guard.select_gpu'),\
         patch.object(s,'budget_remaining',return_value=0),\
         patch.object(s.time,'sleep',side_effect=AssertionError('final checkpoint must not wait')):
        s.main()
    ledger=json.loads((out/'budget.json').read_text())
    assert ledger['training_target']==(target or 5000)
    assert ledger['incomplete_robots']==['q1','bumi','g1']
    assert 'no probe fits' in ledger['stop_reason']


@pytest.mark.parametrize('stored_target',[None,10000,True])
def test_probe_rejects_unbound_or_changed_ledger_target(tmp_path,stored_target):
    import scripts.probe_optimized as s
    argv,out=probe_args(tmp_path)
    ledger=dict(training_start_epoch=100.,used_seconds=0.,completed={})
    if stored_target is not None:ledger['training_target']=stored_target
    path=out/'budget.json';path.write_text(json.dumps(ledger));original=path.read_bytes()
    with patch.object(sys,'argv',argv),patch('morphmimic.scripts.gpu_guard.select_gpu') as guard,\
         patch.object(s.time,'sleep',side_effect=AssertionError('target mismatch must fail before waiting')):
        with pytest.raises(ValueError,match='target'):s.main()
    guard.assert_not_called();assert path.read_bytes()==original


@pytest.mark.parametrize('artifact',['q1.json','g1_pool.yaml','.uncommitted.tmp','prior-directory'])
def test_probe_missing_ledger_with_any_prior_artifact_fails_closed(tmp_path,artifact):
    import scripts.probe_optimized as s
    argv,out=probe_args(tmp_path)
    prior=out/artifact
    if artifact=='prior-directory':prior.mkdir()
    else:prior.write_text('original evidence')
    with patch.object(sys,'argv',argv),patch('morphmimic.scripts.gpu_guard.select_gpu') as guard,\
         patch.object(s,'budget_remaining',return_value=0):
        with pytest.raises(ValueError,match='[Ll]edger.*missing|[Mm]issing.*ledger'):s.main()
    guard.assert_not_called()
    assert not (out/'budget.json').exists()
    assert prior.is_dir() or prior.read_text()=='original evidence'


@pytest.mark.parametrize('state',['new','empty','lock-only'])
def test_probe_fresh_output_can_create_initial_ledger(tmp_path,state):
    import scripts.probe_optimized as s
    argv,out=probe_args(tmp_path)
    if state=='new':out.rmdir()
    elif state=='lock-only':(out/'worker.lock').touch()
    with patch.object(sys,'argv',argv),patch('morphmimic.scripts.gpu_guard.select_gpu'),\
         patch.object(s,'budget_remaining',return_value=0),\
         patch.object(s.time,'sleep',side_effect=AssertionError('final/no-budget must exit')):
        s.main()
    ledger=json.loads((out/'budget.json').read_text())
    assert ledger['used_seconds']==0 and ledger['training_target']==5000
