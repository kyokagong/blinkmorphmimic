"""Delivery audit rejects fabricated progress and keeps missing checkpoints missing."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def api():
    path=Path(__file__).parents[1]/'scripts/audit_importance.py'
    assert path.exists(), 'importance delivery audit is missing'
    spec=importlib.util.spec_from_file_location('audit_importance',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture():
    def adam(step):
        return dict(param_groups=[dict(params=[0])],state={0:dict(step=torch.tensor(float(step)),exp_avg=torch.ones(1),exp_avg_sq=torch.ones(1))})
    def optim(step):
        return {key:(dict(muon=dict(param_groups=[dict(params=[0])],state={0:dict(momentum_buffer=torch.ones(1,1))}),adam=adam(step)) if key in ('wm','policy','kp') else adam(step*20)) for key in ('wm','policy','kp','critic','critic_kp')}
    ancestor=dict(iteration=5000,path='/old/q1/ckpt_5000.pt',sha256='b'*64)
    def checkpoint(iteration,n):
        delta=iteration-6000
        return dict(iteration=iteration,cfg=dict(robot='q1',wm_batches=1,num_envs=2048,iterations=7000,optimizer='muon',importance_sampling=iteration>6000,recovery_reset_share=0.,failure_reset=0.,probe_focus_dir=None,kp_interface=True,policy_goal='motion'),
            optimized_training=dict(optimizer_steps=dict(world_model=5000+delta,motion=5000+delta,human=5000+delta)),
            optimization_state=dict(optimizer_steps=dict(world_model=5000+delta,motion=5000+delta,human=5000+delta)),
            physics=dict(overflow_bits=0,checked_physics_steps=100),optim=optim(5000+delta),ema_updates=5000+delta,ema_policy={'weight':torch.tensor([float(iteration)])},
            coverage=dict(clip_names=['old','flip'][:n],counts={key:torch.tensor([10+delta,delta][:n]) for key in ('collected','world_model','policy')},
                source=dict(since_iteration=5000,start=[0]*n,stop=[99]*n,has_phases=[False]*n,counts={key:[8+delta,delta][:n] for key in ('collected','world_model','policy')})),
            human_coverage=dict(clip_names=['old','flip'][:n],since_iteration=5000,counts=torch.tensor([10+delta,delta][:n]),source_counts=torch.tensor([8+delta,delta][:n])))
    parent=checkpoint(6000,1);parent['finetuned_from']=ancestor
    current=checkpoint(6250,2)
    current['finetuned_from']=dict(iteration=6000,path='/parent/q1/ckpt_6000.pt',sha256='a'*64,ancestor=ancestor,
        optimizer_state_restored=True,ema_state_restored=True)
    from morphmimic.train.importance_sampling import create_state,refresh_probabilities,source_balanced_clip_weights
    manifest={'clips':[dict(robot='q1',name=name,dataset=dataset,source_id=dataset+'/'+name,frames=100,human_available=True,split='train') for name,dataset in [('old','lafan1'),('flip','seed_flips')]]}
    weights,groups,ids=source_balanced_clip_weights(['old','flip'],manifest['clips'],'q1')
    state=create_state(SimpleNamespace(num_clips=2,bin_clip=torch.tensor([0,0,1,1]),bin_start=torch.tensor([0,50,100,150]),
        importance_clip_weights=weights,importance_group_names=groups,importance_clip_groups=ids))
    state['failure_ema'].copy_(torch.tensor([.8,.2,.7,.1]));state['visits'].fill_(40);state['episodes'].fill_(4);state['failures'].fill_(2)
    refresh_probabilities(state);current['importance_sampling']=dict(since_iteration=6000,state=state)
    manifest={'clips':[dict(robot='q1',name=name,dataset=dataset,source_id=dataset+'/'+name,frames=100,human_available=True,split='train') for name,dataset in [('old','lafan1'),('flip','seed_flips')]]}
    return parent,current,manifest


def check(module,parent,current,manifest,previous=None):
    return module.audit_snapshot(current,parent,previous,manifest,'q1',6250,'/parent/q1/ckpt_6000.pt','a'*64)


def test_reports_real_dataset_and_source_deltas_without_claiming_fullclip_success():
    m=api();parent,current,manifest=fixture();report=check(m,parent,current,manifest)
    assert report['sampler_semantics_version']==2
    assert all(r['base_mass']==pytest.approx(.5) for r in report['datasets'])
    assert report['optimizer_steps']['motion']==5250
    assert report['ema_updates']==5250
    assert report['probability_l1_from_base']>0
    assert {r['dataset'] for r in report['datasets']}=={'lafan1','seed_flips'}
    assert all(r['delta']['human']==250 for r in report['datasets'])
    assert all(r['source_delta']['human']==250 for r in report['datasets'])
    assert sum(r['probability_mass'] for r in report['datasets'])==pytest.approx(1)
    assert report['unseen_bins']==[]


@pytest.mark.parametrize('change', ['optimizer_reset','ema_reset','wrong_parent','coverage_rollback','human_epoch','fixed_bias','bad_sampler'])
def test_rejects_broken_continuation_contract(change):
    m=api();parent,current,manifest=fixture()
    if change=='optimizer_reset':current['optim']['kp']['adam']['state'][0]['step'].fill_(1)
    elif change=='ema_reset':current['ema_updates']=1
    elif change=='wrong_parent':current['finetuned_from']['sha256']='c'*64
    elif change=='coverage_rollback':current['coverage']['counts']['policy'][0]=0
    elif change=='human_epoch':current['human_coverage']['since_iteration']=6000
    elif change=='fixed_bias':current['cfg']['recovery_reset_share']=.5
    else:current['importance_sampling']['state']['probabilities'][0]=0
    with pytest.raises(ValueError):check(m,parent,current,manifest)


def test_missing_checkpoints_are_incomplete_not_a_pass(tmp_path):
    result=api().audit_run(tmp_path/'run',tmp_path/'parent')
    assert result['status']=='incomplete'
    assert len(result['missing'])==18
    assert result['checkpoints']==[]


def test_csv_reader_keeps_nonfinite_values_and_missing_tags_visible(tmp_path):
    path=tmp_path/'training-scalars.csv'
    path.write_text('robot,tag,iteration,value\nq1,wm/loss,6020,1.5\nq1,wm/loss,6040,nan\n')
    rows=api().read_training_csv(path)
    assert rows[0]['value']==1.5
    assert rows[1]['value'] is None and rows[1]['status']=='nonfinite'


def test_plot_outputs_keep_observed_values_and_do_not_fill_missing_checkpoints(tmp_path):
    import json
    m=api();parent,current,manifest=fixture()
    snapshot=check(m,parent,current,manifest)
    report=dict(status='incomplete',checkpoints=[snapshot],missing=['ckpt_7000.pt'])
    rows=[dict(robot='q1',tag='wm/loss',iteration=6020,value=1.25,status='ok')]
    out=tmp_path/'audit';m.write_outputs(report,out,rows)
    assert json.loads((out/'audit.json').read_text())['missing']==['ckpt_7000.pt']
    assert m.read_training_csv(out/'training-scalars.csv')[0]['value']==1.25
    assert (out/'probability_mass.png').exists()
    assert (out/'sampling_probability_l1_from_base.png').exists()
    assert {r['iteration'] for r in json.loads((out/'sampling-coverage.json').read_text())}=={6250}


def test_missing_source_usage_is_rejected_even_if_total_counts_increase():
    m=api();parent,current,manifest=fixture()
    current['human_coverage']['source_counts'][1]=0
    with pytest.raises(ValueError,match='missing actual'):
        check(m,parent,current,manifest)


def test_per_clip_gaps_do_not_hide_unused_flip_behind_dataset_totals():
    m=api()
    def row(name, dataset, counts, human=True):
        return dict(clip=name,dataset=dataset,human_available=human,
            delta=dict(zip(m.STAGES,counts)),source_delta=dict(zip(m.STAGES,counts)))
    records=[row('flip-used','seed_flips',[10,20,3,4]),
             row('flip-unused-wm','seed_flips',[8,0,0,2]),
             row('motion-only','motiondecode',[5,7,1,0],False)]
    result=m.clip_coverage_summary(records)
    assert result['zero_coverage']['world_model']==['flip-unused-wm']
    assert result['zero_coverage']['human']==[]
    assert result['human_not_applicable']==['motion-only']
    flips=result['flip_coverage']
    assert flips['train_clip_count']==2
    assert flips['all_stages_all_clips_used'] is False
    assert flips['zero_source_by_stage']['world_model']==['flip-unused-wm']
    assert len(flips['clips'])==2
    assert flips['all_clips_used_by_stage']['collected'] is True


def test_complete_flip_usage_and_source_mass_change_are_explicit():
    m=api();parent,current,manifest=fixture();report=check(m,parent,current,manifest)
    assert report['flip_coverage']['all_stages_all_clips_used'] is True
    assert report['flip_coverage']['train_clip_count']==1
    assert all(not rows for rows in report['zero_source_coverage'].values())
    datasets={row['dataset']:row for row in report['datasets']}
    assert datasets['lafan1']['probability_mass_change_from_base']>0
    assert datasets['seed_flips']['probability_mass_change_from_base']<0
    assert datasets['lafan1']['probability_mass_change']==pytest.approx(
        datasets['lafan1']['probability_mass']-datasets['lafan1']['previous_probability_mass'])


def test_exact_step_deltas_and_physics_are_part_of_delivery_proof():
    m=api();parent,current,manifest=fixture();report=check(m,parent,current,manifest)
    assert report['optimizer_step_deltas']=={'wm':250,'policy':250,'kp':250,'critic':5000,'critic_kp':5000}
    assert report['ema_update_delta']==250
    assert report['physics']['overflow_bits']==0


@pytest.mark.parametrize('bad',['extra_update','overflow'])
def test_rejects_wrong_exact_update_count_or_checkpoint_overflow(bad):
    m=api();parent,current,manifest=fixture()
    if bad=='extra_update':
        current['optimization_state']['optimizer_steps']['world_model']+=1
        current['optimized_training']['optimizer_steps']['world_model']+=1
        current['optim']['wm']['adam']['state'][0]['step']+=1
    else:current['physics']={'overflow_bits':8,'checked_physics_steps':100}
    with pytest.raises(ValueError):check(m,parent,current,manifest)


def test_training_health_uses_observed_nonfinite_and_overflow_counters():
    m=api();rows=[dict(robot='q1',tag='wm/nonfinite_steps',iteration=6020,value=0,status='ok'),
        dict(robot='q1',tag='policy/nonfinite_steps',iteration=6040,value=1,status='ok'),
        dict(robot='bumi',tag='physics/overflow_bits',iteration=6040,value=2,status='ok')]
    bad=m.invalid_training_observations(rows)
    assert [row['tag'] for row in bad]==['policy/nonfinite_steps','physics/overflow_bits']
