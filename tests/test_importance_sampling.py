"""Online reset sampling follows observations without category priors."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def helpers():
    path = Path(__file__).parents[1] / 'src/morphmimic/train/importance_sampling.py'
    assert path.exists(), 'online importance sampling is missing'
    spec = importlib.util.spec_from_file_location('importance_sampling', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def library():
    return SimpleNamespace(num_bins=5, num_clips=2, clip_names=['walk', 'getup'],
        bin_clip=torch.tensor([0, 0, 0, 1, 1]), bin_start=torch.tensor([0, 50, 100, 151, 201]),
        bin_len=torch.tensor([50, 50, 51, 50, 51]), importance_clip_weights=torch.tensor([.5,.5]),
        importance_group_names=('fixture',), importance_clip_groups=torch.zeros(2,dtype=torch.long))


def test_clip_balanced_floor_cap_and_monotonic_difficulty():
    h = helpers(); state = h.create_state(library())
    base = torch.tensor([1/6, 1/6, 1/6, 1/4, 1/4])
    torch.testing.assert_close(state['probabilities'], base)
    state['failure_ema'].copy_(torch.tensor([0., .3, 1., 0., 0.]))
    state['error_ema'].zero_()
    h.refresh_probabilities(state)
    p = state['probabilities']
    assert p[0] < p[1] < p[2]
    assert bool((p >= .3 * base).all())
    assert bool((p <= 8 * base).all())
    torch.testing.assert_close(p.sum(), torch.tensor(1.))
    old = p[2].clone()
    h.record_outcomes(state, torch.tensor([2]*200), torch.zeros(200,dtype=torch.bool), torch.ones(200,dtype=torch.bool))
    h.refresh_probabilities(state)
    assert state['probabilities'][2] < old


def test_duplicate_bins_masked_updates_equal_valid_subset_and_unseen_unchanged():
    h = helpers(); a = h.create_state(library()); b = copy.deepcopy(a)
    bins = torch.tensor([0, 0, 0, 3, 4]); valid = torch.tensor([True, False, True, True, False])
    failure = torch.tensor([1., 1., 0., 0., 1.]); error = torch.tensor([.2, 1., .8, .1, 1.])
    h.update_statistics(a, bins, error, valid)
    h.record_outcomes(a, bins, failure.bool(), valid)
    h.update_statistics(b, bins[valid], error[valid], torch.ones(3,dtype=torch.bool))
    h.record_outcomes(b, bins[valid], failure[valid].bool(), torch.ones(3,dtype=torch.bool))
    for name in a: torch.testing.assert_close(a[name], b[name])
    torch.testing.assert_close(a['visits'], torch.tensor([2, 0, 0, 1, 0]))
    torch.testing.assert_close(a['failure_ema'][4], torch.tensor(.5))
    assert a['error_ema'][3] < .5


def test_extreme_priority_is_capped_and_every_clip_keeps_coverage():
    h = helpers()
    lib = SimpleNamespace(num_bins=100, num_clips=2, bin_clip=torch.tensor([0]*99+[1]), bin_start=torch.arange(100),importance_clip_weights=torch.tensor([.5,.5]),
        importance_group_names=('fixture',),importance_clip_groups=torch.zeros(2,dtype=torch.long))
    state = h.create_state(lib)
    state['failure_ema'].zero_(); state['error_ema'].zero_()
    state['failure_ema'][0]=1; state['error_ema'][0]=1
    h.refresh_probabilities(state)
    assert state['probabilities'][0] <= 8 * state['base'][0]
    assert state['probabilities'][-1] >= .15
    torch.testing.assert_close(state['probabilities'].sum(), torch.tensor(1.))


def test_resume_validation_is_atomic_and_roundtrips():
    h = helpers(); state = h.create_state(library()); saved = h.export_state(state)
    state['visits'][0] = 12
    h.load_state(state, saved)
    assert state['visits'][0] == 0
    before = copy.deepcopy(state)
    bad = copy.deepcopy(saved); bad['failure_ema'][0] = float('nan'); bad['visits'].fill_(7)
    with pytest.raises(ValueError): h.load_state(state, bad)
    for name in state: torch.testing.assert_close(state[name], before[name])
    bad = copy.deepcopy(saved); bad['bin_start'][1] += 1
    with pytest.raises(ValueError): h.load_state(state, bad)
    bad = copy.deepcopy(saved); bad['calibration'][1] = .8
    with pytest.raises(ValueError): h.validate_state(state, bad)
    for name in state: torch.testing.assert_close(state[name], before[name])


def test_real_finished_rollout_failures_match_eager_and_masked_resets():
    from morphmimic.env.commands import MultiMotionCommand
    from morphmimic.env.masked_reset import reset_command_bookkeeping_masked
    h = helpers()
    lib = library()
    lib.sample_frames_in_bins = lambda bins: lib.bin_start[bins]
    lib.frame_bin = lambda frames: torch.searchsorted(lib.bin_start, frames, right=True)-1
    lib.clip_end = lambda frames: torch.full_like(frames, 251)
    lib.remaining = lambda frames: 251-frames
    cfg = SimpleNamespace(sampling='return', failure_reset=1., resampling_time_range=(10.,10.), gamma=.99)
    manager = SimpleNamespace(terminated=torch.tensor([True, False, True, False]),
                              time_outs=torch.tensor([False, True, False, False]))
    def command():
        c = MultiMotionCommand.__new__(MultiMotionCommand)
        c._env = SimpleNamespace(num_envs=4, device='cpu', termination_manager=manager, max_episode_length=100)
        c.cfg=cfg; c.library=lib; c.importance_state=h.create_state(lib)
        c.global_frame=torch.tensor([10,20,30,40]); c.start_bin=torch.tensor([0,0,3,4])
        c.episode_steps=torch.tensor([8,12,0,20]); c.disc_return=torch.zeros(4)
        c.gamma_pow=torch.ones(4); c.horizon_pow=torch.ones(4); c.time_left=torch.ones(4)
        c.command_counter=torch.zeros(4,dtype=torch.long); c.consecutive_local=torch.zeros(4,dtype=torch.long)
        c.metrics={'sampling_entropy':torch.zeros(4)}; c._resample=lambda ids: None
        return c
    for mask in (torch.tensor([False]*4), torch.tensor([True,False,True,False]), torch.tensor([True]*4)):
        eager=command(); masked=command()
        eager.reset(mask.nonzero().flatten())
        reset_command_bookkeeping_masked(masked,mask)
        for name in eager.importance_state:
            torch.testing.assert_close(eager.importance_state[name],masked.importance_state[name])
        expected = 2 if bool(mask[1]) else int(mask[0])
        assert int(eager.importance_state['episodes'].sum()) == expected
        assert int(eager.importance_state['failures'].sum()) == int(mask[0])
        assert int(eager.importance_state['visits'].sum()) == 0


def test_cached_sampling_bypasses_probe_recovery_and_failure_restart_in_both_paths():
    from morphmimic.env.commands import MultiMotionCommand
    from morphmimic.env.masked_reset import sample_start_candidates
    h=helpers(); lib=library()
    lib.sample_frames_in_bins=lambda bins: lib.bin_start[bins]
    lib.frame_bin=lambda frames: torch.searchsorted(lib.bin_start,frames,right=True)-1
    lib.clip_end=lambda frames: torch.full_like(frames,251)
    state=h.create_state(lib)
    state['probabilities'].zero_(); state['probabilities'][2]=1
    c=SimpleNamespace(num_envs=12,device='cpu',library=lib,importance_state=state,
        cfg=SimpleNamespace(sampling='return',failure_reset=1.), recovery={'invalid':True},
        probe_probabilities=torch.tensor([1.,0,0,0,0]),probe_rho=1.,
        global_frame=torch.zeros(12,dtype=torch.long),restart_frame=torch.zeros(12,dtype=torch.long),
        consecutive_local=torch.zeros(12,dtype=torch.long),metrics={'sampling_entropy':torch.zeros(12)})
    eager=MultiMotionCommand._sample_start(c,torch.arange(12))
    masked=sample_start_candidates(c)
    torch.testing.assert_close(eager[1],torch.full((12,),2))
    torch.testing.assert_close(masked[1],eager[1])
    torch.testing.assert_close(masked[0],eager[0])


def test_tracking_error_ignores_unavailable_canonical_joints_and_counts_native_rmse():
    h=helpers(); z=torch.zeros(2,3); q=torch.tensor([[1.,0,0,0]]).repeat(2,1)
    c=SimpleNamespace(library=SimpleNamespace(root_pos_w=z,root_quat_w=q),global_frame=torch.arange(2),
        robot=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=z[:,None,:],body_link_quat_w=q[:,None,:])),
        spec=SimpleNamespace(height_scale=1.),_env=SimpleNamespace(scene=SimpleNamespace(env_origins=z)),
        _importance_root_index=0,_importance_joint_mask=torch.tensor([1.,0.]),
        joint_pos=torch.tensor([[.5,1000.],[.5,0.]]),robot_joint_pos=torch.zeros(2,2),
        metrics={name:torch.zeros(2) for name in ('error_anchor_pos','error_anchor_rot','error_body_pos','error_body_rot')})
    actual=h.tracking_error(c)
    torch.testing.assert_close(actual,torch.full((2,),1-torch.exp(torch.tensor(-1/7))))
    assert bool(((actual>=0)&(actual<=1)).all())


def test_report_labels_unobserved_outcomes_without_inventing_success():
    import json
    h=helpers(); lib=library(); state=h.create_state(lib)
    report=h.report(state,lib)
    assert report['clips'][0]['rollout_failure_rate'] is None
    assert all(row['rollout_failure_rate'] is None for row in report['bins'])
    assert report['unseen_bins'] == 5
    h.record_outcomes(state,torch.tensor([0,0,0]),torch.tensor([True,False,True]),torch.tensor([True,True,False]))
    report=h.report(state,lib)
    assert report['clips'][0]['rollout_failure_rate'] == .5
    assert report['bins'][0]['episodes'] == 2
    json.dumps(report,allow_nan=False)


def test_source_balance_prevents_9934_seed_clips_from_drowning_four_flips():
    h=helpers(); sizes={'seed':9934,'seed_flips':4,'lafan1':38,'motiondecode':396,'neutral_recovery':25}
    rows=[dict(robot='q1',split='train',name=f'{group}_{i}',dataset=group) for group,n in sizes.items() for i in range(n)]
    names=[row['name'] for row in rows]
    weights,groups,clip_groups=h.source_balanced_clip_weights(names,rows,'q1')
    lib=SimpleNamespace(num_clips=len(names),clip_names=names,bin_clip=torch.arange(len(names)),bin_start=torch.arange(len(names))*50,
        importance_clip_weights=weights,importance_group_names=groups,importance_clip_groups=clip_groups)
    state=h.create_state(lib)
    for index, group in enumerate(groups):
        assert float(state['base'][clip_groups==index].sum())==pytest.approx(.2,abs=1e-6)
    flip=clip_groups==groups.index('seed_flips')
    state['failure_ema'][flip]=1;state['error_ema'][flip]=1
    h.refresh_probabilities(state)
    assert float(state['probabilities'][flip].sum())>.2
    assert float(state['probabilities'][~flip].sum())<.8
    report=h.report(state,lib)
    actual={row['group']:row for row in report['groups']}
    assert actual['seed']['base_mass']==pytest.approx(.2,abs=1e-6)
    assert actual['seed_flips']['probability_mass']>.2
    assert state['calibration'][0]==2


def test_source_weighting_is_dataset_metadata_only_and_validates_inventory():
    h=helpers();rows=[dict(robot='q1',split='train',name='seed',dataset='seed'),dict(robot='q1',split='train',name='getup',dataset='seed')]
    weights,groups,ids=h.source_balanced_clip_weights(['getup','seed'],rows,'q1')
    torch.testing.assert_close(weights,torch.tensor([.5,.5]));assert groups==('seed',)
    with pytest.raises(ValueError):h.source_balanced_clip_weights(['missing'],rows,'q1')
    with pytest.raises(ValueError):h.source_balanced_clip_weights(['seed'],rows+[rows[0]],'q1')
    with pytest.raises(ValueError):h.create_state(SimpleNamespace(num_clips=2,bin_clip=torch.tensor([0,1]),bin_start=torch.tensor([0,50])))


def test_version_one_or_different_source_base_cannot_resume_version_two():
    h=helpers();lib=library();state=h.create_state(lib);saved=h.export_state(state)
    saved['calibration'][0]=1
    with pytest.raises(ValueError):h.validate_state(state,saved)
    saved=h.export_state(state);saved['base'][0]*=1.01
    with pytest.raises(ValueError):h.validate_state(state,saved)


@pytest.mark.parametrize('signal',['failure','error'])
def test_each_observed_difficulty_signal_independently_increases_probability(signal):
    h=helpers();state=h.create_state(library())
    before=state['probabilities'][0].clone()
    if signal=='failure':
        h.record_outcomes(state,torch.tensor([0]*10),torch.ones(10,dtype=torch.bool),torch.ones(10,dtype=torch.bool))
    else:
        h.update_statistics(state,torch.tensor([0]*10),torch.ones(10),torch.ones(10,dtype=torch.bool))
    h.refresh_probabilities(state)
    assert state['probabilities'][0]>before
    assert bool((state['probabilities'] >= .3*state['base']).all())
    assert bool((state['probabilities'] <= 8*state['base']).all())


def test_source_group_metadata_is_pinned_and_manual_boost_is_rejected():
    h=helpers();lib=library();state=h.create_state(lib);saved=h.export_state(state)
    saved['clip_group'][0]=1
    with pytest.raises(ValueError):h.validate_state(state,saved)
    lib.importance_clip_weights=torch.tensor([.8,.2])
    with pytest.raises(ValueError):h.create_state(lib)


def test_report_describes_actual_masked_collector_and_pooled_interface_statistics():
    h=helpers();lib=library();report=h.report(h.create_state(lib),lib)
    assert 'teacher' not in report['failure_definition'].lower()
    assert 'teacher' not in report['policy'].lower()
    assert 'actual collector policy' in report['failure_definition'].lower()
    assert 'trainer masks/configuration' in report['policy']
    assert 'pooled across environments' in report['policy']
    assert 'not separated by interface' in report['policy']
