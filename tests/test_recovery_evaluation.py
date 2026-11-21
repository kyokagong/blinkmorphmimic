import copy
import json
from pathlib import Path
import pytest
from scripts.evaluate_recovery import StandingTracker, aggregate, file_hash, validate_windows, verify_hashes


def sample(height=.9, up=1., speed=0., angular_speed=0., pose_error=.1):
    return dict(height=height, up=up, speed=speed, angular_speed=angular_speed, pose_error=pose_error)


def test_success_requires_a_continuous_one_second_actual_stand():
    tracker = StandingTracker(1., .02)
    tracker.initial(sample(height=.4, up=0.))
    for _ in range(49):
        assert tracker.update(sample()) is None
    tracker.update(sample(speed=.5))
    for _ in range(49):
        assert tracker.update(sample()) is None
    assert tracker.update(sample()) == 'success'
    assert tracker.result()['time_to_stand_s'] == pytest.approx(1.02)
    assert tracker.result()['stand_confirmed_s'] == pytest.approx(2.)


def test_survival_reward_and_termination_do_not_imply_recovery():
    tracker = StandingTracker(1., .02)
    tracker.initial(sample(height=.3, up=0.))
    for _ in range(500):
        assert tracker.update(sample(height=.3)) is None
    assert not tracker.result()['success']
    tracker = StandingTracker(1., .02)
    tracker.initial(sample(height=.3))
    for _ in range(49): tracker.update(sample())
    assert tracker.update(sample(), terminated=True) == 'termination'
    assert not tracker.result()['success']
    tracker = StandingTracker(1., .02)
    assert tracker.initial(sample()) == 'initial_pose_already_standing'


@pytest.mark.parametrize('key', ['height','up','speed','angular_speed','pose_error'])
def test_nonfinite_physics_cannot_succeed(key):
    tracker = StandingTracker(1., .02)
    tracker.initial(sample(height=.3))
    bad=sample();bad[key]=float('nan')
    assert tracker.update(bad) == 'nonfinite_state'
    assert not tracker.result()['success']


@pytest.mark.parametrize('height,dt', [(True,.02),(0.,.02),(float('inf'),.02),(1.,0.),(1.,True)])
def test_standing_reference_and_timestep_validation(height,dt):
    with pytest.raises(ValueError): StandingTracker(height,dt)


def document():
    return dict(schema_version=1,fps=50,training_manifest_sha256='a'*64,subset_sha256={'q1':'b'*64},windows=[
        dict(robot='q1',clip='getup',source_id='source',category='lying',start_frame=2,stop_frame=20,stand_frame=50,hold_end_frame=105)])


def test_window_bounds_types_and_category_validation():
    doc=document(); assert validate_windows(doc,'q1',{'getup':120})[0]['start_frame']==2
    for field,value in [('start_frame',True),('stop_frame',121),('stand_frame',1),('hold_end_frame',51),('category','walking')]:
        bad=copy.deepcopy(doc);bad['windows'][0][field]=value
        with pytest.raises(ValueError): validate_windows(bad,'q1',{'getup':120})
    bad=copy.deepcopy(doc);bad['windows']*=2
    with pytest.raises(ValueError): validate_windows(bad,'q1',{'getup':120})



def test_reference_standing_hold_must_finish_within_ten_seconds():
    doc=document()
    row=doc['windows'][0]
    row.update(start_frame=2,stop_frame=3,stand_frame=452,hold_end_frame=502)
    assert validate_windows(doc,'q1',{'getup':600})[0]['hold_end_frame']==502
    row.update(stand_frame=453,hold_end_frame=503)
    with pytest.raises(ValueError,match='10 seconds'):
        validate_windows(doc,'q1',{'getup':600})


def test_category_aggregation_preserves_missing_categories_and_failures():
    rows=[dict(category='lying',success=True,time_to_stand_s=2.,failure_reason=None,final_root_height_ratio=.9,final_root_up=1.,mean_pose_error_rad=.1),
          dict(category='lying',success=False,time_to_stand_s=None,failure_reason='termination',final_root_height_ratio=.4,final_root_up=.1,mean_pose_error_rad=.5)]
    result=aggregate(rows)
    assert result['lying']['success_rate']==.5
    assert result['lying']['failures']=={'termination':1}
    assert result['lying']['mean_time_to_stand_s']==2.
    assert result['seated']['cases']==0
    assert result['seated']['success_rate'] is None


def test_sha_verification_detects_changed_input(tmp_path):
    path=tmp_path/'input';path.write_text('immutable')
    proof={str(path):file_hash(path)}
    verify_hashes(proof)
    path.write_text('changed')
    with pytest.raises(ValueError,match='SHA'): verify_hashes(proof)


def test_native_rollout_uses_explicit_starts_and_resets_finished_worlds():
    import torch
    from types import SimpleNamespace
    from morphmimic.constants import PROPRIO_DIM
    from scripts.evaluate_recovery import run_recovery
    n=2
    lib=SimpleNamespace(clip_names=['a','b','c'],clip_start=torch.tensor([0,200,400]),
        joint_pos=torch.zeros(600,2),has_human=False)
    command=SimpleNamespace(library=lib,cfg=SimpleNamespace(sampling='start'),
        global_frame=torch.zeros(n,dtype=torch.long),robot_body_pos_w=torch.zeros(n,1,3),
        robot_body_quat_w=torch.tensor([[[1.,0.,0.,0.]]]*n),robot_body_lin_vel_w=torch.zeros(n,1,3),
        robot_body_ang_vel_w=torch.zeros(n,1,3),robot_joint_pos=torch.zeros(n,2))
    clips=torch.zeros(n,dtype=torch.long);starts=torch.zeros(n,dtype=torch.long)
    pending=torch.zeros(n,dtype=torch.bool);assigned=[]
    def assign(ids,clip_ids,start_frames):
        clips[ids]=clip_ids;starts[ids]=start_frames;assigned.extend(start_frames.tolist())
    command.assign_clips=assign
    obs={'policy':torch.zeros(n,PROPRIO_DIM),'goal':torch.zeros(n,1),'goal_kp':torch.zeros(n,1)}
    def reset(env_ids):
        pending[env_ids]=False
        command.global_frame[env_ids]=lib.clip_start[clips[env_ids]]+starts[env_ids]+1
        command.robot_body_pos_w[env_ids,0,2]=.3
        return obs,{}
    def step(action):
        assert not pending.any(), 'finished worlds must be reset before the next native step'
        local=command.global_frame-lib.clip_start[clips]
        terminated=(clips==1)&(local==5)
        command.global_frame+=1;command.robot_body_pos_w[:,0,2]=.9
        pending.copy_(terminated)
        return obs,torch.ones(n)*1000,terminated,torch.zeros(n,dtype=torch.bool),{}
    env=SimpleNamespace(num_envs=n,device='cpu',step_dt=.02,cfg=SimpleNamespace(auto_reset=False),
        scene=SimpleNamespace(env_origins=torch.zeros(n,3)),command_manager=SimpleNamespace(get_term=lambda _:command),
        physics_diagnostics=SimpleNamespace(report=lambda:dict(overflow_bits=0)),reset=reset,step=step)
    cases=[dict(clip=clip,category=category,start_frame=start,stop_frame=100,
        reference_standing_height_m=1.) for clip,category,start in zip(lib.clip_names,['lying','seated','half_squat'],[2,3,4])]
    rows=run_recovery(env,lambda history,goals:torch.zeros(n,2),1,cases)
    assert [row['success'] for row in rows]==[True,False,True]
    assert rows[1]['failure_reason']=='termination'
    assert assigned[:2]==[2,3]
    assert 4 in assigned


def test_checkpoint_ancestry_requires_muon_and_the_exact_finetune_parent():
    from scripts.evaluate_recovery import validate_checkpoint_ancestry
    parent=dict(iteration=5000,sha256='a'*64)
    candidate=dict(iteration=5250,sha256='b'*64)
    parent_saved=dict(cfg=dict(optimizer='muon'))
    saved=dict(cfg=dict(optimizer='muon'),finetuned_from=dict(sha256=parent['sha256']))
    validate_checkpoint_ancestry(candidate,saved,parent,parent_saved)
    validate_checkpoint_ancestry(parent,parent_saved,parent,parent_saved)
    for wrong_saved in (dict(cfg=dict(optimizer='muon')),
                        dict(cfg=dict(optimizer='muon'),finetuned_from=dict(sha256='c'*64))):
        with pytest.raises(ValueError,match='parent SHA'):
            validate_checkpoint_ancestry(candidate,wrong_saved,parent,parent_saved)
    with pytest.raises(ValueError,match='parent SHA'):
        validate_checkpoint_ancestry(parent,parent_saved,dict(iteration=5000,sha256='c'*64),parent_saved)
    for candidate_cfg,parent_cfg in [('adam','muon'),('muon','adam')]:
        with pytest.raises(ValueError,match='Muon'):
            validate_checkpoint_ancestry(candidate,dict(saved,cfg=dict(optimizer=candidate_cfg)),
                parent,dict(cfg=dict(optimizer=parent_cfg)))


def neutral_sample(**changes):
    return dict(sample(),upper_arm_deg=5.,shoulder_hand_deg=10.,elbow_bend_deg=20.,
                wrist_root_height=.04,arm_joint_rmse=.1,**changes)


def test_neutral_standing_requires_three_seconds_and_both_arms_down():
    criteria=dict(hold_seconds=3.,timeout_seconds=15.,upper_arm_deg_max=20.,
        shoulder_hand_deg_max=25.,elbow_bend_deg_max=35.,wrist_root_height_max=.06,arm_joint_rmse_max=.25)
    tracker=StandingTracker(1.,.02,criteria=criteria)
    tracker.initial(sample(height=.3))
    for key in ('upper_arm_deg','shoulder_hand_deg','elbow_bend_deg','wrist_root_height','arm_joint_rmse'):
        bad=neutral_sample();bad[key]=100.
        for _ in range(160): assert tracker.update(bad) is None
    tracker=StandingTracker(1.,.02,criteria=criteria);tracker.initial(sample(height=.3))
    for _ in range(149): assert tracker.update(neutral_sample()) is None
    assert tracker.update(neutral_sample())=='success'
    assert tracker.result()['final_elbow_bend_deg']==20.


def test_extended_neutral_windows_require_explicit_limit_and_three_second_hold():
    doc=document();doc['max_recovery_seconds']=15
    doc['windows'][0].update(stand_frame=550,hold_end_frame=700)
    assert validate_windows(doc,'q1',{'getup':750},hold_seconds=3.)
    doc['windows'][0]['hold_end_frame']=699
    with pytest.raises(ValueError,match='standing hold'):
        validate_windows(doc,'q1',{'getup':750},hold_seconds=3.)


def test_dataset_transition_binds_both_manifests_and_saved_receipt(tmp_path):
    from scripts.evaluate_recovery import validate_dataset_transition
    parent_manifest=tmp_path/'old.json';parent_manifest.write_text('{}')
    manifest=tmp_path/'new.json';manifest.write_text('{"complete":true}')
    receipt=tmp_path/'transition.json'
    receipt.write_text(json.dumps(dict(schema_version=1,parent_manifest=dict(path=str(parent_manifest),sha256=file_hash(parent_manifest)),
        manifest=dict(path=str(manifest),sha256=file_hash(manifest)))))
    saved=dict(dataset_transition=dict(path=str(receipt),sha256=file_hash(receipt)))
    proof={}
    assert validate_dataset_transition(receipt,saved,file_hash(manifest),proof)==file_hash(parent_manifest)
    assert proof[str(receipt)]==file_hash(receipt)
    saved['dataset_transition']['sha256']='0'*64
    with pytest.raises(ValueError,match='transition'):
        validate_dataset_transition(receipt,saved,file_hash(manifest),{})


def test_neutral_standing_rejects_squat_lean_and_moving_limbs():
    criteria=dict(whole_body_height_min=.95,torso_up_min=.9,body_speed_max=.5,hold_seconds=3.)
    for invalid in (dict(whole_body_height=.8),dict(torso_up=.6),dict(body_speed=1.)):
        tracker=StandingTracker(1.,.02,criteria=criteria);tracker.initial(sample(height=.3))
        value=dict(neutral_sample(),whole_body_height=1.,torso_up=1.,body_speed=.1);value.update(invalid)
        for _ in range(150):assert tracker.update(value) is None
        assert not tracker.result()['success']


def test_explicit_6000_parent_requires_exact_new_run_ancestry():
    from scripts.evaluate_recovery import validate_checkpoint_ancestry
    parent=dict(iteration=6000,sha256='a'*64)
    saved=dict(cfg=dict(optimizer='muon'),finetuned_from=dict(sha256='a'*64))
    validate_checkpoint_ancestry(dict(iteration=6250,sha256='b'*64),saved,parent,saved,parent_iteration=6000)
    validate_checkpoint_ancestry(parent,saved,parent,saved,parent_iteration=6000)
    with pytest.raises(ValueError,match='parent SHA'):
        validate_checkpoint_ancestry(dict(iteration=6250,sha256='b'*64),dict(saved,finetuned_from={}),parent,saved,parent_iteration=6000)
    with pytest.raises(ValueError,match='6000'):
        validate_checkpoint_ancestry(dict(iteration=6250,sha256='b'*64),saved,dict(parent,iteration=5000),saved,parent_iteration=6000)


def test_transition_baseline_is_explicit_parent_iteration(tmp_path):
    from scripts.evaluate_recovery import validate_dataset_transition
    before=tmp_path/'old.json';before.write_text('{}')
    after=tmp_path/'new.json';after.write_text('{"complete":true}')
    receipt=tmp_path/'transition.json'
    receipt.write_text(json.dumps(dict(schema_version=1,parent_manifest=dict(path=str(before),sha256=file_hash(before)),
        manifest=dict(path=str(after),sha256=file_hash(after)))))
    saved=dict(iteration=6000,optimized_training=dict(prepared_manifest_sha256=file_hash(before)))
    assert validate_dataset_transition(receipt,saved,file_hash(after),{},parent_iteration=6000)==file_hash(before)
    with pytest.raises(ValueError,match='transition receipt'):
        validate_dataset_transition(receipt,saved,file_hash(after),{})
    with pytest.raises(ValueError,match='parent manifest'):
        validate_dataset_transition(receipt,dict(saved,optimized_training={}),file_hash(after),{},parent_iteration=6000)
