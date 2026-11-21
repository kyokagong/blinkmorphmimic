import math
import pytest
from scripts.evaluate_motion_suite import flip_metrics, pitch, select_rows


def trace(angles, *, contact=True):
    return [dict(pitch_rad=a,foot_floor_contacts=int(contact),root_up=1.,root_height_m=1.,
        root_speed_m_s=0.,root_angular_speed_rad_s=0.,joint_rmse_rad=.1,
        actual_native_torques_nm=[2.,-3.]) for a in angles]


def test_actual_rotation_flight_and_landing_are_all_required():
    rows=trace([0.]*25+[i*math.pi/25 for i in range(51)]+[2*math.pi]*25)
    for row in rows[25:76]:row['foot_floor_contacts']=0
    result=flip_metrics(rows,2*math.pi,1.,failed=False)
    assert result['successful_full_rotation_and_stable_landing']
    assert result['actual_stable_landing_seconds']==pytest.approx(.5)
    assert not flip_metrics(rows,2*math.pi,1.,failed=True)['successful_full_rotation_and_stable_landing']
    assert not flip_metrics(trace([0.]*100),2*math.pi,1.,failed=False)['actual_full_rotation']
    assert not flip_metrics(trace([i*math.pi/25 for i in range(51)]+[2*math.pi]*25),2*math.pi,1.,failed=False)['successful_full_rotation_and_stable_landing']
    rows[-1]['joint_rmse_rad']=2.
    assert not flip_metrics(rows,2*math.pi,1.,failed=False)['successful_full_rotation_and_stable_landing']


def test_reference_turn_is_not_counted_as_actual_and_rotation_direction_matters():
    rows=trace([0.]*100)
    assert flip_metrics(rows,2*math.pi,1.,failed=False)['failure_reason']=='incomplete_actual_rotation'
    rows=trace([-i*math.pi/25 for i in range(51)])
    assert not flip_metrics(rows,2*math.pi,1.,failed=False)['actual_full_rotation']
    assert flip_metrics(rows,-2*math.pi,1.,failed=False)['actual_full_rotation']
    assert pitch([math.cos(.2),0,math.sin(.2),0],[1,0,0,0])==pytest.approx(.4)


def test_suite_includes_both_train_and_heldout_flips():
    rows=[dict(robot='q1',name='seed_flip_'+s,dataset='seed_flips',split=s) for s in ('train','val')]
    assert {r['split'] for r in select_rows(dict(clips=rows),'q1','flips')}=={'train','val'}


def test_torso_floor_contact_is_not_airborne_even_if_feet_are_lifted():
    rows=trace([i*math.pi/25 for i in range(51)]+[2*math.pi]*25)
    for row in rows:
        row['foot_floor_contacts']=0
        row['robot_floor_contacts']=1
    assert flip_metrics(rows,2*math.pi,1.,failed=False)['actual_flight_seconds']==0.


def test_undefined_crossing_time_is_json_null_without_hiding_nonfinite_physics():
    import json
    import pandas as pd
    from scripts.evaluate_motion_suite import json_records
    rows=json_records(pd.DataFrame([dict(t_xy_over=float('nan'),t_yaw_over=2.,norm_return=.5)]))
    assert rows==[dict(t_xy_over=None,t_yaw_over=2.,norm_return=.5)]
    json.dumps(rows,allow_nan=False)
    with pytest.raises(ValueError):
        json.dumps(json_records(pd.DataFrame([dict(t_xy_over=float('nan'),norm_return=float('nan'))])),allow_nan=False)
