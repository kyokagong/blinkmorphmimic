import numpy as np
import pytest
from morphmimic.standing import arm_geometry, native_arm_rmse, load_standing_target


def pose():
    p=np.zeros((14,3));p[0,2]=1.
    for s,e,w in ((8,9,10),(11,12,13)):
        p[s]=[0,0,1.5];p[e]=[0,0,1.2];p[w]=[0,0,.9]
    return p


def test_physical_elbow_bend_uses_bone_vectors_and_both_arms():
    p=pose();m=arm_geometry(p)
    assert m['elbow_bend_deg']==pytest.approx(0.)
    assert m['upper_arm_deg']==pytest.approx(0.)
    assert m['shoulder_hand_deg']==pytest.approx(0.)
    p[13]=p[12]+[.3,0,0]
    assert arm_geometry(p)['elbow_bend_deg']==pytest.approx(90.)
    p[9]=p[8]
    assert not np.isfinite(arm_geometry(p)['elbow_bend_deg'])


def test_geometry_numpy_and_torch_batches_agree():
    import torch
    p=np.stack((pose(),pose()))
    for key,value in arm_geometry(p).items():
        np.testing.assert_allclose(value,arm_geometry(torch.tensor(p))[key].numpy())


def test_arm_rmse_matches_native_names_instead_of_canonical_order():
    target=dict(joint_names=['l_elbow_pitch_joint','waist_yaw_joint','r_arm_roll_joint'],joint_pos=np.array([.3,9.,-.2]))
    assert native_arm_rmse(np.array([8.,-.2,.3]),['waist_yaw_joint','r_arm_roll_joint','l_elbow_pitch_joint'],target)==pytest.approx(0.)
    with pytest.raises(ValueError,match='joint'):
        native_arm_rmse(np.zeros(2),['waist_yaw_joint','r_arm_roll_joint'],target)


def test_target_receipt_binds_frame_and_native_source_hash(tmp_path):
    import hashlib,json
    source=tmp_path/'pose.npz';np.savez(source,fps=50,body_pos_w=pose()[None],joint_pos=np.zeros((1,2)),joint_names=['left_elbow_joint','right_elbow_joint'],body_names=[str(i) for i in range(14)])
    receipt=tmp_path/'target.json';receipt.write_text(json.dumps(dict(prepared_frame_zero_based=0,prepared_fps=50,robot_sources={'q1':dict(npz=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())})))
    target=load_standing_target(receipt,'q1')
    assert target['joint_pos'].shape==(2,)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        load_standing_target(receipt,'q1')
