import numpy as np
import pytest
import torch

from morphmimic.constants import HUMAN_JOINTS
from morphmimic.data.human_convert import sidecar_path
from morphmimic.data.motion_library import MotionLibrary


def test_motion_only_clip_has_no_valid_human_targets(tmp_path):
    paths = [tmp_path/'q1'/f'{name}.npz' for name in ('paired','robot_only')]
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        quat = np.zeros((3,1,4),dtype=np.float32);quat[...,0]=1
        vec = np.zeros((3,1,3),dtype=np.float32)
        np.savez(path,fps=50,joint_pos=np.zeros((3,29),dtype=np.float32),
            joint_vel=np.zeros((3,29),dtype=np.float32),body_names=['root'],
            body_pos_w=vec,body_quat_w=quat,body_lin_vel_w=vec,body_ang_vel_w=vec)
    human_root=tmp_path/'human50'
    side=sidecar_path(paths[0],human_root);side.parent.mkdir(parents=True)
    quat=np.zeros((3,24,4),dtype=np.float32);quat[...,0]=1
    np.savez(side,fps=50,human_joint_names=HUMAN_JOINTS,
        human_pos_w=np.ones((3,24,3),dtype=np.float32),human_quat_w=quat)
    with pytest.raises(ValueError,match='sidecar'):
        MotionLibrary([str(p) for p in paths],body_names=('root',),human_root=str(human_root))
    lib=MotionLibrary([str(p) for p in paths],body_names=('root',),human_root=str(human_root),allow_missing_human=True)
    assert lib.has_human
    assert lib.human_valid_at(torch.tensor([0])).all()
    assert not lib.human_valid_at(torch.tensor([3])).any()
    assert lib.human_pos_w.shape == (6,24,3)
    assert torch.isfinite(lib.human_future(torch.tensor([3]),2)).all()

