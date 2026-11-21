"""Approved native standing reference and geometry shared by recovery checks."""
from pathlib import Path
import hashlib
import json
import numpy as np


def arm_geometry(body_pos):
    """Worst of both arms; zero elbow bend means collinear shoulder/elbow/wrist."""
    if type(body_pos).__module__.startswith('torch'):
        import torch as xp
    else:
        xp = np
    shoulder = body_pos[..., [8, 11], :]
    upper = body_pos[..., [9, 12], :] - shoulder
    lower = body_pos[..., [10, 13], :] - body_pos[..., [9, 12], :]
    arm = body_pos[..., [10, 13], :] - shoulder
    def norm(v): return (v*v).sum(axis=-1)**.5
    def angle(cos): return xp.arccos(xp.clip(cos, -1., 1.))*180./np.pi
    with np.errstate(invalid='ignore', divide='ignore'):
        return dict(upper_arm_deg=xp.amax(angle(-upper[..., 2]/norm(upper)), axis=-1),
            shoulder_hand_deg=xp.amax(angle(-arm[..., 2]/norm(arm)), axis=-1),
            elbow_bend_deg=xp.amax(angle((upper*lower).sum(axis=-1)/(norm(upper)*norm(lower))), axis=-1),
            wrist_root_height=xp.amax(body_pos[..., [10, 13], 2]-body_pos[..., 0, 2, None], axis=-1))


def native_arm_rmse(joint_pos, joint_names, target):
    """Compare named native joints, including the robot's actual elbow convention."""
    names = list(joint_names)
    arms = [i for i, name in enumerate(target['joint_names'])
            if any(stem in name for stem in ('shoulder', 'elbow', 'wrist', '_arm_'))]
    if not arms or len(set(names)) != len(names):
        raise ValueError('Missing or duplicate native arm joint names')
    try:
        indices = [names.index(target['joint_names'][i]) for i in arms]
    except ValueError as error:
        raise ValueError('Target native arm joint missing in actual robot') from error
    expected = target['joint_pos'][arms]
    if type(joint_pos).__module__.startswith('torch'):
        expected = joint_pos.new_tensor(expected)
    return ((joint_pos[..., indices]-expected)**2).mean(axis=-1)**.5


def load_standing_target(receipt, robot):
    path = Path(receipt).absolute()
    if path.is_symlink(): raise ValueError('Standing target receipt must not be a symlink')
    document = json.loads(path.read_text())
    frame = document['prepared_frame_zero_based']
    if type(frame) is not int or frame < 0 or document['prepared_fps'] != 50:
        raise ValueError('Invalid standing target frame or fps')
    source = document['robot_sources'][robot]; motion = Path(source['npz'])
    digest = hashlib.sha256(motion.read_bytes()).hexdigest()
    if motion.is_symlink() or digest != source['sha256']:
        raise ValueError('Standing target native source SHA differs')
    with np.load(motion, allow_pickle=False) as data:
        if int(data['fps']) != 50: raise ValueError('Standing target fps differs')
        target = dict(joint_pos=data['joint_pos'][frame].copy(), joint_names=list(data['joint_names']),
            body_pos=data['body_pos_w'][frame].copy(), body_names=list(data['body_names']))
    if (len(target['body_names']) != 14 or len(set(target['joint_names'])) != len(target['joint_names'])
            or not np.isfinite(target['joint_pos']).all() or not np.isfinite(target['body_pos']).all()):
        raise ValueError('Invalid standing target native state')
    target.update(receipt=str(path), receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source=str(motion), source_sha256=digest, frame=frame)
    return target
