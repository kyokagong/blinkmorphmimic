"""LAFAN BVH to the 24-slot human convention; missing calibrations stay explicit.

Rest-frame offsets follow GMR's bvh_lafan1_to_g1/smplx_to_g1 configuration:
R_human = R_lafan * offset_lafan * inv(offset_smpl). Its FootMod uses the toe's
orientation at the ankle position. Existing local G1 references use an extra Rz90
world basis (verified independently by their root trajectories).
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from morphmimic.data.bvh import load_bvh_poses
from morphmimic.data.human_convert import YUP_CM_TO_ZUP_M

SLOTS=('Hips','LeftUpLeg','RightUpLeg','Spine','LeftLeg','RightLeg','Spine1',
       'LeftFoot','RightFoot','Spine2','LeftToe','RightToe','Neck','LeftShoulder',
       'RightShoulder','Head','LeftArm','RightArm','LeftForeArm','RightForeArm',
       'LeftHand','RightHand','LeftHand','RightHand')
ROOT_L=(.5,.5,.5,.5)
ROOT_S=(.5,-.5,-.5,-.5)
LEG_L=(-.5,.5,.5,-.5)
FOOT_L=(-2**-.5,2**-.5,0,0)
ARM_L=(.5,.5,-.5,-.5)
HAND_L=(2**-.5,2**-.5,0,0)
# Only these joints have an explicit compatible anatomical-frame calibration.
CALIBRATED={0,1,2,4,5,7,8,9,10,11,16,17,18,19,20,21}


def _offset(slot):
    left=slot in (16,18,20,22)
    if slot in (1,2,4,5):a,b=LEG_L,ROOT_S
    elif slot in (7,8,10,11):a,b=FOOT_L,ROOT_S
    elif slot in (16,17):a,b=ARM_L,((2**-.5,0,-2**-.5,0) if left else (0,2**-.5,0,2**-.5))
    elif slot in (18,19,20,21,22,23):a,b=HAND_L,((1,0,0,0) if left else (0,0,0,-1))
    else:a,b=ROOT_L,ROOT_S
    return Rotation.from_quat(a,scalar_first=True)*Rotation.from_quat(b,scalar_first=True).inv()


def lafan_slot_poses(path):
    pos,quat,names,dt=load_bvh_poses(Path(path))
    missing=set(SLOTS)-names.keys()
    if missing:raise ValueError(f'LAFAN skeleton missing {sorted(missing)}')
    if round(1/dt)!=30:raise ValueError(f'LAFAN source is not 30Hz: {dt}')
    pos,quat=YUP_CM_TO_ZUP_M(pos,quat)
    world=Rotation.from_euler('z',90,degrees=True)
    index=[names[n] for n in SLOTS]
    output=world.apply(pos[:,index].reshape(-1,3)).reshape(len(pos),24,3)
    rotations=np.empty((len(pos),24,4))
    for slot,name in enumerate(SLOTS):
        rotation_name=('LeftToe' if slot==7 else 'RightToe' if slot==8 else name)
        raw=Rotation.from_quat(quat[:,names[rotation_name]],scalar_first=True)
        rotations[:,slot]=(world*raw*_offset(slot)).as_quat(scalar_first=True)
    rotations=np.where(rotations[...,:1]<0,-rotations,rotations)
    valid=np.array([i in CALIBRATED for i in range(24)])
    return output,rotations,dt,valid
