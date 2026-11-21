"""MotionDecode's explicitly labelled metre/wxyz G1 CSV to native 30 Hz rows."""
import csv
from pathlib import Path
import numpy as np
from morphmimic.constants import JOINT_NAMES
from morphmimic.data.human_convert import resample_poses
from morphmimic.data.local_prepare import _validate

CONVERSION_VERSION = 2
VALIDATION_SCHEMA = 'bumi-native-reference-v2'


def native_rows(path,*,source_fps):
    if source_fps not in (30,50,120):raise ValueError('An explicit supported source FPS is required')
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        header=next(csv.reader(f))
    required=['root_pos_x(m)','root_pos_y(m)','root_pos_z(m)',
              'root_rot_w','root_rot_x','root_rot_y','root_rot_z']+['dof_'+n+'(rad)' for n in JOINT_NAMES]
    if len(header)!=len(set(header)) or set(header)!=set(required):
        raise ValueError('MotionDecode header must label metres, wxyz and all 29 named joints')
    values=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
    if values.size==0:raise ValueError(f'{path}: no motion frames')
    if values.shape[1]!=len(header):raise ValueError(f'{path}: MotionDecode column count differs')
    values=values[:,[header.index(n) for n in required]]
    pos,quat,joints=values[:,:3],values[:,3:7],values[:,7:]
    _validate(pos,quat,joints,source_fps)
    sampled,q=resample_poses(pos[:,None,:],quat[:,None,:],source_fps,30)
    time=np.arange(len(sampled))/30
    qj=np.stack([np.interp(time,np.arange(len(joints))/source_fps,joints[:,j]) for j in range(29)],-1)
    rows=np.concatenate([sampled[:,0],q[:,0,[1,2,3,0]],qj],-1)
    return rows,dict(original_frames=len(pos),original_fps=source_fps,fps=30,
                     original_duration_seconds=(len(pos)-1)/source_fps,
                     root_units='m',root_input_quaternion='wxyz',native_output_quaternion='xyzw')


def validate_bumi_reference(source, output):
    """Check the serialized reference against SDK columns and the actual native model."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from supermimic.bumi import JOINT_NAMES as SDK_JOINT_NAMES
    from supermimic.bumi_ik import support_geometry,sole_heights
    from morphmimic.data.local_prepare import _robot_model,resample,sha256
    from morphmimic.robots.bumi import motor

    spec,model,body_ids=_robot_model('bumi')
    names=tuple(spec.joint_names)
    if set(SDK_JOINT_NAMES)!=set(names) or len(set(SDK_JOINT_NAMES))!=len(names):
        raise ValueError('Bumi SDK joint names differ from native model')
    raw=np.loadtxt(source,delimiter=',',ndmin=2)
    if raw.shape[1]!=7+len(names):raise ValueError('Bumi source column count differs')
    # Derive this permutation from the source columns, independently of the CSV reader.
    joints=raw[:,[7+SDK_JOINT_NAMES.index(name) for name in names]]
    pos,quat,joints=resample(raw[:,:3],raw[:,[6,3,4,5]],joints,30)
    count=len(pos)
    with np.load(output,allow_pickle=False) as saved:z={k:saved[k] for k in saved.files}
    shapes={'joint_pos':(count,len(names)),'joint_vel':(count,len(names)),
            'body_pos_w':(count,14,3),'body_quat_w':(count,14,4),
            'body_lin_vel_w':(count,14,3),'body_ang_vel_w':(count,14,3)}
    if 'fps' not in z or z['fps'].shape!=() or z['fps']!=50:raise ValueError('Bumi reference must be 50 Hz')
    for key,expected in [('joint_names',names),('body_names',spec.tracked_bodies),('robot',np.array('bumi')),('root_body',np.array(spec.root_body))]:
        if key not in z or not np.array_equal(z[key],expected):raise ValueError(f'Bumi reference names/identity differ: {key}')
    for key,shape in shapes.items():
        if key not in z or z[key].shape!=shape or not np.isfinite(z[key]).all():raise ValueError(f'Bumi reference shape/nonfinite: {key}')
    if not np.allclose(z['joint_pos'],joints,atol=2e-7,rtol=0):raise ValueError('Bumi joint mapping differs from SDK source columns')
    if not np.allclose(z['joint_vel'],np.gradient(joints,.02,axis=0),atol=2e-5,rtol=2e-6):raise ValueError('Bumi joint velocity differs from source derivative')
    ids=[model.joint(n).id for n in names];lo,hi=model.jnt_range[ids].T
    violations=(z['joint_pos']<lo-1e-4)|(z['joint_pos']>hi+1e-4)
    if violations.any():raise ValueError(f'Bumi joint limit violation in {int(violations.any(-1).sum())} frames')
    speed=float(np.max(np.abs(z['joint_vel'])/np.array([motor(n)[1] for n in names])))
    if speed>1.0001:raise ValueError(f'Bumi joint speed exceeds nominal limit: {speed}')
    data=mujoco.MjData(model)
    feet=[model.body(n).id for n in ('l_ankle_roll_link','r_ankle_roll_link')]
    geometry=support_geometry(model,feet)
    fk_pos=np.empty((count,14,3));fk_quat=np.empty((count,14,4))
    minimum=0.;contact=0.
    for index in range(count):
        data.qpos[:]=np.r_[pos[index],quat[index],joints[index]]
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        fk_pos[index]=data.xpos[body_ids];fk_quat[index]=data.xquat[body_ids]
        minimum=min(minimum,float(sole_heights(model,data,feet,geometry).min()))
        mujoco.mj_collision(model,data)
        if data.ncon:contact=min(contact,float(np.min(data.contact.dist)))
    position_error=float(np.max(np.abs(z['body_pos_w']-fk_pos)))
    quat_dot=np.abs(np.sum(z['body_quat_w']*fk_quat,axis=-1))
    if position_error>2e-6 or not np.allclose(np.linalg.norm(z['body_quat_w'],axis=-1),1,atol=2e-6,rtol=0) or np.min(quat_dot)<1-2e-6:
        raise ValueError('Bumi FK body/root positions or orientations differ from source')
    before=np.r_[0,np.arange(count-2),count-2];after=np.r_[1,np.arange(2,count),count-1]
    a=Rotation.from_quat(fk_quat[before][...,[1,2,3,0]].reshape(-1,4))
    b=Rotation.from_quat(fk_quat[after][...,[1,2,3,0]].reshape(-1,4))
    angular=(b*a.inv()).as_rotvec().reshape(count,14,3)*(50/(after-before))[:,None,None]
    for key,expected in [('body_lin_vel_w',np.gradient(fk_pos,.02,axis=0)),('body_ang_vel_w',angular)]:
        if not np.allclose(z[key],expected,atol=5e-5,rtol=2e-6):raise ValueError(f'Bumi body velocity differs from FK derivative: {key}')
    if minimum<-.002 or contact<-.002:raise ValueError(f'Bumi converted geometry violation: sole={minimum}, self_contact={contact}')
    return dict(schema=VALIDATION_SCHEMA,source_csv_sha256=sha256(source),output_sha256=sha256(output),
                source_joint_names=list(SDK_JOINT_NAMES),native_joint_names=list(names),frames=count,
                joint_limit_violating_frames=0,max_joint_speed_ratio=speed,min_sole_height_m=minimum,
                min_self_contact_m=contact,fk_max_position_error_m=position_error,
                max_joint_mapping_error_rad=float(np.max(np.abs(z['joint_pos']-joints))))


def convert_reference(source, output, robot):
    """Only publish a Bumi NPZ after its SDK order and converted geometry pass."""
    from morphmimic.data.local_prepare import convert_motion
    if robot!='bumi':return convert_motion(source,output,robot,30)
    from tempfile import TemporaryDirectory
    from supermimic.bumi import JOINT_NAMES as SDK_JOINT_NAMES
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.'+output.stem+'-',dir=output.parent) as work:
        pending=Path(work)/'motion.pending'
        stats=convert_motion(source,pending,robot,30,source_joint_names=SDK_JOINT_NAMES)
        stats['converted_validation']=validate_bumi_reference(source,pending)
        pending.replace(output)
    return stats
