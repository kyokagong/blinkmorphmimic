"""Native neutral bookends; synthetic bridges never imply dynamics qualification."""
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from morphmimic.data.local_prepare import _robot_model
from morphmimic.data.seed_convert import finite_diff_velocities

FPS = 50
DT = 1/FPS


def _rot(q):
    return Rotation.from_quat(np.asarray(q).reshape(-1, 4)[:, [1, 2, 3, 0]])


def _quat(rotation):
    q = rotation.as_quat()[:, [3, 0, 1, 2]]
    return np.where(q[:, :1] < 0, -q, q)


def _heading(q):
    matrix = _rot(q).as_matrix()[0]
    forward = matrix[:2, 0]
    if np.linalg.norm(forward) < 1e-6:
        return float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
    return float(np.arctan2(forward[1], forward[0]))


def source_frame_map(frames, ramp_frames):
    """Authorized edge slow clock in original frame units; middle samples are exact."""
    if type(frames) is not int or frames < 2 or type(ramp_frames) is not int or not 0 <= ramp_frames <= (frames-1)//4:
        raise ValueError('Invalid source frame count or ramp length')
    if ramp_frames == 0:
        return np.arange(frames, dtype=float)
    length = 2*ramp_frames
    u = np.arange(length, dtype=float)/length
    left = length*(u**3-.5*u**4)
    return np.r_[left, np.arange(ramp_frames, frames-ramp_frames), frames-1-left[::-1]]


def _sample(values, mapping):
    lo = np.floor(mapping).astype(int); hi = np.minimum(lo+1, len(values)-1)
    w = (mapping-lo).reshape((-1,)+(1,)*(np.ndim(values)-1))
    return (1-w)*values[lo]+w*values[hi]


def _sample_quat(values, mapping):
    q = np.asarray(values)
    flat = q.reshape(len(q), -1, 4)
    result = np.empty((len(mapping), flat.shape[1], 4))
    for joint in range(flat.shape[1]):
        result[:, joint] = _quat(Slerp(np.arange(len(q)), _rot(flat[:, joint]))(mapping))
    return result.reshape((len(mapping),)+q.shape[1:])


def _hermite(p0, p1, v0, v1, a0, a1, frames):
    """Quintic samples including both endpoints, plus analytic derivatives."""
    duration = frames/FPS
    c0, c1, c2 = p0, v0*duration, .5*a0*duration**2
    a = p1-c0-c1-c2; b = v1*duration-c1-2*c2; c = a1*duration**2-2*c2
    coefficients = [c0, c1, c2, 10*a-4*b+.5*c, -15*a+7*b-c, 6*a-3*b+.5*c]
    u = np.arange(frames+1).reshape((-1,)+(1,)*np.ndim(p0))/frames
    p = sum(coef*u**i for i, coef in enumerate(coefficients))
    v = sum(i*coefficients[i]*u**(i-1) for i in range(1, 6))/duration
    acc = sum(i*(i-1)*coefficients[i]*u**(i-2) for i in range(2, 6))/duration**2
    p[0], p[-1] = p0, p1
    return p, v, acc


@lru_cache(maxsize=3)
def _speed_limits(robot):
    spec = _robot_model(robot)[0]
    if robot == 'q1':
        from morphmimic.robots.q1 import profile
        return np.array([min(j['configured_policy_slew_rad_s'], j['configured_measured_speed_soft_limit_rad_s'])
                         for j in profile()['joints']]), 'Q1 hash-bound configured slew/measured-speed soft caps, not measured hardware limits'
    if robot == 'bumi':
        from morphmimic.robots.bumi import motor
        return np.array([motor(n)[1] for n in spec.joint_names]), 'Bumi pinned native motor velocity table'
    if robot == 'g1':
        from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g
        limits = []
        for name in spec.joint_names:
            if 'hip_roll' in name or 'knee' in name: motor = g.ACTUATOR_7520_22
            elif 'hip_pitch' in name or 'hip_yaw' in name or name == 'waist_yaw_joint': motor = g.ACTUATOR_7520_14
            elif 'wrist_pitch' in name or 'wrist_yaw' in name: motor = g.ACTUATOR_4010
            else: motor = g.ACTUATOR_5020
            limits.append(motor.velocity_limit)
        return np.array(limits), 'Pinned mjlab G1 motor limits; parallel ankle/waist nominal 1:1 gearing'
    raise ValueError('Neutral bookends support q1, bumi, g1')


def _bridge(position, orientation, joints, neutral_pos, neutral_quat, neutral_joints, speed, *, end):
    edge = -1 if end else 0
    base = _rot(orientation[edge])[0]
    source = np.r_[position[edge], np.zeros(3), joints[edge]]
    target = np.r_[neutral_pos, (_rot(neutral_quat)[0]*base.inv()).as_rotvec(), neutral_joints]
    zero = np.zeros_like(target)
    args = (source, target, zero, zero, zero, zero) if end else (target, source, zero, zero, zero, zero)
    effective_v = np.r_[np.ones(3), np.full(3, 3.), speed]
    # shortcut: acceleration caps are preparation heuristics, replace with measured dynamics before hardware qualification.
    effective_a = 4*effective_v
    best = None
    for frames in range(FPS, 10*FPS+1, 5):
        values, v, a = _hermite(*args, frames)
        ratio = max(float(np.max(abs(v)/effective_v)), float(np.max(abs(a)/effective_a)),
                    float(np.linalg.norm(v[:, :3], axis=1).max()), float(np.linalg.norm(v[:, 3:6], axis=1).max()/3),
                    float(np.linalg.norm(a[:, :3], axis=1).max()/4), float(np.linalg.norm(a[:, 3:6], axis=1).max()/12))
        if best is None or ratio < best[0]: best = (ratio, frames, values, v, a)
        if ratio <= 1.00001: break
    ratio, frames, values, v, a = best
    diagnostics = dict(frames=frames, sampled_limit_ratio=ratio, derivative_limits_met=bool(ratio <= 1.00001),
        endpoint_derivatives='Zero velocity/acceleration static endpoints; original motion edges use authorized slow clock.',
        maximum_joint_speed_rad_s=float(abs(v[:, 6:]).max()), maximum_joint_acceleration_rad_s2=float(abs(a[:, 6:]).max()),
        maximum_root_speed_m_s=float(np.linalg.norm(v[:, :3], axis=1).max()))
    return values[:, :3], _quat(Rotation.from_rotvec(values[:, 3:6])*base), values[:, 6:], diagnostics


def _validate_motion(motion, spec):
    n = len(motion['joint_pos'])
    if int(motion['fps']) != FPS or n < 2 or tuple(motion['joint_names']) != spec.joint_names or tuple(motion['body_names']) != spec.tracked_bodies:
        raise ValueError('Invalid native fps, frame count, joint or body order')
    shapes = {'joint_pos': (n, len(spec.joint_names)), 'body_pos_w': (n, 14, 3), 'body_quat_w': (n, 14, 4)}
    for key, shape in shapes.items():
        if np.shape(motion[key]) != shape or not np.isfinite(motion[key]).all():
            raise ValueError('Invalid/nonfinite native '+key)
    if not np.allclose(np.linalg.norm(motion['body_quat_w'], axis=-1), 1, atol=1e-4, rtol=0):
        raise ValueError('Nonunit native body quaternion')


def neutralize_motion(source, target, robot, *, target_frame=11446, hold_frames=50, source_floor_offset=0., in_place=False):
    """Canonicalize and gently retime action edges; retain every middle sample/time step.

    Source start/stop are inclusive/exclusive. Entry/exit frame counts exclude
    holds and the source core. A one-second hold includes a duplicate at each
    bridge boundary so central-difference velocities are zero throughout holds.
    """
    import mujoco
    if type(in_place) is not bool or type(hold_frames) is not int or hold_frames < FPS or not np.isfinite(source_floor_offset):
        raise ValueError('Finite floor offset and at least one-second integer holds required')
    spec, model, ids = _robot_model(robot)
    _validate_motion(source, spec)
    if type(target_frame) is not int or not 0 <= target_frame < len(target['joint_pos']):
        raise ValueError('Approved target frame is outside motion')
    # Validate only the selected target, not a long unrelated trajectory per clip.
    selected = {k: np.asarray(target[k])[target_frame:target_frame+1] for k in ('joint_pos', 'body_pos_w', 'body_quat_w')}
    selected.update(fps=target['fps'], joint_names=target['joint_names'], body_names=target['body_names'])
    if int(selected['fps']) != FPS or tuple(selected['joint_names']) != spec.joint_names or tuple(selected['body_names']) != spec.tracked_bodies:
        raise ValueError('Approved native layout differs')
    if any(not np.isfinite(selected[k]).all() for k in ('joint_pos', 'body_pos_w', 'body_quat_w')):
        raise ValueError('Nonfinite approved native target')
    if not np.allclose(np.linalg.norm(selected['body_quat_w'], axis=-1), 1, atol=1e-4):
        raise ValueError('Invalid approved quaternion')
    if ids[0] != model.jnt_bodyid[0]: raise ValueError('Tracked root is not the native free-joint body')
    position = source['body_pos_w'][:, 0].astype(float)
    orientation = source['body_quat_w'][:, 0].astype(float)
    upright = np.flatnonzero(_rot(orientation).as_matrix()[:, 2, 2] > .7)
    heading_frame = len(position)-1 if in_place else (int(upright[0]) if len(upright) else 0)
    yaw = -_heading(orientation[heading_frame])
    turn = Rotation.from_rotvec([0., 0., yaw])
    moved = turn.apply(position)
    translation = np.r_[-moved[-1 if in_place else 0, :2], -float(source_floor_offset)]
    position = moved+translation; orientation = _quat(turn*_rot(orientation))
    original_frames = len(position); ramp = min(25, (original_frames-1)//4)
    mapping = source_frame_map(original_frames, ramp)
    position = _sample(position, mapping); orientation = _sample_quat(orientation, mapping)
    joints = _sample(source['joint_pos'].astype(float), mapping)
    tp, tq, tj = selected['body_pos_w'][0, 0].astype(float), selected['body_quat_w'][0, 0].astype(float), selected['joint_pos'][0].astype(float)
    target_yaw = _heading(tq)
    neutral_q = _quat(Rotation.from_rotvec([0., 0., -target_yaw])*_rot(tq))[0]
    first_pos = np.r_[0., 0., tp[2]]
    last_heading_frame = int(upright[-1]) if len(upright) else original_frames-1
    last_yaw = 0. if in_place else _heading(_quat(turn*_rot(source['body_quat_w'][last_heading_frame, 0]))[0])
    last_pos = first_pos.copy() if in_place else np.r_[position[-1, :2], tp[2]]
    last_quat = _quat(Rotation.from_rotvec([0., 0., last_yaw])*_rot(neutral_q))[0]
    speed, speed_origin = _speed_limits(robot)
    ep, eq, ej, entry = _bridge(position[:3], orientation[:3], joints[:3], first_pos, neutral_q, tj, speed, end=False)
    xp, xq, xj, exit = _bridge(position[-3:], orientation[-3:], joints[-3:], last_pos, last_quat, tj, speed, end=True)
    def assemble(start, incoming, core, outgoing, finish):
        return np.concatenate([np.repeat(start[None], hold_frames, axis=0), incoming[:-1], core,
                               outgoing[1:], np.repeat(finish[None], hold_frames, axis=0)])
    root_pos = assemble(first_pos, ep, position, xp, last_pos)
    root_quat = assemble(neutral_q, eq, orientation, xq, last_quat)
    angles = assemble(tj, ej, joints, xj, tj)
    a = hold_frames+entry['frames']; b = a+len(joints); end_hold = b+exit['frames']
    data = mujoco.MjData(model); pos = np.empty((len(angles), 14, 3)); quat = np.empty((len(angles), 14, 4))
    for i in range(len(angles)):
        data.qpos[:] = np.r_[root_pos[i], root_quat[i], angles[i]]
        mujoco.mj_kinematics(model, data)
        pos[i], quat[i] = data.xpos[ids], data.xquat[ids]
    quat = np.where(quat[..., :1] < 0, -quat, quat)
    # Retain float64 until validation: float32 quantization can create spurious high derivatives.
    velocities = finite_diff_velocities(angles, pos, quat, FPS)
    out = dict(fps=np.array(FPS), joint_pos=angles, body_pos_w=pos, body_quat_w=quat,
        joint_vel=velocities[0], body_lin_vel_w=velocities[1], body_ang_vel_w=velocities[2],
        body_names=np.array(spec.tracked_bodies), joint_names=np.array(spec.joint_names), robot=np.array(robot), root_body=np.array(spec.root_body))
    synthetic_angles = np.concatenate([angles[:a], angles[b:]])
    limit_error = np.maximum(model.jnt_range[1:, 0]-synthetic_angles, synthetic_angles-model.jnt_range[1:, 1]).max(initial=0)
    source_limit_error = np.maximum(model.jnt_range[1:, 0]-source['joint_pos'], source['joint_pos']-model.jnt_range[1:, 1]).max(initial=0)
    # Verify source FK, rather than silently repairing an inconsistent action core.
    wanted = turn.apply(source['body_pos_w'].reshape(-1, 3)).reshape(source['body_pos_w'].shape)+translation
    integer = mapping == np.floor(mapping); indices = mapping[integer].astype(int)
    fk_error = float(abs(pos[a:b][integer]-wanted[indices]).max())
    wanted_q = _quat(turn*_rot(source['body_quat_w'])).reshape(original_frames, 14, 4)
    if fk_error > 5e-5 or not np.allclose(abs((quat[a:b][integer]*wanted_q[indices]).sum(axis=-1)), 1., atol=1e-5, rtol=0):
        raise ValueError('Source action body poses do not match native FK')
    contract = dict(schema_version=1, robot=robot, fps=FPS, recovery_in_place=in_place, target_frame=target_frame, frames=len(angles),
        source_start_frame=a, source_stop_frame=b, entry_hold_end_frame=hold_frames, exit_hold_start_frame=end_hold,
        entry_frames=entry['frames'], exit_frames=exit['frames'], root_yaw_rotation_rad=yaw, root_translation_xyz=translation.tolist(),
        approved_root_yaw_rad=target_yaw, exit_neutral_yaw_rad=last_yaw, heading_source_frame=heading_frame,
        synthetic_intervals=[[0, a], [b, len(angles)]], source_frame_interval=[0, original_frames],
        original_source_frames=original_frames, source_ramp_frames=ramp,
        retiming='User-authorized first/last m source-frame ramps occupy 2m output frames each; smooth cubic rate 0-to-1 and 1-to-0, middle exact original samples at 50Hz. Map reconstructed by source_frame_map(original_source_frames,source_ramp_frames).',
        human_coordinate_convention='Robot and human original world frames differ. Independently canonicalize each core by its own root heading and initial root XY; apply source_floor_offset only to native robot. Apply the same authorized slow source clock after each independent rigid transform; preserve original middle samples.',
        entry=entry, exit=exit, dynamics_qualified=False,
        quality=dict(native_source_fk_error_m=fk_error, synthetic_joint_limit_error_rad=float(limit_error), source_joint_limit_error_rad=float(source_limit_error),
            heading_from_upright=bool(len(upright)), entry_source_upright=bool(_rot(orientation[0]).as_matrix()[0, 2, 2] > .7),
            exit_source_upright=bool(_rot(orientation[-1]).as_matrix()[0, 2, 2] > .7), speed_limit_origin=speed_origin,
            acceleration_limit_origin='Heuristic four times velocity cap per second; not measured acceleration limits.',
            qualification='Synthetic kinematic bridges only. Contacts, support, collision, torque, balance and physical switching require separate verification.'))
    return out, contract


# SMPL/SOMA 24-slot hierarchy used by the existing human schema.
_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def _human_bridge(positions, quaternions, target_pos, target_quat, frames, *, end):
    edge = -1 if end else 0
    zero = np.zeros_like(positions[edge])
    args = (positions[edge], target_pos, zero, zero, zero, zero) if end else (target_pos, positions[edge], zero, zero, zero, zero)
    raw = _hermite(*args, frames)[0]; result = raw.copy()
    start = positions[edge] if end else target_pos; finish = target_pos if end else positions[edge]
    t = np.arange(frames+1)/frames; w = t**3*(10-15*t+6*t*t)
    for joint, parent in enumerate(_PARENTS[1:], 1):
        first, last = start[joint]-start[parent], finish[joint]-finish[parent]
        l0, l1 = np.linalg.norm(first), np.linalg.norm(last)
        bone = raw[:, joint]-raw[:, parent]; length = np.linalg.norm(bone, axis=-1)
        # Bone direction plus interpolated length avoids shrinking limbs through a Cartesian chord.
        desired = (1-w)*l0+w*l1
        fallback = (1-w[:, None])*first+w[:, None]*last
        tiny = length < 1e-6
        bone[tiny] = fallback[tiny]; length = np.linalg.norm(bone, axis=-1)
        if np.any(length < 1e-8):
            axis = np.cross(first, np.eye(3)[np.argmin(abs(first))])
            if np.linalg.norm(axis) < 1e-8: axis = np.array([1., 0., 0.])
            bone[length < 1e-8] = axis; length = np.linalg.norm(bone, axis=-1)
        result[:, joint] = result[:, parent]+bone/length[:, None]*desired[:, None]
    result[0], result[-1] = start, finish
    orientations = []
    for joint in range(24):
        base = _rot(quaternions[edge, joint])[0]
        local = (_rot(quaternions[:, joint])*base.inv()).as_rotvec()
        target = (_rot(target_quat[joint])[0]*base.inv()).as_rotvec()
        zero = np.zeros(3)
        args = (local[edge], target, zero, zero, zero, zero) if end else (target, local[edge], zero, zero, zero, zero)
        orientations.append(_quat(Rotation.from_rotvec(_hermite(*args, frames)[0])*base))
    return result, np.stack(orientations, axis=1)


def neutralize_human(human, target_human, contract, *, target_frame=11446):
    """Preserve existing human sidecar core; new bookends are synthetic derivatives."""
    from morphmimic.data.human_prepare import validate_human, five_points
    validate_human(human); validate_human(target_human)
    n = len(human['human_pos_w']); a, b = contract['source_start_frame'], contract['source_stop_frame']
    mapping = source_frame_map(n, contract['source_ramp_frames'])
    if b-a != len(mapping) or n != contract['original_source_frames'] or type(target_frame) is not int or not 0 <= target_frame < len(target_human['human_pos_w']):
        raise ValueError('Human source/target frame interval differs')
    anchor = -1 if contract.get('recovery_in_place') else 0
    # SMPL24 root X is lateral; physical forward is -Y, unlike native robot +X.
    human_yaw = np.pi/2-_heading(human['human_quat_w'][anchor, 0])
    turn = Rotation.from_rotvec([0., 0., human_yaw])
    moved = turn.apply(human['human_pos_w'].reshape(-1, 3)).reshape(n, 24, 3)
    translation = np.r_[-moved[anchor, 0, :2], 0.]
    p = moved+translation
    q = _quat(turn*_rot(human['human_quat_w'])).reshape(n, 24, 4)
    p = _sample(p, mapping); q = _sample_quat(q, mapping)
    tp, tq = target_human['human_pos_w'][target_frame], target_human['human_quat_w'][target_frame]
    target_yaw = _heading(tq[0])-np.pi/2
    def endpoint(position, yaw):
        rotation = Rotation.from_rotvec([0., 0., yaw-target_yaw])
        pose = rotation.apply(tp-tp[0])+tp[0]
        pose[:, :2] += position[:2]-pose[0, :2]
        return pose, _quat(rotation*_rot(tq))
    first_pos, first_q = endpoint(np.zeros(3), 0.)
    last_pos, last_q = (first_pos.copy(), first_q.copy()) if contract.get('recovery_in_place') else endpoint(p[-1, 0], _heading(q[-1, 0])-np.pi/2)
    entry_frames = contract.get('human_entry_frames', contract['entry_frames'])
    exit_frames = contract.get('human_exit_frames', contract['exit_frames'])
    hold = a-entry_frames; tail = contract['frames']-b-exit_frames
    if min(hold, tail) < FPS:
        raise ValueError('Aligned human bridge must retain at least one-second neutral holds')
    ep, eq = _human_bridge(p[:3], q[:3], first_pos, first_q, entry_frames, end=False)
    xp, xq = _human_bridge(p[-3:], q[-3:], last_pos, last_q, exit_frames, end=True)
    positions = np.concatenate([np.repeat(first_pos[None], hold, axis=0), ep[:-1], p, xp[1:], np.repeat(last_pos[None], tail, axis=0)])
    quats = np.concatenate([np.repeat(first_q[None], hold, axis=0), eq[:-1], q, xq[1:], np.repeat(last_q[None], tail, axis=0)])
    timestamps = np.full(contract['frames'], np.nan)
    if 'source_time_s' in human:
        original_time = np.asarray(human['source_time_s'])
        if original_time.shape != (n,) or not np.isfinite(original_time).all():
            raise ValueError('Invalid original human source timestamps')
        timestamps[a:b] = np.interp(mapping, np.arange(n), original_time)
    frame_index = np.full(contract['frames'], -1., dtype=np.float64); frame_index[a:b] = mapping
    valid = np.asarray(human.get('human_joint_valid', np.ones(24, dtype=bool))) & np.asarray(target_human.get('human_joint_valid', np.ones(24, dtype=bool)))
    side = dict(fps=np.array(FPS), human_joint_names=human['human_joint_names'].copy(), human_joint_valid=valid,
        human_pos_w=positions, human_quat_w=quats, time_s=np.arange(contract['frames'])/FPS,
        source_time_s=timestamps, source_frame_index=frame_index,
        root_yaw_rotation_rad=np.array(human_yaw), root_translation_xyz=translation,
        entry_hold_end_frame=np.array(hold), exit_hold_start_frame=np.array(b+exit_frames),
        derivation=np.array('Existing human sidecar core independently canonicalized in its original human world frame; user-authorized slow edge clock with exact original middle samples; original source/derived provenance remains in the manifest; synthetic quintic bridges with interpolated bone lengths/directions, not recorded transitions. NaN source_time_s means no original source clock.'))
    validate_human(side); five_points(side)
    return side


def validate_boundary(out, target, robot, contract, *, target_frame=11446):
    """Verify neutral contract, native FK and derivatives; no dynamics claim."""
    import mujoco
    spec, model, ids = _robot_model(robot)
    _validate_motion(out, spec)
    n = len(out['joint_pos']); a, b = contract['source_start_frame'], contract['source_stop_frame']
    hold, end_hold = contract['entry_hold_end_frame'], contract['exit_hold_start_frame']
    if (contract['robot'] != robot or contract['fps'] != FPS or contract['frames'] != n
            or not FPS <= hold < a < b <= end_hold <= n-FPS
            or a-hold != contract['entry_frames'] or end_hold-b != contract['exit_frames']):
        raise ValueError('Neutral contract frame intervals differ')
    if len(source_frame_map(contract['original_source_frames'], contract['source_ramp_frames'])) != b-a:
        raise ValueError('Retimed core frame count differs from source clock map')
    expected = target['joint_pos'][target_frame]
    if tuple(target['joint_names']) != spec.joint_names:
        raise ValueError('Approved target native joint order differs')
    for span in (slice(0, hold), slice(end_hold, n)):
        if not np.allclose(out['joint_pos'][span], expected, atol=1e-6, rtol=0):
            raise ValueError('Neutral native joints differ from approved target')
        for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
            if not np.isfinite(out[key]).all() or not np.allclose(out[key][span], 0., atol=1e-7, rtol=0):
                raise ValueError('Neutral hold is moving/nonfinite')
        if not np.allclose(out['body_pos_w'][span], out['body_pos_w'][span.start], atol=1e-6, rtol=0):
            raise ValueError('Neutral body hold is moving')
    if not np.allclose(out['body_pos_w'][0, 0, :2], 0., atol=1e-6, rtol=0):
        raise ValueError('Initial neutral root XY is not canonical zero')
    if not np.allclose(out['body_pos_w'][-1, 0, :2], out['body_pos_w'][b-1, 0, :2], atol=1e-6, rtol=0):
        raise ValueError('Exit neutral root returns away from action endpoint')
    synthetic_limit_error = 0.
    for span, endpoint in ((slice(0, a), out['joint_pos'][a]), (slice(b, n), out['joint_pos'][b-1])):
        lo, hi = np.minimum(expected, endpoint), np.maximum(expected, endpoint)
        angles = out['joint_pos'][span]
        if np.any(angles < lo-1e-6) or np.any(angles > hi+1e-6):
            raise ValueError('Synthetic bridge adds native joint excursion beyond endpoint poses')
        synthetic_limit_error = max(synthetic_limit_error, float(np.maximum(
            model.jnt_range[1:, 0]-angles, angles-model.jnt_range[1:, 1]).max(initial=0)))
    target_q = target['body_quat_w'][target_frame, 0]
    canonical = Rotation.from_rotvec([0., 0., -_heading(target_q)])*_rot(target_q)
    for index, yaw in ((0, 0.), (-1, contract['exit_neutral_yaw_rad'])):
        actual = _rot(out['body_quat_w'][index, 0])
        desired = Rotation.from_rotvec([0., 0., yaw])*canonical
        if np.linalg.norm((actual*desired.inv()).as_rotvec()) > 1e-5:
            raise ValueError('Neutral root orientation differs from approved intrinsic pose')
        if abs(out['body_pos_w'][index, 0, 2]-target['body_pos_w'][target_frame, 0, 2]) > 1e-6:
            raise ValueError('Neutral root height differs from approved pose')
    if contract.get('recovery_in_place') and (not np.allclose(out['body_pos_w'][0], out['body_pos_w'][-1], atol=1e-6, rtol=0)
            or not np.allclose(out['body_quat_w'][0], out['body_quat_w'][-1], atol=1e-6, rtol=0)):
        raise ValueError('In-place recovery must end at the identical full initial neutral pose')
    max_error = 0.; data = mujoco.MjData(model)
    for i in range(n):
        data.qpos[:] = np.r_[out['body_pos_w'][i, 0], out['body_quat_w'][i, 0], out['joint_pos'][i]]
        mujoco.mj_kinematics(model, data)
        max_error = max(max_error, float(abs(data.xpos[ids]-out['body_pos_w'][i]).max()))
        if max_error > 3e-6 or not np.allclose(abs((data.xquat[ids]*out['body_quat_w'][i]).sum(axis=-1)), 1., atol=1e-5, rtol=0):
            raise ValueError('Prepared pose does not match native FK')
    for key, wanted in zip(('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'), finite_diff_velocities(
            out['joint_pos'].astype(float), out['body_pos_w'].astype(float), out['body_quat_w'].astype(float), FPS)):
        if not np.allclose(out[key], wanted, atol=2e-4, rtol=1e-4):
            raise ValueError('Prepared velocities do not match assembled poses')
    return dict(valid=True, maximum_native_fk_error_m=max_error, synthetic_joint_limit_error_rad=synthetic_limit_error, dynamics_qualified=False)
