"""Persist uploaded PICO recordings independently of fixed model releases."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading

MAX_BYTES = 32 * 1024 * 1024
SLOTS = (0, 7, 8, 20, 21)
ROLES = ('pelvis', 'left_ankle', 'right_ankle', 'left_wrist', 'right_wrist')


def read_recording(raw):
    import numpy as np
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('PICO 文件大小须为 1 字节至 32 MB')
    try:
        rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    except (UnicodeError, ValueError) as error:
        raise ValueError('PICO 文件须为 UTF-8 JSONL') from error
    if len(rows) < 4 or not all(isinstance(row, dict) for row in rows):
        raise ValueError('PICO 录制至少需要两帧和完整首尾记录')
    head, end = rows[0], rows[-1]
    if (head.get('type') != 'header' or head.get('schema') != 'embodied.mocap.five-point-recording/v1'
            or head.get('coordinate_frame') != 'pico_source_y_up' or head.get('position_unit') != 'm'
            or head.get('quaternion_order') != 'wxyz' or head.get('slots') != list(SLOTS)
            or head.get('roles') != list(ROLES)):
        raise ValueError('不支持的 PICO 五点格式或坐标约定')
    frames = rows[1:-1]
    if (end.get('type') != 'end' or end.get('complete') is not True
            or end.get('frame_count') != len(frames) or any(f.get('type') != 'frame' for f in frames)):
        raise ValueError('录制不完整或帧数与结束记录不一致')
    try:
        ns = [int(frame['elapsed_ns']) for frame in frames]
        time = np.array([n - ns[0] for n in ns], dtype=float) * 1e-9
        positions = np.asarray([f['positions_m'] for f in frames], dtype=float)
        quaternions = np.asarray([f['quaternions_wxyz'] for f in frames], dtype=float)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ValueError('PICO 帧中的时间或点位数值无效') from error
    if (not np.isfinite(time).all() or not np.all(np.diff(time) > 0)
            or not .02 <= time[-1] <= 600 or len(frames) > 100000):
        raise ValueError('时间戳须严格递增，录制时长须为 0.02 至 600 秒')
    if (positions.shape != (len(frames), 5, 3) or quaternions.shape != (len(frames), 5, 4)
            or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
            or np.max(np.abs(positions)) > 100
            or np.max(np.abs(np.linalg.norm(quaternions, axis=-1) - 1)) > .02):
        raise ValueError('五点位置或单位四元数无效')
    return time, positions, quaternions


def convert_recording(raw, directory, templates):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from morphmimic.constants import HUMAN_JOINTS

    time, positions, quaternions = read_recording(raw)
    axis = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    sdk = Rotation.from_quat(quaternions.reshape(-1, 4), scalar_first=True).as_matrix().reshape(-1, 5, 3, 3)
    forward = axis @ sdk[0, 0] @ np.array([0., 0., -1.])
    if np.linalg.norm(forward[:2]) < .1:
        raise ValueError('首帧骨盆朝向接近竖直，无法标定水平朝向')
    yaw = float(np.arctan2(forward[1], forward[0]))
    heading = Rotation.from_rotvec([0., 0., -yaw]).as_matrix()
    positions = positions @ axis.T @ heading.T
    floor = float(np.median(positions[:min(90, len(time)), 1:3, 2]))
    positions[:, :, 2] -= floor - .08
    positions[:, :, :2] -= positions[0, 0, :2].copy()
    physical = heading @ axis @ sdk @ axis.T
    grid = np.arange(int(np.floor(time[-1] * 50)) + 1) / 50
    frames = len(grid)
    rows = []
    for robot, template in templates.items():
        path = directory / robot / 'pico.npz'
        human = directory / 'human50' / robot / 'pico.npz'
        path.parent.mkdir(parents=True)
        human.parent.mkdir(parents=True)
        with np.load(template['motion'], allow_pickle=False) as source:
            motion = {k: source[k].copy() for k in source.files}
        for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
            motion[key] = np.repeat(motion[key][:1], frames, axis=0)
        for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
            motion[key].fill(0)
        np.savez(path, **motion)
        with np.load(template['human'], allow_pickle=False) as source:
            neutral = source['human_quat_w'][0, list(SLOTS)].copy()
        pos = np.zeros((frames, 24, 3), np.float32)
        quat = np.zeros((frames, 24, 4), np.float32)
        quat[:, :, 0] = 1
        for point, slot in enumerate(SLOTS):
            pos[:, slot] = np.stack([np.interp(grid, time, positions[:, point, i]) for i in range(3)], axis=-1)
            rotations = physical[:, point] @ physical[0, point].T @ Rotation.from_quat(neutral[point], scalar_first=True).as_matrix()
            quat[:, slot] = Slerp(time, Rotation.from_matrix(rotations))(grid).as_quat(scalar_first=True)
        valid = np.zeros(24, bool)
        valid[list(SLOTS)] = True
        np.savez(human, fps=np.array(50), human_pos_w=pos, human_quat_w=quat,
                 human_joint_names=np.array(HUMAN_JOINTS), human_joint_valid=valid,
                 source_time_s=grid, source_frames=np.array(len(time)))
        rows.append(dict(robot=robot, name='pico', output=str(path.relative_to(directory)),
                         output_sha256=file_sha(path), human_sidecar=str(human.relative_to(directory)),
                         human_sha256=file_sha(human), human_available=True,
                         source_id='pico/' + hashlib.sha256(raw).hexdigest(), frames=frames))
    manifest = dict(complete=True, human_data=True, clips=rows,
                    calibration=dict(initial_yaw_rad=yaw, ankle_origin_height_m=.08,
                                     source_floor_z_m=floor, position_scale=1.,
                                     orientation='Per-point first-frame relative rotation to approved neutral SMPL axes'),
                    purpose='Uploaded human five-point input; static robot reference supplies initialization and monitoring.')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return dict(duration_seconds=float(grid[-1]), source_frames=len(time), frames=frames, fps=50)


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PicoRecordings:
    def __init__(self, root, templates):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.templates = templates
        self.lock = threading.Lock()

    def list(self):
        with self.lock:
            return sorted((self.metadata(p.name) for p in self.root.iterdir()
                           if p.is_dir() and not p.name.startswith('.')), key=lambda r: r['uploaded_at'], reverse=True)

    def metadata(self, identity):
        import re
        if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{24}', identity):
            raise ValueError('未知 PICO 录制')
        folder = self.root / identity
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError('未知 PICO 录制')
        metadata = json.loads((folder / 'recording.json').read_text())
        if metadata['id'] != identity or file_sha(folder / 'manifest.json') != metadata['manifest_sha256']:
            raise ValueError('PICO 录制清单校验失败')
        return metadata

    def upload(self, raw, filename):
        import time
        if not isinstance(filename, str) or not filename or len(filename) > 240:
            raise ValueError('上传文件名无效')
        identity = hashlib.sha256(raw).hexdigest()[:24]
        with self.lock:
            dest = self.root / identity
            if dest.exists():
                return self.metadata(identity)
            stage = Path(tempfile.mkdtemp(prefix='.upload-', dir=self.root))
            try:
                summary = convert_recording(raw, stage, self.templates)
                (stage / 'source.jsonl').write_bytes(raw)
                meta = dict(id=identity, filename=Path(filename.replace('\\', '/')).name,
                            uploaded_at=time.time(), source_sha256=hashlib.sha256(raw).hexdigest(),
                            manifest_sha256=file_sha(stage / 'manifest.json'), **summary)
                (stage / 'recording.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
                os.replace(stage, dest)
                return meta
            finally:
                if stage.exists():
                    shutil.rmtree(stage)

    def catalog(self, base, identity):
        from preview.server import Catalog
        if base.interface != 'human':
            raise ValueError('PICO 录制需要选择 Human 模型')
        meta = self.metadata(identity)
        catalog = Catalog(self.root / identity, base.checkpoints, interface='human')
        if catalog.manifest_sha256 != meta['manifest_sha256']:
            raise ValueError('PICO 录制清单校验失败')
        catalog.training_manifest = base.training_manifest
        catalog.training_manifest_sha256 = base.training_manifest_sha256
        return catalog
