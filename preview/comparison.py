"""Synchronized independent policy worlds, displayed in one visual-only scene."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import threading

ROBOTS = ('q1', 'bumi', 'g1')


class AsyncComparisonRenderer:
    """Render immutable completed-step snapshots, keeping EGL on one thread."""
    def __init__(self, factory):
        self.condition = threading.Condition()
        self.pending = self.result = self.error = None
        self.epoch = self.serial = 0
        self.stopped = False
        self.thread = threading.Thread(target=self._run, args=(factory,), name='comparison-render')
        self.thread.start()

    def submit(self, poses, reset=False):
        with self.condition:
            if reset:
                self.epoch += 1
                self.result = None
            reset_poses = poses if reset else (self.pending[2] if self.pending and self.pending[0] == self.epoch else None)
            self.pending = (self.epoch, poses, reset_poses)
            self.condition.notify()

    def latest(self):
        with self.condition:
            if self.error:
                raise RuntimeError(f'Comparison rendering failed: {self.error}') from self.error
            return self.result

    def wait_after(self, serial):
        with self.condition:
            self.condition.wait_for(lambda: self.error or self.stopped or
                                    (self.result is not None and self.result[0] > serial))
            if self.error:
                raise RuntimeError(f'Comparison rendering failed: {self.error}') from self.error
            if self.stopped:
                raise RuntimeError('Comparison renderer is closed')
            return self.result

    def _run(self, factory):
        renderer = None
        try:
            renderer = factory()
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.stopped)
                    if self.stopped:
                        return
                    epoch, poses, reset_poses = self.pending
                    self.pending = None
                frame = renderer.frame_rgb(poses, reset_poses)
                with self.condition:
                    if epoch == self.epoch:
                        self.serial += 1
                        self.result = (self.serial, frame)
                        self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            if renderer is not None:
                renderer.close()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join()


class ComparisonSession:
    def __init__(self, paths, checkpoints, devices, session_factory=None, policy_factory=None):
        if set(paths) != set(ROBOTS) or not set(ROBOTS).issubset(checkpoints):
            raise ValueError('Comparison requires Q1, Bumi and G1 motions and fixed checkpoints')
        if len(devices) == 1:
            devices = tuple(devices) * 3
        if len(devices) != 3:
            raise ValueError('Comparison requires one device or three devices')
        if session_factory is None:
            from preview.server import PolicySession
            session_factory = PolicySession
        self.devices = tuple(devices)
        self.sessions, self.errors = {}, {}
        self.renderer = self.executor = self.policy = None
        try:
            if policy_factory is None:
                from preview.shared_policy import SharedPolicy
                policy_factory = SharedPolicy
            self.policy = policy_factory(checkpoints, self.devices[0])
            for robot, device in zip(ROBOTS, self.devices):
                self.sessions[robot] = session_factory(robot, paths[robot], checkpoints[robot], device,
                                                      render=False, source_body=True, shared_policy=self.policy)
            if len(set(self.devices)) == 3:
                self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='comparison-physics')
            self.reset()
        except Exception:
            self.close()
            raise

    def reset(self):
        self.errors.clear()
        for child in self.sessions.values():
            child.reset()
        self.policy.calls = 0
        self.steps = 0
        self.failed = self.completed = False
        self.clip_frames = min(child.status()['clip_frames'] for child in self.sessions.values())

    def step(self):
        if self.failed or self.completed:
            return
        try:
            actions = self.policy(self.sessions)
        except Exception as error:
            self.errors.update({r: f'{type(error).__name__}: {error}' for r in ROBOTS})
            self.failed = True
            return
        # Torch CUDA capture is process-wide and explicitly not concurrent.
        # Capture the first step (and the first after reset) in series, then
        # retain the existing parallel replay and three-world barrier.
        needs_capture = any(getattr(child, 'physics_graph', None) is not None
                            and child.physics_graph.graph is None for child in self.sessions.values())
        if self.executor is not None and not needs_capture:
            pending = {robot: self.executor.submit(child.step, actions[robot]) for robot, child in self.sessions.items()}
            for robot, future in pending.items():
                try:
                    future.result()
                except Exception as error:
                    self.errors[robot] = f'{type(error).__name__}: {error}'
        else:
            for robot, child in self.sessions.items():
                try:
                    child.step(actions[robot])
                except Exception as error:
                    self.errors[robot] = f'{type(error).__name__}: {error}'
        self.steps = min(child.steps for child in self.sessions.values())
        self.failed = bool(self.errors) or any(child.failed for child in self.sessions.values())
        holding = all(getattr(child, 'hold_frame', None) is not None for child in self.sessions.values())
        self.completed = not self.failed and ((not holding and self.steps >= self.clip_frames - 1)
                                             or any(child.completed for child in self.sessions.values()))

    def status(self):
        participants = {}
        for robot, child in self.sessions.items():
            participants[robot] = dict(child.status(), robot=robot, device=self.devices[ROBOTS.index(robot)])
            if robot in self.errors:
                participants[robot].update(error=self.errors[robot], failed=True)
        return dict(mode='compare', steps=self.steps, simulated_seconds=self.steps * .02,
                    input_mode=next(iter(participants.values())).get('input_mode'),
                    interface=next(iter(participants.values())).get('interface'),
                    clip_frames=self.clip_frames, failed=self.failed, completed=self.completed,
                    participants=list(participants.values()),
                    failed_robots=[r for r, s in participants.items() if s['failed']],
                    parallel_physics=self.executor is not None,
                    shared_policy=True, inference_batch_size=3, actor_forward_calls=self.policy.calls,
                    control_graph=all(getattr(child, 'physics_graph', None) is not None for child in self.sessions.values()),
                    source_time_seconds=self.steps * .02,
                    source_duration_seconds=(self.clip_frames - 1) * .02,
                    tail_frames={r: s['clip_frames'] - self.clip_frames for r, s in participants.items()},
                    rendering='Common camera; native-size solid policy/physics Q1, Bumi and G1')

    def submit_frame(self, reset=False):
        if self.errors:
            return
        if self.renderer is None:
            self.renderer = AsyncComparisonRenderer(lambda: ComparisonRenderer(self.sessions))
            reset = True
        poses = {}
        for robot, child in self.sessions.items():
            value = child.env.sim.data.qpos[0]
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            poses[robot] = value.copy()
        self.renderer.submit(poses, reset=reset)

    def latest_render(self):
        return self.renderer.latest() if self.renderer is not None else None

    def frame_rgb(self):
        if self.errors:
            raise RuntimeError('Cannot render an incomplete comparison step')
        serial = self.renderer.serial if self.renderer is not None else 0
        self.submit_frame()
        return self.renderer.wait_after(serial)[1]

    def frame(self):
        from PIL import Image
        output = io.BytesIO()
        Image.fromarray(self.frame_rgb()).save(output, format='JPEG', quality=80)
        return output.getvalue()

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        try:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
        finally:
            for child in self.sessions.values():
                child.close()
            self.sessions.clear()
            self.policy = None


class ComparisonRenderer:
    """A shared MjSpec owns all meshes; only actual native qpos enter this model."""
    def __init__(self, sessions):
        import mujoco
        import numpy as np
        from morphmimic.robots import get_robot

        self.mujoco, self.np, self.sessions = mujoco, np, sessions
        self.renderer = None
        spec = mujoco.MjSpec.from_string('''<mujoco model="policy_comparison">
          <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/></visual>
          <asset><texture name="ground" type="2d" builtin="checker" width="128" height="128"
            rgb1=".16 .19 .24" rgb2=".19 .23 .29"/>
            <material name="ground" texture="ground" texrepeat="2 2" texuniform="true"/></asset>
          <worldbody><light pos="0 -3 6" dir="0 .3 -1" diffuse=".8 .8 .8"/>
            <light pos="2 3 5" dir="-.2 -.2 -1" diffuse=".5 .5 .5" castshadow="false"/>
            <geom name="ground" type="plane" size="0 0 .1" group="1" material="ground"/></worldbody>
        </mujoco>''')
        for robot in ROBOTS:
            child = get_robot(robot).entity_cfg_fn().spec_fn()
            # This copy is visual-only; its unused solver globals must not conflict on attach.
            child.option.timestep = spec.option.timestep
            child.njmax, child.nconmax = spec.njmax, spec.nconmax
            spec.attach(child, prefix=robot + '/', frame=spec.worldbody.add_frame())
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.maps, self.roots = {}, {}
        widths = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
                  mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}
        for robot in ROBOTS:
            source = sessions[robot].env.sim.mj_model
            mappings = []
            for joint in range(self.model.njnt):
                name = self.model.joint(joint).name
                if not name.startswith(robot + '/'):
                    continue
                native = name[len(robot) + 1:]
                try:
                    source_joint = source.joint('robot/' + native)
                except KeyError:
                    source_joint = source.joint(native)
                kind = int(self.model.jnt_type[joint])
                if int(source_joint.type[0]) != kind:
                    raise ValueError(f'{robot}: native joint type mismatch for {native}')
                mapping = int(self.model.jnt_qposadr[joint]), int(source_joint.qposadr[0]), widths[kind]
                mappings.append(mapping)
                if kind == mujoco.mjtJoint.mjJNT_FREE:
                    if robot in self.roots:
                        raise ValueError(f'{robot}: comparison requires one floating base')
                    self.roots[robot] = mapping
            if robot not in self.roots:
                raise ValueError(f'{robot}: no native floating base in visual model')
            self.maps[robot] = mappings
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth, self.camera.elevation = 180, -24
        self.camera.distance = 4.4
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = [0, 1, 1, 0, 0, 0]
        self.options.sitegroup[:] = 0

    def reset(self, poses):
        self.origins = {}
        for index, robot in enumerate(ROBOTS):
            _, src, _ = self.roots[robot]
            pose = poses[robot][src:src + 7]
            w, x, y, z = pose[3:]
            yaw = self.np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, s = self.np.cos(yaw), self.np.sin(yaw)
            self.origins[robot] = (pose[:2].copy(), self.np.array([[c, s], [-s, c]]),
                                    self.np.array([self.np.cos(yaw / 2), 0, 0, -self.np.sin(yaw / 2)]),
                                    self.np.array([0, (index - 1) * 2.]))

    def sync(self, poses):
        for robot in ROBOTS:
            qpos = poses[robot]
            if not self.np.isfinite(qpos).all():
                raise ValueError(f'{robot}: nonfinite native state cannot be rendered')
            for dest, src, width in self.maps[robot]:
                self.data.qpos[dest:dest + width] = qpos[src:src + width]
            dest, src, _ = self.roots[robot]
            origin, rotation, yaw_inverse, lane = self.origins[robot]
            self.data.qpos[dest:dest + 2] = rotation @ (qpos[src:src + 2] - origin) + lane
            self.mujoco.mju_mulQuat(self.data.qpos[dest + 3:dest + 7], yaw_inverse, qpos[src + 3:src + 7])
        self.mujoco.mj_forward(self.model, self.data)

    def frame_rgb(self, poses, reset_poses=None):
        if reset_poses is not None:
            self.reset(reset_poses)
        self.sync(poses)
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(self.model, height=720, width=1280)
        positions = self.np.array([self.data.qpos[self.roots[r][0]:self.roots[r][0] + 3] for r in ROBOTS])
        self.camera.lookat[:] = positions.mean(axis=0)
        self.camera.lookat[2] = .65
        spread = self.np.ptp(positions, axis=0)
        self.camera.distance = max(4.4, spread[1] * .95 + .5, spread[0] * 1.8 + 3.)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        # MuJoCo labels follow each actual robot, including locomotion within its lane.
        for robot in ROBOTS:
            scene = self.renderer.scene
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError('Comparison scene has no room for robot labels')
            label = scene.geoms[scene.ngeom]
            root = self.roots[robot][0]
            position = self.data.qpos[root:root + 3].copy()
            position[2] += .85
            self.mujoco.mjv_initGeom(label, self.mujoco.mjtGeom.mjGEOM_LABEL,
                                    self.np.zeros(3), position, self.np.eye(3).ravel(),
                                    self.np.array([1., 1., 1., 1.], dtype=self.np.float32))
            label.label = robot.upper() + ' policy'
            scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
