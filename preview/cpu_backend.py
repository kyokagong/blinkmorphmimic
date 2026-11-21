"""Native MuJoCo CPU physics for the existing fixed-checkpoint preview session.

This is an evaluation-only, single-world backend. It uses the same compiled scene,
native built-in PD, sparse translator and tensor observation/reward functions as
training. It never substitutes reference animation for simulated state.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.scene import Scene
from mjlab.utils.lab_api.math import quat_apply_inverse

from morphmimic.constants import GOAL_K, HUMAN_ENV_DIM
from morphmimic.data.manifest import load_subset
from morphmimic.data.motion_library import MotionLibrary
from morphmimic.model.human import pack_human_goal
from morphmimic.model.rewards_torch import analytic_obs, goal_features, termination_features, tracking_reward
from morphmimic.model.state import StateParts, anchor_quat, join_state
from morphmimic.robots import get_robot


class CpuPhysicsDiagnostics:
    """Sticky native-CPU warnings and explicit capacities, never Warp flag claims."""
    def __init__(self, env):
        self.env = env
        self.bits = self.steps = self.forwards = self.flagged = 0
        self.names = {1: 'CONSTRAINT_CAPACITY', 2: 'CONTACT_CAPACITY', 4: 'NATIVE_WARNING', 8: 'NONFINITE'}
        self.warnings = set()

    def sample(self, *, forward=False):
        d, cfg = self.env.native_data, self.env.cfg
        bits = int(d.nefc > cfg.sim.njmax) | (2 * int(d.ncon > cfg.sim.nconmax))
        for index, warning in enumerate(d.warning):
            if warning.number:
                self.warnings.add(mujoco.mjtWarning(index).name)
                bits |= 4
        if not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.qacc, d.ctrl)):
            bits |= 8
        self.bits |= bits
        self.flagged += int(bool(bits))
        self.forwards += int(forward)
        self.steps += int(not forward)
        if bits:
            raise RuntimeError(f'Native CPU physics failure before publication: {self.report()}')

    def report(self):
        return dict(backend='mujoco_cpu', scope='current_process; preserved across resets',
                    overflow_bits=self.bits, overflow_flags=[v for k, v in self.names.items() if self.bits & k],
                    native_warnings=sorted(self.warnings), checked_physics_steps=self.steps,
                    checked_forward_calls=self.forwards, flagged_step_or_forward_observations=self.flagged,
                    constraint_capacity=self.env.cfg.sim.njmax, contact_capacity=self.env.cfg.sim.nconmax,
                    coverage='Native CPU warnings/nonfinite, raw nefc and ncon; no Warp CSR/global broadphase counters',
                    zero_overflow_observed=self.bits == 0)


class CpuCommand:
    def __init__(self, env, library, cfg):
        self.env, self.library, self.cfg = env, library, cfg
        self.global_frame = torch.zeros(1, dtype=torch.long, device=env.device)

    def assign_clips(self, env_ids, clip_ids):
        if tuple(env_ids.shape) != (1,) or tuple(clip_ids.shape) != (1,) or int(env_ids[0]) != 0 or int(clip_ids[0]) != 0:
            raise ValueError('CPU preview requires its single selected clip/world')
        self.global_frame.zero_()

    def _debug_vis_impl(self, *args, **kwargs):
        # CpuRenderer adds the reference ghost using the native model directly.
        pass


class CpuPreviewEnv:
    """Small PolicySession-compatible native simulator; device selects actor tensors."""
    def __init__(self, cfg, device='cpu'):
        if torch.device(device).type != 'cpu':
            raise ValueError('CPU native physics currently requires CPU storage')
        self.cfg, self.device, self.num_envs = cfg, torch.device(device), 1
        command = cfg.commands['motion']
        self.spec = get_robot(command.robot_name)
        entity = cfg.scene.entities['robot']
        if cfg.scene.num_envs != 1 or cfg.auto_reset or cfg.events:
            raise ValueError('CPU preview requires one eval world, no DR/events and manual resets')
        if command.pose_range or command.velocity_range or tuple(command.joint_position_range) != (0., 0.):
            raise ValueError('CPU preview requires deterministic reference resets')
        if set(cfg.terminations) != {'time_out', 'anchor_pos', 'anchor_ori', 'clip_end'}:
            raise ValueError('Unsupported CPU preview termination terms')
        for actuator in entity.articulation.actuators:
            if type(actuator) is not BuiltinPositionActuatorCfg or actuator.delay_max_lag != 0:
                raise ValueError('CPU preview supports the registered native built-in PD eval actuators without delay')
        action_cfg = cfg.actions['joint_pos']
        if action_cfg.clip is not None or action_cfg.noise_std != 0.:
            raise ValueError('CPU preview requires unclipped, noise-free canonical actions')
        scene = Scene(cfg.scene, device='cpu')
        model = scene.compile()
        cfg.sim.mujoco.apply(model)
        self.native_data = mujoco.MjData(model)
        self.step_dt = float(model.opt.timestep) * cfg.decimation
        if abs(self.step_dt - .02) > 1e-10:
            raise ValueError('CPU preview requires the native 50Hz control interval')
        if model.nkey != 1:
            raise ValueError('Expected one native initial-state keyframe')
        joints = [model.joint('robot/' + n).id for n in self.spec.joint_names]
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError('CPU preview native actuator mapping requires hinge joints')
        self.joint_qadr, self.joint_dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        native_index = {joint: index for index, joint in enumerate(joints)}
        self.actuator_native = np.array([native_index[int(x)] for x in model.actuator_trnid[:, 0]])
        if (model.nu != len(joints) or len(set(self.actuator_native.tolist())) != len(joints)
                or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)):
            raise ValueError('CPU preview requires one native built-in position actuator per joint')
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or model.jnt_qposadr[free[0]] != 0 or model.jnt_dofadr[free[0]] != 0:
            raise ValueError('CPU preview requires one leading floating base')
        self.body_ids = np.array([model.body('robot/' + name).id for name in self.spec.tracked_bodies])
        self._jp, self._jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        self.default_joint_pos = self.tensor(model.key_qpos[0, self.joint_qadr])
        native_scales = []
        for name in self.spec.joint_names:
            matches = [value for pattern, value in self.spec.action_scale.items() if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f'Ambiguous native action scale: {name}')
            native_scales.append(matches[0])
        tr = self.spec.translator
        self.action_scale = self.tensor([native_scales[i] if i >= 0 else 0. for i in tr.perm])
        self.joint_mask = torch.tensor(tr.joint_mask, device=self.device)
        limits = self.tensor(model.jnt_range[joints])
        middle = limits.mean(-1)
        radius = (limits[:, 1] - limits[:, 0]) * entity.articulation.soft_joint_pos_limit_factor / 2
        self.native_joint_limits = torch.stack((middle - radius, middle + radius), -1)
        self.joint_limits = tr.limits_to_canonical(self.native_joint_limits)
        self.last_action = torch.zeros(1, 29, device=self.device)
        library = MotionLibrary(load_subset(command.subset_yaml), device=str(self.device),
            dtype=getattr(torch, command.library_dtype), robot=self.spec,
            human_root=command.human_root, allow_missing_human=command.allow_missing_human)
        if len(library.clip_names) != 1:
            raise ValueError('CPU preview loads exactly one selected clip')
        self.command = CpuCommand(self, library, command)
        self.command_manager = SimpleNamespace(get_term=lambda name: self.command if name == 'motion' else None)
        self._terms = {name: torch.zeros(1, dtype=torch.bool, device=self.device) for name in cfg.terminations}
        self.termination_manager = SimpleNamespace(active_terms=list(self._terms), get_term=self._terms.__getitem__)
        # Comparison consumes copied qpos from this surface; the native data owns storage.
        self.sim = SimpleNamespace(mj_model=model, data=SimpleNamespace(qpos=torch.from_numpy(self.native_data.qpos).unsqueeze(0)),
                                   forward=self.forward)
        self.physics_diagnostics = CpuPhysicsDiagnostics(self)
        self.scene = scene
        self.closed = False
        self._done = False

    def tensor(self, array):
        return torch.tensor(np.asarray(array).copy(), dtype=torch.float32, device=self.device)

    def forward(self):
        mujoco.mj_forward(self.sim.mj_model, self.native_data)
        self.physics_diagnostics.sample(forward=True)

    def state(self):
        m, d = self.sim.mj_model, self.native_data
        linear, angular = [], []
        for body in self.body_ids:
            # Body-origin Jacobians avoid confusing COM cvel with link-origin velocity.
            mujoco.mj_jacBody(m, d, self._jp, self._jr, int(body))
            linear.append(self._jp @ d.qvel)
            angular.append(self._jr @ d.qvel)
        return join_state(StateParts(
            self.spec.translator.to_canonical(self.tensor(d.qpos[self.joint_qadr])).unsqueeze(0),
            self.spec.translator.to_canonical_delta(self.tensor(d.qvel[self.joint_dadr])).unsqueeze(0),
            self.tensor(d.xpos[self.body_ids]).unsqueeze(0), self.tensor(d.xquat[self.body_ids]).unsqueeze(0),
            self.tensor(linear).unsqueeze(0), self.tensor(angular).unsqueeze(0)))

    def observations(self):
        state = self.state()
        lib, frame = self.command.library, self.command.global_frame
        policy = analytic_obs(state, torch.zeros(1, 29, device=self.device),
            self.spec.translator.to_canonical(self.default_joint_pos), self.last_action,
            imu_idx=self.spec.imu_slot, joint_mask=self.joint_mask)
        goal = goal_features(state, lib.future(frame, GOAL_K), self.command.cfg.goal_anchor)
        human = torch.zeros(1, HUMAN_ENV_DIM, device=self.device)
        if lib.has_human:
            window = lib.human_future(frame, GOAL_K)
            human = pack_human_goal(window[..., :3], window[..., 3:], anchor_quat(state))
        return {'state': state, 'policy': policy, 'policy_clean': policy.clone(), 'goal': goal, 'goal_kp': human}

    def reference_qpos(self, frame):
        lib = self.command.library
        index = torch.tensor([frame], dtype=torch.long, device=self.device)
        qpos = self.sim.mj_model.key_qpos[0].copy()
        qpos[:3] = lib.root_pos_w[index][0].float().cpu().numpy()
        qpos[3:7] = lib.root_quat_w[index][0].float().cpu().numpy()
        qpos[self.joint_qadr] = self.spec.translator.to_native(lib.joint_pos[index].float(), self.default_joint_pos)[0].cpu().numpy()
        return qpos

    def reset(self, *, env_ids=None):
        if self.closed:
            raise RuntimeError('CPU preview environment is closed')
        if env_ids is not None and (tuple(env_ids.shape) != (1,) or int(env_ids[0]) != 0):
            raise ValueError('CPU preview reset requires world0')
        m, d, lib = self.sim.mj_model, self.native_data, self.command.library
        mujoco.mj_resetData(m, d)
        self._done = False
        d.qpos[:] = self.reference_qpos(0)
        # Match MultiMotionCommand._write_reference_state_to_sim: clip only the
        # reset state; the unmodified reference remains available for rendering.
        limits = self.native_joint_limits.numpy()
        d.qpos[self.joint_qadr] = np.clip(d.qpos[self.joint_qadr], limits[:, 0], limits[:, 1])
        d.qvel[:3] = lib.root_lin_vel_w[0].float().cpu().numpy()
        d.qvel[3:6] = quat_apply_inverse(lib.root_quat_w[:1].float(), lib.root_ang_vel_w[:1].float())[0].cpu().numpy()
        d.qvel[self.joint_dadr] = self.spec.translator.to_native_delta(lib.joint_vel[:1].float())[0].cpu().numpy()
        d.ctrl[:] = d.qpos[self.joint_qadr][self.actuator_native]
        self.last_action.zero_()
        for term in self._terms.values():
            term.zero_()
        self.command.global_frame.fill_(min(1, int(lib.clip_len[0]) - 1))
        self.forward()
        return self.observations(), {}

    def step(self, action):
        if self.closed:
            raise RuntimeError('CPU preview environment is closed')
        if self._done:
            raise RuntimeError('CPU preview world must be reset after completion or failure')
        if tuple(action.shape) != (1, 29) or not bool(torch.isfinite(action).all()):
            raise ValueError('CPU preview action must be a finite [1,29] tensor')
        action = action.detach().to(self.device) * self.joint_mask
        previous = self.last_action.clone()
        targets = self.default_joint_pos + self.spec.translator.to_native_delta(action * self.action_scale)
        self.native_data.ctrl[:] = targets[0, self.actuator_native].cpu().numpy()
        for _ in range(self.cfg.decimation):
            mujoco.mj_step(self.sim.mj_model, self.native_data)
            self.physics_diagnostics.sample()
        self.forward()
        state = self.state()
        lib, frame = self.command.library, self.command.global_frame
        ref = lib.state(frame)
        reward, _ = tracking_reward(state, ref, action, previous, self.joint_limits, joint_mask=self.joint_mask)
        height, tilt = termination_features(state, ref)
        self._terms['anchor_pos'].copy_(height > self.cfg.terminations['anchor_pos'].params['threshold'])
        self._terms['anchor_ori'].copy_(tilt > self.cfg.terminations['anchor_ori'].params['threshold'])
        self._terms['clip_end'].copy_(lib.remaining(frame) == 0)
        terminated = self._terms['anchor_pos'] | self._terms['anchor_ori']
        truncated = self._terms['clip_end'].clone()
        self._done = bool((terminated | truncated)[0])
        self.last_action.copy_(action)
        frame.copy_(torch.minimum(frame + 1, lib.clip_end(frame)))
        return self.observations(), reward, terminated, truncated, {}

    def close(self):
        self.closed = True


class CpuRenderer:
    """Native renderer for a single CPU session, created/used by its worker thread."""
    def __init__(self, env, width=960, height=640):
        self.env = env
        model = env.sim.mj_model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.ghost = mujoco.MjData(model)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.elevation, self.camera.azimuth = 3., -20., 135.
        self.options = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.options.geomgroup[:] = [0, 1, 1, 0, 0, 0]

    def update(self, *args, **kwargs):
        pass

    def render(self):
        env, renderer = self.env, self.renderer
        self.camera.lookat[:] = env.native_data.xpos[env.body_ids[7]]
        renderer.update_scene(env.native_data, camera=self.camera, scene_option=self.options)
        self.ghost.qpos[:] = env.reference_qpos(int(env.command.global_frame[0]))
        mujoco.mj_forward(env.sim.mj_model, self.ghost)
        start = renderer.scene.ngeom
        mujoco.mjv_addGeoms(env.sim.mj_model, self.ghost, self.options, self.perturb,
                           mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
        for geom in renderer.scene.geoms[start:renderer.scene.ngeom]:
            geom.rgba[:] = [.25, .7, .95, .25]
        return renderer.render().copy()

    def close(self):
        self.renderer.close()
