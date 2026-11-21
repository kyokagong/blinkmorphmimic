"""Export one fixed Q1/Bumi/G1 motion checkpoint with embedded robot constants."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
from torch import nn

from morphmimic.constants import DT, GOAL_DIM, GOAL_VERSION, JOINT_NAMES, PROPRIO_DIM
from morphmimic.model.policy import Policy, PolicyCfg
from morphmimic.robots import check_imu_body, get_robot
from morphmimic.train.normalizers import RunningNormalizer, normalize_history


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}\n'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FixedRobotActor(nn.Module):
    def __init__(self, policy: Policy, obs_norm: RunningNormalizer, goal_norm: RunningNormalizer):
        super().__init__()
        self.policy, self.obs_norm, self.goal_norm = policy, obs_norm, goal_norm

    def forward(self, obs_history: torch.Tensor, motion_goal: torch.Tensor) -> torch.Tensor:
        return self.policy(normalize_history(self.obs_norm, obs_history), self.goal_norm(motion_goal))


def load_actor(checkpoint: Path, weights: str = 'ema') -> tuple[FixedRobotActor, dict]:
    """Load a trusted, numbered checkpoint on CPU; never silently fall back from EMA."""
    checkpoint = Path(checkpoint)
    match = re.fullmatch(r'ckpt_(\d+)\.pt', checkpoint.name)
    if checkpoint.is_symlink() or match is None:
        raise ValueError('use a fixed ckpt_<iteration>.pt file, not latest or a symlink')
    if weights not in ('ema', 'online'):
        raise ValueError('weights must be ema or online')
    before = _sha256(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if _sha256(checkpoint) != before:
        raise ValueError('checkpoint changed while being read')
    if int(match[1]) != ckpt['iteration']:
        raise ValueError('checkpoint filename and saved iteration disagree')
    if ckpt.get('goal_version') != GOAL_VERSION:
        raise ValueError(f'expected position-free goal version {GOAL_VERSION}')
    cfg = ckpt['cfg']
    if cfg.get('policy_goal', 'motion') != 'motion' or cfg.get('kp_interface', False):
        raise ValueError('this exporter supports the motion-only policy interface')
    if cfg.get('goal_anchor', 'reference') != 'reference':
        raise ValueError('this exporter requires the position-free reference goal anchor')
    check_imu_body(ckpt, checkpoint)
    key = 'ema_policy' if weights == 'ema' else 'policy'
    if key not in ckpt:
        raise ValueError('checkpoint has no EMA policy; request weights=online explicitly')
    state, morph = ckpt[key], ckpt['morph']
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('policy contains nonfinite values')
    spec = get_robot(cfg['robot'])
    mask = torch.tensor(spec.translator.joint_mask, dtype=torch.float32)
    if not torch.equal(state['decoder.joint_mask'], mask):
        raise ValueError('checkpoint joint mask differs from the robot asset')
    actor_morph = state['decoder.morph_actor']
    if actor_morph.numel() not in (145, 154) or not torch.equal(actor_morph[-29:], mask):
        raise ValueError('checkpoint morphology joint mask differs from the robot asset')
    if not torch.equal(actor_morph, morph['actor']):
        raise ValueError('checkpoint morphology differs from selected policy buffers')
    from morphmimic.robots.morphology import morphology_vector
    current_morph = morphology_vector(spec.name, actor=True, actor_geometry=actor_morph.numel() == 154)
    if not torch.allclose(actor_morph, current_morph, atol=1e-6, rtol=0):
        raise ValueError('checkpoint morphology differs from current robot assets/controls')
    policy = Policy(PolicyCfg(enc_hidden=tuple(cfg['policy_hidden']), dec_hidden=tuple(cfg['policy_hidden']),
                              latent=int(cfg.get('latent', 32)), act=cfg['act'], hist_len=int(cfg['hist_len']),
                              morph_dim=actor_morph.numel(), film=bool(morph['film']), bf16=False))
    policy.load_state_dict(state)
    norms = []
    for key, dim in (('obs_norm', PROPRIO_DIM), ('goal_norm', GOAL_DIM)):
        norm = RunningNormalizer(dim)
        norm.load_state_dict(ckpt[key])
        if not torch.isfinite(norm.mean).all() or not torch.isfinite(norm.var).all() or (norm.var < 0).any():
            raise ValueError(f'invalid normalizer {key}')
        norms.append(norm)
    ckpt['_checkpoint_sha256'] = before
    return FixedRobotActor(policy, *norms).eval(), ckpt


def _metadata(model: FixedRobotActor, ckpt: dict, checkpoint: Path, weights: str) -> dict:
    import mujoco
    from mjlab.utils.string import resolve_field
    from morphmimic.robots.morphology import _actuator_terms

    spec = get_robot(ckpt['cfg']['robot'])
    entity = spec.entity_cfg_fn()
    asset_model = entity.spec_fn().compile()
    buffer = np.empty(mujoco.mj_sizeModel(asset_model), dtype=np.uint8)
    mujoco.mj_saveModel(asset_model, buffer=buffer)
    tr = spec.translator
    controls = _actuator_terms(spec)
    native = {}
    for key, value in controls.items():
        values = np.zeros(tr.num_native)
        for c, n in enumerate(tr.perm):
            if n >= 0:
                values[n] = value[c]
        native[key] = values.tolist()
    # The checkpoint default is the exact pose used by the trained actor's observations.
    default_canonical = ckpt['wm']['default_joint_pos'].float()
    default = tr.to_native(default_canonical, torch.zeros(tr.num_native))
    if not torch.isfinite(default).all():
        raise ValueError('nonfinite default pose')
    configured_default = torch.tensor(resolve_field(entity.init_state.joint_pos, spec.joint_names, 0.), dtype=torch.float32)
    if not torch.allclose(default, configured_default, atol=1e-6, rtol=0):
        raise ValueError('checkpoint default pose differs from current robot configuration')
    native['default_joint_pos'] = default.tolist()
    state = model.policy.state_dict()
    shared = {k: v for k, v in state.items() if k not in ('decoder.morph_actor', 'decoder.joint_mask')}
    return dict(
        schema_version=1, robot=spec.name, iteration=int(ckpt['iteration']), weights=weights,
        checkpoint=dict(path=str(checkpoint.resolve()), sha256=ckpt['_checkpoint_sha256']),
        shared_policy_sha256=_state_sha256(shared), policy_with_robot_sha256=_state_sha256(state),
        goal_version=GOAL_VERSION, goal_anchor='reference', control_hz=int(1 / DT),
        training_robot_contract=ckpt.get('robot_contract'),
        history_length=model.policy.cfg.hist_len, history_order='oldest first',
        inputs={'obs_history': ['B', model.policy.cfg.hist_len * PROPRIO_DIM], 'motion_goal': ['B', GOAL_DIM]},
        output={'canonical_action': ['B', 29]}, dtype='float32',
        canonical_joint_order=list(JOINT_NAMES), native_joint_order=list(spec.joint_names),
        joint_mask=list(tr.joint_mask), morphology=state['decoder.morph_actor'].tolist(),
        native_mapping=dict(perm=list(tr.perm), sign=list(tr.sign), offset=list(tr.offset)),
        native_control=native,
        native_target_formula='target[perm[c]] = default_joint_pos[perm[c]] + sign[c] * action[c] * action_scale[perm[c]] for perm[c] >= 0',
        default_canonical_joint_pos=default_canonical.tolist(),
        normalizers={name: dict(mean=norm.mean.tolist(), var=norm.var.tolist(), count=int(norm.count), eps=norm.eps,
                               formula='(x - mean) / (sqrt(var) + eps)')
                     for name, norm in (('obs', model.obs_norm), ('goal', model.goal_norm))},
        imu_body=spec.imu_body, anchor_body=spec.anchor_body,
        physics_timestep=spec.physics_timestep, physics_decimation=spec.physics_decimation,
        training_actuator_delay=bool(ckpt['cfg'].get('actuator_delay', False)),
        asset=dict(compiled_mjb_sha256=hashlib.sha256(buffer.tobytes()).hexdigest(),
                   mujoco_version=mujoco.__version__, adapter_source=str(Path(inspect.getfile(spec.entity_cfg_fn))),
                   adapter_source_sha256=_sha256(Path(inspect.getfile(spec.entity_cfg_fn)))),
        execution_model='nominal ideal position PD; no hardware qualification',
        precision='float32 CPU inference, matching evaluation; training bf16 autocast is not exported',
    )


def export(checkpoint: Path, output: Path, weights: str = 'ema') -> dict:
    """Write a new verified bundle. Existing output directories are never overwritten."""
    import onnx
    import onnxruntime as ort

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    model, ckpt = load_actor(checkpoint, weights)
    metadata = _metadata(model, ckpt, Path(checkpoint), weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.export-', dir=output.parent) as temp:
        stage = Path(temp)
        hdim = model.policy.cfg.hist_len * PROPRIO_DIM
        args = (torch.zeros(2, hdim), torch.zeros(2, GOAL_DIM))
        graph = stage / 'policy.onnx'
        with torch.no_grad():
            batch = torch.export.Dim('batch')
            program = torch.onnx.export(model, args, opset_version=18, dynamo=True, external_data=False,
                                        input_names=['obs_history', 'motion_goal'], output_names=['canonical_action'],
                                        dynamic_shapes=({0: batch}, {0: batch}))
        for node in program.model.graph.all_nodes():
            node.metadata_props.pop('pkg.torch.onnx.stack_trace', None)
        program.save(str(graph), external_data=False)
        onnx.checker.check_model(onnx.load(str(graph)))
        session = ort.InferenceSession(str(graph), providers=['CPUExecutionProvider'])
        rng = np.random.default_rng(0)
        errors = {}
        mask = np.array(metadata['joint_mask'], dtype=bool)
        for b in (1, 4, 32):
            h = rng.normal(size=(b, hdim)).astype('float32')
            g = rng.normal(size=(b, GOAL_DIM)).astype('float32')
            got = session.run(None, dict(obs_history=h, motion_goal=g))[0]
            with torch.no_grad():
                want = model(torch.from_numpy(h), torch.from_numpy(g)).numpy()
            error = float(np.max(np.abs(got - want)))
            if not np.isfinite(got).all() or error > 1e-4 or np.count_nonzero(got[:, ~mask]):
                raise ValueError(f'ONNX parity or missing-slot check failed: batch={b}, error={error}')
            errors[str(b)] = error
        metadata['parity'] = dict(provider='CPUExecutionProvider', max_abs_error=errors, absolute_tolerance=1e-4,
                                  scope='deterministic random feature inputs; not learned tracking quality')
        metadata['onnx_sha256'] = _sha256(graph)
        (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        (stage / 'SHA256SUMS').write_text(''.join(f'{_sha256(stage / name)}  {name}\n' for name in ('policy.onnx', 'metadata.json')))
        stage.rename(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted fixed ckpt_<iteration>.pt; no latest aliases')
    parser.add_argument('--output', type=Path, required=True, help='New per-robot bundle directory')
    parser.add_argument('--weights', choices=('ema', 'online'), default='ema')
    args = parser.parse_args()
    result = export(args.checkpoint, args.output, args.weights)
    print(json.dumps(dict(robot=result['robot'], iteration=result['iteration'], weights=result['weights'], parity=result['parity'])))


if __name__ == '__main__':
    main()
