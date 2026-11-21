"""Check fixed identities for legacy and jointly trained human releases."""
import re


def verify_human_identity(ckpt, *, robot, update, data_sha256, training_sha256=None):
    if training_sha256 is None:
        from morphmimic.train.human_adapt import verify_adaptation_identity
        return verify_adaptation_identity(ckpt, robot=robot, update=update, data_sha256=data_sha256)
    cfg = ckpt.get('cfg', {})
    identity = ckpt.get('optimized_training', {})
    if (not isinstance(training_sha256, str) or not re.fullmatch('[0-9a-f]{64}', training_sha256)
        or cfg.get('robot') != robot or ckpt.get('iteration') != update
        or cfg.get('optimization_schema') != 1 or cfg.get('kp_interface') is not True
        or cfg.get('kp_fixed_horizon') != 1 or cfg.get('policy_goal') != 'motion'
        or cfg.get('optimizer') != 'muon' or identity.get('schema_version') != 1
        or identity.get('iteration') != update or identity.get('optimizer') != 'muon'
        or type(cfg.get('num_envs')) is not int or cfg['num_envs'] not in (2048, 4096)
        or type(identity.get('num_envs')) is not int or identity['num_envs'] != cfg['num_envs']
        or identity.get('joint_motion_human') is not True
        or identity.get('human_horizon') != 1 or identity.get('human_points') != 'five'
        or identity.get('prepared_manifest_sha256') != training_sha256):
        raise ValueError('Joint checkpoint identity differs from its fixed robot, iteration, training mode or manifest')
