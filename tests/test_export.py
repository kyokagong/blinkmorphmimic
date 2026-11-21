"""Fixed-robot exports must preserve the checkpoint's action path and identity."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from morphmimic.model.policy import Policy, PolicyCfg
from morphmimic.train.normalizers import RunningNormalizer, normalize_history


def exporter():
    path = Path(__file__).parents[1] / 'src/morphmimic/scripts/export.py'
    assert path.exists(), 'fixed-robot export is not implemented'
    from morphmimic.scripts import export
    return export


def test_export_rejects_mutable_latest_and_symlinks(tmp_path):
    latest = tmp_path / 'latest.pt'
    latest.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='fixed|ckpt_'):
        exporter().load_actor(latest)
    alias = tmp_path / 'ckpt_1.pt'
    alias.symlink_to(latest)
    with pytest.raises(ValueError, match='symlink|fixed'):
        exporter().load_actor(alias)


@pytest.fixture
def small_checkpoint(tmp_path):
    pytest.importorskip('mjlab')
    from morphmimic.robots import get_robot
    from morphmimic.robots.morphology import morphology_vector
    spec = get_robot('q1')
    torch.manual_seed(10)
    policy = Policy(PolicyCfg(enc_hidden=(8,), dec_hidden=(8,), latent=4,
                              hist_len=2, morph_dim=154, film=True))
    policy.set_joint_mask(torch.tensor(spec.translator.joint_mask))
    policy.set_morphology(morphology_vector('q1', actor=True))
    obs, goal = RunningNormalizer(93), RunningNormalizer(670)
    obs.update(torch.randn(10, 93) * 2 + .5)
    goal.update(torch.randn(10, 670) * 3 - .3)
    online = {k: v.clone() for k, v in policy.state_dict().items()}
    online['decoder.net.2.bias'].add_(1.)
    default = torch.tensor([j['home_rad'] for j in __import__('morphmimic.robots.q1', fromlist=['profile']).profile()['joints']])
    ckpt = dict(cfg=dict(robot='q1', hist_len=2, policy_hidden=(8,), latent=4,
                        act='elu', goal_anchor='reference', policy_goal='motion'),
                iteration=7, goal_version=2, imu_body='pelvis', policy=online,
                ema_policy=policy.state_dict(), obs_norm=obs.state_dict(), goal_norm=goal.state_dict(),
                morph=dict(actor=policy.decoder.morph_actor.clone(), film=True),
                wm={'default_joint_pos': spec.translator.to_canonical(default)})
    path = tmp_path / 'ckpt_7.pt'
    torch.save(ckpt, path)
    return path, policy.eval(), obs.eval(), goal.eval()


def test_actor_uses_ema_by_default_and_online_only_when_requested(small_checkpoint):
    path, expected, obs, goal = small_checkpoint
    actor, _ = exporter().load_actor(path)
    online, _ = exporter().load_actor(path, weights='online')
    h, g = torch.randn(3, 186), torch.randn(3, 670)
    with torch.no_grad():
        want = expected(normalize_history(obs, h), goal(g))
        torch.testing.assert_close(actor(h, g), want)
        mask = expected.decoder.joint_mask.bool()
        torch.testing.assert_close(online(h, g)[:, mask] - want[:, mask], torch.ones(3, 22))
        assert torch.count_nonzero(actor(h, g)[:, ~mask]) == 0


def test_export_requires_ema_and_consistent_physical_identity(small_checkpoint):
    path, _, _, _ = small_checkpoint
    ckpt = torch.load(path, weights_only=False)
    del ckpt['ema_policy']
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='EMA|ema'):
        exporter().load_actor(path)
    exporter().load_actor(path, weights='online')
    ckpt['policy']['decoder.joint_mask'][14] = 1.
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='mask'):
        exporter().load_actor(path, weights='online')


def test_export_refuses_asset_controls_that_differ_from_saved_morphology(small_checkpoint):
    path, _, _, _ = small_checkpoint
    ckpt = torch.load(path, weights_only=False)
    ckpt['ema_policy']['decoder.morph_actor'][0] += .1
    ckpt['morph']['actor'] = ckpt['ema_policy']['decoder.morph_actor'].clone()
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='morphology|asset'):
        exporter().load_actor(path)


def assert_bundle(path, output, native_count, weights='ema'):
    ort = pytest.importorskip('onnxruntime')
    result = exporter().export(path, output, weights=weights)
    doc = json.loads((output / 'metadata.json').read_text())
    assert doc == result
    assert doc['weights'] == weights
    assert doc['checkpoint']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc['onnx_sha256'] == hashlib.sha256((output / 'policy.onnx').read_bytes()).hexdigest()
    assert len(doc['native_joint_order']) == native_count
    assert sum(doc['joint_mask']) == native_count
    for key in ('kp', 'kd', 'effort', 'action_scale', 'default_joint_pos'):
        assert len(doc['native_control'][key]) == native_count
    assert len(doc['morphology']) == 154
    assert len(doc['normalizers']['obs']['mean']) == 93
    assert len(doc['normalizers']['goal']['mean']) == 670
    assert len(doc['asset']['compiled_mjb_sha256']) == 64
    for line in (output / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ')
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    session = ort.InferenceSession(str(output / 'policy.onnx'), providers=['CPUExecutionProvider'])
    assert [v.name for v in session.get_inputs()] == ['obs_history', 'motion_goal']
    assert [v.name for v in session.get_outputs()] == ['canonical_action']
    model, _ = exporter().load_actor(path, weights=weights)
    mask = np.array(doc['joint_mask'], dtype=bool)
    rng = np.random.default_rng(42)
    for b in (1, 5):
        h = rng.normal(size=(b, model.policy.cfg.hist_len * 93)).astype('float32')
        g = rng.normal(size=(b, 670)).astype('float32')
        actual = session.run(None, dict(obs_history=h, motion_goal=g))[0]
        with torch.no_grad():
            expected = model(torch.from_numpy(h), torch.from_numpy(g)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        assert np.isfinite(actual).all()
        assert np.count_nonzero(actual[:, ~mask]) == 0
        h.reshape(b, -1, 93)[..., np.r_[np.flatnonzero(~mask), np.flatnonzero(~mask)+29, np.flatnonzero(~mask)+64]] = 999
        g.reshape(b, 10, 67)[..., np.r_[np.flatnonzero(~mask), np.flatnonzero(~mask)+29]] = -999
        np.testing.assert_array_equal(session.run(None, dict(obs_history=h, motion_goal=g))[0], actual)
    assert max(doc['parity']['max_abs_error'].values()) <= 1e-4
    return doc


def test_small_export_bundle_parity_and_no_overwrite(small_checkpoint, tmp_path):
    pytest.importorskip('onnxruntime')
    path, _, _, _ = small_checkpoint
    output = tmp_path / 'export'
    doc = assert_bundle(path, output, 22)
    assert doc['control_hz'] == 50
    assert doc['native_mapping']['perm'][12:15] == [13, 12, -1]
    with pytest.raises(FileExistsError):
        exporter().export(path, output)


@pytest.mark.parametrize(('robot', 'native_count'), [('q1', 22), ('bumi', 21), ('g1', 29)])
def test_pinned_training_checkpoint_export(robot, native_count, tmp_path):
    root = Path(os.environ.get('BLINKMORPH_EXPORT_CHECKPOINT_ROOT', '/data/kk/blinkmorphmimic/runs/shared-v1'))
    path = root / robot / 'ckpt_1000.pt'
    if not path.is_file():
        pytest.skip('requires the pinned shared-v1 ckpt_1000 artifacts')
    assert_bundle(path, tmp_path / robot, native_count)
