"""Reference migrations retain checkpoint provenance and neutral bookends."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from preview.server import Catalog, PolicySession, ROBOTS, load_release_catalogs, sha256


def reference_data(tmp_path, in_place=False):
    import os
    import shutil
    from preview.reference import source_slice_digest
    try:
        import mujoco
        from test_neutral_bookends import fixture
        from morphmimic.data.neutral_bookends import neutralize_motion
        from morphmimic.data.local_prepare import _robot_model
        _robot_model('q1')
    except (ImportError, FileNotFoundError):
        pytest.skip('requires frozen native robot assets')
    root = tmp_path / 'reference'
    root.mkdir()
    rows = []
    repository = Path(__file__).resolve().parents[1]
    review_source = repository / 'prepared/inplace-recovery-neutral-20261117-v2/standing-review.json'
    if not review_source.is_file():
        review_source = repository / 'runs/reports/neutral-standing-review-20261116/review-manifest.json'
    reviewed = json.loads(review_source.read_text())
    review = dict(prepared_frame_zero_based=11446, prepared_fps=50, robot_sources={})
    cache = tmp_path.parent / 'neutral-preview-target-cache'
    cache.mkdir(exist_ok=True)
    for robot in ROBOTS:
        source, target, _ = fixture(robot, speed=1.)
        out, contract = neutralize_motion(source, target, robot, target_frame=0, in_place=in_place)
        contract['target_frame'] = 11446
        contract['source_slice_sha256'] = source_slice_digest(out, contract['source_start_frame'], contract['source_stop_frame'])
        original = Path(reviewed['robot_sources'][robot]['npz'])
        cached = cache / (robot + '-' + reviewed['robot_sources'][robot]['sha256'] + '.npz')
        if not cached.exists():
            shutil.copyfile(original, cached)
        archive = root / 'provenance' / robot / 'walk2_subject4.npz'
        archive.parent.mkdir(parents=True)
        os.link(cached, archive)
        review['robot_sources'][robot] = dict(npz=str(archive), sha256=sha256(archive))
        (root / robot).mkdir()
        for name, human_available in (('dance', True), ('flip', True), ('motion_only', False)):
            path = root / robot / (name + '.npz')
            np.savez(path, **out)
            frames = len(out['joint_pos'])
            row = dict(robot=robot, name=name, source_id='source/' + name, split='train',
                       frames=frames, output=str(path), output_sha256=sha256(path),
                       human_available=human_available, phases=dict(entry=.08, dance=.16),
                       neutral_contract=dict(contract))
            if human_available:
                side = root / 'human50' / robot / (name + '.npz')
                side.parent.mkdir(parents=True, exist_ok=True)
                np.savez(side, fps=50, human_pos_w=np.zeros((frames, 24, 3)),
                         human_quat_w=np.tile([1., 0., 0., 0.], (frames, 24, 1)))
                with np.load(side) as data:
                    human_digest = source_slice_digest(data, contract['source_start_frame'], contract['source_stop_frame'])
                row.update(human_sidecar=str(side), human_sha256=sha256(side), human_source_slice_sha256=human_digest)
            rows.append(row)
    review_path = root / 'standing-review.json'
    review_path.write_text(json.dumps(review))
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps(dict(complete=True, human_data=True, human_root=str(root / 'human50'), clips=rows,
        neutral_review=dict(path='standing-review.json', sha256=sha256(review_path)))))
    training = tmp_path / 'immutable-training.json'
    training.write_text(json.dumps(dict(clips=[dict(robot='q1', name='dance', output='/obsolete/reference.npz')])))
    return root, manifest, training, rows


def reference_catalog(tmp_path, interface='motion', in_place=False):
    root, manifest, training, rows = reference_data(tmp_path, in_place=in_place)
    catalog = Catalog(root, {r: object() for r in ROBOTS}, interface=interface,
                      reference_manifest=dict(path=str(manifest), sha256=sha256(manifest)))
    catalog.training_manifest = training
    catalog.training_manifest_sha256 = sha256(training)
    return catalog, rows


def test_migrated_catalog_uses_verified_reference_inventory_and_authentic_human_only(tmp_path):
    catalog, rows = reference_catalog(tmp_path, 'human')
    assert catalog.search('all', '')['motions'] == ['dance', 'flip']
    assert catalog.select('q1', 'dance') == Path(rows[0]['output'])
    Path(rows[0]['human_sidecar']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA'):
        catalog.select('q1', 'dance')


def test_reference_manifest_tampering_is_rejected_before_selection(tmp_path):
    catalog, _ = reference_catalog(tmp_path)
    catalog.reference_manifest.write_text('{}')
    with pytest.raises(ValueError, match='Reference manifest SHA'):
        catalog.select('q1', 'dance')


def test_reference_inventory_rejects_missing_neutral_contract_and_escaped_outputs(tmp_path):
    root, manifest, training, rows = reference_data(tmp_path)
    for corrupt in ('missing_contract', 'escaped_output'):
        modified = json.loads(manifest.read_text())
        if corrupt == 'missing_contract':
            modified['clips'][0].pop('neutral_contract')
        else:
            modified['clips'][0]['output'] = str(training)
        manifest.write_text(json.dumps(modified))
        with pytest.raises(ValueError, match='neutral contract|fixed reference root'):
            Catalog(root, {}, reference_manifest=dict(path=str(manifest), sha256=sha256(manifest)))
        modified['clips'] = rows
        manifest.write_text(json.dumps(modified))


def test_neutral_reference_comparison_keeps_entry_action_and_exit(tmp_path):
    root, manifest, _, rows = reference_data(tmp_path)
    path = Path(rows[0]['output'])
    session = PolicySession.__new__(PolicySession)
    session.human_root = None
    session.tmp = tempfile.TemporaryDirectory(dir=tmp_path)
    try:
        cropped = session._prepare_source_body('q1', path)
        with np.load(cropped) as data:
            assert len(data['joint_pos']) == rows[0]['frames']
        assert session.source_window['start_frame'] == 0
        assert session.source_window['stop_frame'] == rows[0]['frames']
        assert session.source_metadata['neutral_contract']['source_start_frame'] > 50
    finally:
        session.tmp.cleanup()


@pytest.mark.parametrize('hold_location', ['original', 'exit'])
def test_recovery_uses_shifted_reference_window_and_rejects_neutral_start(tmp_path, hold_location):
    catalog, rows = reference_catalog(tmp_path)
    windows = catalog.root / 'windows.json'
    poses = {'lying': {}}
    for row in rows:
        if row['name'] != 'dance': continue
        c = row['neutral_contract']
        poses['lying'][row['robot']] = dict(clip='dance', start_frame=c['source_start_frame'] + 1,
            hold_frame=c['source_stop_frame'] - 1 if hold_location == 'original' else c['exit_hold_start_frame'])
    document = dict(training_manifest_sha256=catalog.training_manifest_sha256,
                    reference_manifest_sha256=catalog.reference_manifest_sha256, preview_poses=poses)
    windows.write_text(json.dumps(document))
    catalog.set_recovery_windows(dict(path=str(windows), sha256=sha256(windows)))
    selected = catalog.select_recovery('all', 'lying')
    assert selected['q1'] == dict(path=Path(rows[0]['output']), start_frame=poses['lying']['q1']['start_frame'],
        hold_frame=poses['lying']['q1']['hold_frame'])
    poses['lying']['q1']['start_frame'] = 0
    windows.write_text(json.dumps(document))
    catalog.set_recovery_windows(dict(path=str(windows), sha256=sha256(windows)))
    with pytest.raises(ValueError, match='source action window'):
        catalog.select_recovery('q1', 'lying')


def test_flip_proof_binds_new_reference_and_retains_source_identity(tmp_path):
    catalog, rows = reference_catalog(tmp_path, 'human')
    source = tmp_path / 'raw-source.bvh'
    source.write_bytes(b'authentic raw source')
    proof = catalog.root / 'flip-preview.json'
    clips = [dict(robot=r['robot'], clip=r['name'], source_id=r['source_id'], split=r['split'], frames=r['frames'],
                  source=dict(path=str(source), sha256=sha256(source)),
                  human_source=dict(path=str(source), sha256=sha256(source))) for r in rows if r['name'] == 'flip']
    document = dict(schema_version=1, fps=50, training_manifest_sha256=catalog.training_manifest_sha256,
                    reference_manifest_sha256=catalog.reference_manifest_sha256, clips=clips)
    proof.write_text(json.dumps(document))
    catalog.set_flip_sources(dict(path=str(proof), sha256=sha256(proof)))
    assert set(catalog.select('all', 'flip')) == set(ROBOTS)
    document['reference_manifest_sha256'] = '0' * 64
    proof.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='reference manifest'):
        catalog.set_flip_sources(dict(path=str(proof), sha256=sha256(proof)))


def test_release_keeps_checkpoint_training_identity_separate_from_new_reference(tmp_path):
    import torch
    root, manifest, training, rows = reference_data(tmp_path)
    document = json.loads(manifest.read_text())
    document.pop('human_data')  # Mixed references have per-clip human availability.
    manifest.write_text(json.dumps(document))
    checks = {}
    for robot in ROBOTS:
        path = tmp_path / robot / 'ckpt_7000.pt'
        path.parent.mkdir()
        torch.save(dict(iteration=7000, cfg=dict(robot=robot, num_envs=2048, optimization_schema=1,
            kp_interface=True, kp_fixed_horizon=1, policy_goal='motion', optimizer='muon'),
            optimized_training=dict(schema_version=1, iteration=7000, optimizer='muon', num_envs=2048,
                joint_motion_human=True, human_horizon=1, human_points='five',
                prepared_manifest_sha256=sha256(training))), path)
        checks[robot] = dict(path=str(path), sha256=sha256(path))
    release = dict(id='joint-human-7000', label='Neutral references', interface='human', data_root=str(root),
        checkpoints=checks, parent_iteration=7000, human_updates=7000,
        training_manifest=dict(path=str(training), sha256=sha256(training)),
        reference_manifest=dict(path=str(manifest), sha256=sha256(manifest)))
    releases = tmp_path / 'releases.json'
    releases.write_text(json.dumps([release]))
    catalog = load_release_catalogs(releases)['joint-human-7000']['catalog']
    assert catalog.training_manifest == training
    assert catalog.reference_manifest == manifest
    assert catalog.select('q1', 'dance') == Path(rows[0]['output'])
    release['training_manifest'] = release['reference_manifest']
    releases.write_text(json.dumps([release]))
    with pytest.raises(ValueError, match='checkpoint identity'):
        load_release_catalogs(releases)


@pytest.mark.parametrize('interface', ['motion', 'human'])
def test_repinning_a_changed_source_slice_does_not_bypass_original_action_proof(tmp_path, interface):
    catalog, rows = reference_catalog(tmp_path, interface)
    row = rows[0]
    path = Path(row['human_sidecar'] if interface == 'human' else row['output'])
    with np.load(path) as source:
        data = {k: source[k] for k in source.files}
    start = row['neutral_contract']['source_start_frame']
    if interface == 'human':
        data['human_pos_w'][start, 0, 0] += .1
    else:
        data['joint_pos'][start, 0] += .1
    np.savez(path, **data)
    if interface == 'human':
        catalog.records['q1', 'dance']['human_sha256'] = sha256(path)
    else:
        catalog.motion_hashes['q1', 'dance'] = sha256(path)
        catalog.records['q1', 'dance']['output_sha256'] = sha256(path)
    with pytest.raises(ValueError, match='source slice'):
        catalog.select('q1', 'dance')


def test_reviewed_pose_is_verified_even_if_changed_bookend_is_rehashed(tmp_path):
    catalog, rows = reference_catalog(tmp_path)
    row = rows[0]
    path = Path(row['output'])
    with np.load(path) as source:
        data = {k: source[k] for k in source.files}
    data['joint_pos'][:row['neutral_contract']['entry_hold_end_frame'], 0] += .1
    np.savez(path, **data)
    catalog.motion_hashes['q1', 'dance'] = sha256(path)
    catalog.records['q1', 'dance']['output_sha256'] = sha256(path)
    with pytest.raises(ValueError, match='approved target'):
        catalog.select('q1', 'dance')


def test_source_slice_digest_ignores_bookends_and_binds_name_dtype_and_shape():
    from preview.reference import source_slice_digest
    data = dict(joint_pos=np.array([[0., 0.], [1., 2.], [0., 0.]], dtype=np.float32))
    original = source_slice_digest(data, 1, 2)
    data['joint_pos'][0] += 10.
    assert source_slice_digest(data, 1, 2) == original
    for changed in (dict(joint_vel=data['joint_pos']),
                    dict(joint_pos=data['joint_pos'].astype(np.float64)),
                    dict(joint_pos=data['joint_pos'].reshape(3, 2, 1))):
        assert source_slice_digest(changed, 1, 2) != original
    data['joint_pos'][1, 0] += .01
    assert source_slice_digest(data, 1, 2) != original


@pytest.mark.parametrize('iteration', [5000, 6000, 7000])
@pytest.mark.parametrize('interface', ['motion', 'human'])
def test_migrated_standup_requires_shifted_windows_for_every_joint_release(iteration, interface):
    from types import SimpleNamespace
    from preview.server import Preview
    release = f'joint-{interface}-{iteration}'
    catalog = SimpleNamespace(interface=interface, recovery_windows=None,
        reference_manifest=Path('/reviewed/reference.json'), select=lambda *args: None)
    preview = Preview(catalog, releases={release: dict(catalog=catalog, label='Reviewed neutral')})
    with pytest.raises(ValueError, match='reviewed neutral'):
        preview.control(dict(action='standup', robot='q1', release=release, pose='lying'))
    catalog.recovery_windows = True
    catalog.select_recovery = lambda *args: {}
    preview.control(dict(action='standup', robot='q1', release=release, pose='lying'))
    assert preview.requests.get_nowait()['motion'] == 'neutral_lying'


@pytest.mark.parametrize('human_folder', ['../outside-human', 'unchecked-human'])
def test_reference_rejects_human_loader_roots_that_bypass_verified_sidecars(tmp_path, human_folder):
    root = tmp_path / 'reference'
    (root / 'q1').mkdir(parents=True)
    native = root / 'q1/dance.npz'; native.write_bytes(b'native')
    side = root / 'human50/q1/dance.npz'; side.parent.mkdir(parents=True); side.write_bytes(b'human')
    (root / human_folder).mkdir(parents=True)
    review = root / 'standing-review.json'
    review.write_text(json.dumps(dict(prepared_frame_zero_based=11446, prepared_fps=50)))
    row = dict(robot='q1', name='dance', frames=14, output=str(native), output_sha256=sha256(native),
        human_available=True, human_sidecar=str(side), human_sha256=sha256(side),
        neutral_contract=dict(entry_hold_end_frame=2, source_start_frame=4, source_stop_frame=8,
            exit_hold_start_frame=10, frames=14))
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps(dict(clips=[row], human_root=human_folder,
        neutral_review=dict(path='standing-review.json', sha256=sha256(review)))))
    with pytest.raises(ValueError, match='human root|human sidecar'):
        Catalog(root, {}, reference_manifest=dict(path=str(manifest), sha256=sha256(manifest)))


def preview_migration_inputs(tmp_path):
    import hashlib
    root = tmp_path / 'new'; root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(dict(complete=True, clips=[dict(robot='q1', name='dance')])))
    poses = dict(lying=dict(q1=dict(clip='dance', start_frame=140, hold_frame=220)))
    (root / 'windows.json').write_text(json.dumps(dict(preview_poses=poses, training_manifest_sha256='obsolete')))
    (root / 'flip-preview.json').write_text(json.dumps(dict(clips=[dict(clip='flip', source_window=[110, 180])], training_manifest_sha256='obsolete')))
    rows = []
    for iteration in (5000, 6000, 7000):
        folder = tmp_path / 'training' / str(iteration); folder.mkdir(parents=True)
        manifest = folder / 'manifest.json'; manifest.write_text(json.dumps(dict(iteration=iteration)))
        for robot in ROBOTS:
            (folder / robot).mkdir(); (folder / robot / 'train.yaml').write_text(robot + str(iteration))
        for interface in ('motion', 'human'):
            rows.append(dict(id=f'joint-{interface}-{iteration}', interface=interface, label='Historical joint policy',
                data_root='/obsolete', checkpoints={r: dict(path=f'/fixed/{iteration}/{r}/ckpt_{iteration}.pt', sha256=r*32) for r in ROBOTS},
                parent_iteration=iteration, human_updates=iteration,
                training_manifest=dict(path=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
    rows.append(dict(rows[2], id='joint-motion-6000-previous'))
    catalog = tmp_path / 'old-catalog.json'; catalog.write_text(json.dumps(rows))
    return root, catalog, rows


def test_preview_catalog_migration_preserves_policy_identity_and_shifted_templates(tmp_path):
    import hashlib
    from scripts.migrate_neutral_preview import migrate
    root, old_catalog, originals = preview_migration_inputs(tmp_path)
    old_bytes = old_catalog.read_bytes()
    output = migrate(root, old_catalog)
    rows = json.loads(output.read_text())
    assert [row['id'] for row in rows] == [f'joint-{i}-{n}' for n in (7000, 6000, 5000) for i in ('motion', 'human')]
    for row in rows:
        original = next(r for r in originals if r['id'] == row['id'])
        assert row['checkpoints'] == original['checkpoints']
        assert row['training_manifest'] == original['training_manifest']
        assert row['data_root'] == str(root)
        assert row['reference_manifest']['sha256'] == hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
        for kind in ('recovery_windows', 'flip_sources'):
            proof = json.loads(Path(row[kind]['path']).read_text())
            assert proof['training_manifest_sha256'] == original['training_manifest']['sha256']
            assert proof['reference_manifest_sha256'] == row['reference_manifest']['sha256']
            for robot in ROBOTS:
                source = Path(original['training_manifest']['path']).parent / robot / 'train.yaml'
                assert proof['subset_sha256'][robot] == hashlib.sha256(source.read_bytes()).hexdigest()
        windows = json.loads(Path(row['recovery_windows']['path']).read_text())
        assert windows['preview_poses']['lying']['q1'] == dict(clip='dance', start_frame=140, hold_frame=220)
        flips = json.loads(Path(row['flip_sources']['path']).read_text())
        assert flips['clips'][0]['source_window'] == [110, 180]
    assert old_catalog.read_bytes() == old_bytes
    assert migrate(root, old_catalog).read_bytes() == output.read_bytes()


def test_preview_catalog_migration_refuses_changed_training_identity_before_writing(tmp_path):
    from scripts.migrate_neutral_preview import migrate
    root, old_catalog, originals = preview_migration_inputs(tmp_path)
    Path(originals[0]['training_manifest']['path']).write_text('{"changed": true}')
    with pytest.raises(ValueError, match='training manifest SHA'):
        migrate(root, old_catalog)
    assert not (root / 'releases.json').exists()
    assert not list(root.glob('preview-*.json'))


def test_preview_catalog_migration_refuses_overwriting_a_different_reviewed_catalog(tmp_path):
    from scripts.migrate_neutral_preview import migrate
    root, old_catalog, _ = preview_migration_inputs(tmp_path)
    catalog = root / 'releases.json'; catalog.write_text('previous reviewed release')
    with pytest.raises(FileExistsError):
        migrate(root, old_catalog)
    assert catalog.read_text() == 'previous reviewed release'


def test_source_body_crop_preserves_robot_and_human_clock_correspondence(tmp_path):
    pytest.importorskip('mjlab')
    from morphmimic.data.human_convert import sidecar_path
    root = tmp_path / 'source'; (root / 'q1').mkdir(parents=True)
    path = root / 'q1/dance.npz'
    clock = np.arange(210) / 50
    source_index = np.arange(210, dtype=float) / 2
    np.savez(path, fps=50, joint_pos=np.zeros((210, 1)), time_s=clock, source_frame_index=source_index)
    (root / 'manifest.json').write_text(json.dumps(dict(clips=[dict(robot='q1', name='dance', phases=dict(entry=3, dance=3.2))])))
    side = root / 'human50/q1/dance.npz'; side.parent.mkdir(parents=True)
    np.savez(side, fps=50, human_pos_w=np.zeros((210, 24, 3)), human_quat_w=np.tile([1., 0., 0., 0.], (210, 24, 1)),
        time_s=clock, source_frame_index=source_index, source_time_s=clock + 4.)
    session = PolicySession.__new__(PolicySession)
    session.human_root = str(root / 'human50')
    session.tmp = tempfile.TemporaryDirectory(dir=tmp_path)
    try:
        cropped = session._prepare_source_body('q1', path)
        for output in (cropped, sidecar_path(cropped, session.human_root)):
            with np.load(output) as data:
                np.testing.assert_array_equal(data['time_s'], np.arange(150, 160) / 50)
                np.testing.assert_array_equal(data['source_frame_index'], np.arange(150, 160) / 2)
    finally:
        session.tmp.cleanup()


def test_in_place_recovery_cannot_freeze_at_the_inherited_old_standing_frame(tmp_path):
    catalog,rows=reference_catalog(tmp_path,in_place=True)
    c=rows[0]['neutral_contract'];path=catalog.root/'windows.json'
    option=dict(clip='dance',start_frame=c['source_start_frame'],hold_frame=c['source_stop_frame']-1)
    doc=dict(training_manifest_sha256=catalog.training_manifest_sha256,reference_manifest_sha256=catalog.reference_manifest_sha256,preview_poses=dict(lying=dict(q1=option)))
    path.write_text(json.dumps(doc));catalog.set_recovery_windows(dict(path=str(path),sha256=sha256(path)))
    with pytest.raises(ValueError,match='terminal neutral'):catalog.select_recovery('q1','lying')
    option['hold_frame']=c['exit_hold_start_frame'];path.write_text(json.dumps(doc));catalog.set_recovery_windows(dict(path=str(path),sha256=sha256(path)))
    assert catalog.select_recovery('q1','lying')['q1']['hold_frame']==c['exit_hold_start_frame']
