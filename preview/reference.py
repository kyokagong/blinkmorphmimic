"""Checks for reviewed neutral references, separate from checkpoint training identity."""
import hashlib
import json

FRAME_KEYS = frozenset(('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
    'body_lin_vel_w', 'body_ang_vel_w', 'root_pos_w', 'root_quat_w',
    'root_lin_vel_w', 'root_ang_vel_w', 'human_pos_w', 'human_quat_w',
    'source_time_s', 'source_frame_index', 'time_s'))


def source_slice_digest(arrays, start, stop):
    """Bind frame keys, native dtypes/shapes and exact saved source-slice bytes."""
    import numpy as np
    digest = hashlib.sha256()
    for key in sorted(FRAME_KEYS.intersection(arrays)):
        value = np.ascontiguousarray(arrays[key][start:stop])
        header = json.dumps([key, value.dtype.str, list(value.shape)], separators=(',', ':'))
        digest.update(header.encode('utf-8'))
        digest.update(b'\0')
        digest.update(value.tobytes(order='C'))
        digest.update(b'\0')
    return digest.hexdigest()


def reference_path(root, value):
    path = root / value
    if not path.is_file() or path != path.resolve(strict=True) or not path.is_relative_to(root):
        raise ValueError('Motion outside its fixed reference root')
    return path


def review_document(root, proof):
    from preview.server import sha256
    if not isinstance(proof, dict) or set(proof) != {'path', 'sha256'}:
        raise ValueError('Migrated references require a pinned neutral review')
    path = reference_path(root, proof['path'])
    if sha256(path) != proof['sha256']:
        raise ValueError('Neutral review SHA differs')
    review = json.loads(path.read_text())
    if review.get('prepared_frame_zero_based') != 11446 or review.get('prepared_fps') != 50:
        raise ValueError('Neutral review differs from the approved frame')
    return review


def neutral_contract(row):
    contract = row.get('neutral_contract', {})
    keys = ('entry_hold_end_frame', 'source_start_frame', 'source_stop_frame', 'exit_hold_start_frame', 'frames')
    if (any(type(contract.get(k)) is not int for k in keys)
            or not 0 < contract['entry_hold_end_frame'] <= contract['source_start_frame']
            < contract['source_stop_frame'] <= contract['exit_hold_start_frame'] < contract['frames']
            or contract['frames'] != row.get('frames')):
        raise ValueError('Invalid neutral contract frame intervals')
    return contract


def verify_neutral_reference(row, path, root, *, review, human=False):
    import numpy as np
    from morphmimic.data.neutral_bookends import validate_boundary
    from preview.server import sha256
    contract = neutral_contract(row)
    if path != reference_path(root, row['output']) or sha256(path) != row['output_sha256']:
        raise ValueError('Neutral reference SHA differs')
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    start, stop = contract['source_start_frame'], contract['source_stop_frame']
    expected = row.get('source_slice_sha256', contract.get('source_slice_sha256'))
    if expected != source_slice_digest(data, start, stop):
        raise ValueError('Neutral reference source slice SHA differs')
    document = review_document(root, review)
    identity = document['robot_sources'][row['robot']]
    target = reference_path(root, identity['npz'])
    if sha256(target) != identity['sha256']:
        raise ValueError('Approved neutral target SHA differs')
    with np.load(target, allow_pickle=False) as source:
        validate_boundary(data, {key: source[key] for key in source.files}, row['robot'], contract, target_frame=11446)
    if human:
        if not row.get('human_available'):
            raise ValueError('Neutral reference has no authentic human sidecar')
        side = reference_path(root, row['human_sidecar'])
        if sha256(side) != row['human_sha256']:
            raise ValueError('Neutral reference human sidecar SHA differs')
        with np.load(side, allow_pickle=False) as source:
            if row.get('human_source_slice_sha256') != source_slice_digest(source, start, stop):
                raise ValueError('Neutral reference human source slice SHA differs')
