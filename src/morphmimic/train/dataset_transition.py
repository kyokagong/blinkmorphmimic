"""Verify append-only training data changes and extend only inventory metadata."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import torch


def _checked(entry):
    path = Path(entry['path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('Dataset transition requires fixed files')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
            raise ValueError('Dataset transition SHA differs: ' + str(path))
    return json.loads(path.read_text())


def _paths(subset):
    return [str((Path(subset['root']) / p).resolve(strict=True)) for p in subset['clips']]


def verify_append_transition(path, expected_sha, manifest_path, robot):
    document = _checked(dict(path=str(path), sha256=expected_sha))
    if document.get('schema_version') != 1 or robot not in document.get('robots', {}):
        raise ValueError('Invalid dataset transition schema/robot')
    if Path(document['manifest']['path']).resolve() != Path(manifest_path).resolve():
        raise ValueError('Dataset transition names another training manifest')
    old, new = (_checked(document[k]) for k in ('parent_manifest', 'manifest'))
    prefix = old['clips']
    if not old.get('complete') or not new.get('complete') or new['clips'][:len(prefix)] != prefix or len(new['clips']) <= len(prefix):
        raise ValueError('Dataset transition must preserve every original manifest row')
    keys = [(r['robot'], r['name']) for r in new['clips']]
    if len(set(keys)) != len(keys):
        raise ValueError('Dataset transition has duplicate clips')
    for body, info in document['robots'].items():
        before, after = (_paths(_checked(info[k])) for k in ('parent_subset', 'subset'))
        names = [Path(p).stem for p in after]
        if (after[:len(before)] != before or len(after) <= len(before)
                or [Path(p).stem for p in before] != info['original_clips']
                or names != info['original_clips'] + info['appended_clips']):
            raise ValueError('Dataset transition must append to original subset order')
        rows = {str(Path(r['output']).resolve()):r for r in new['clips'] if r['robot'] == body}
        try:
            lengths = [rows[p]['frames'] for p in after]
        except KeyError as exc:
            raise ValueError('Dataset transition subset absent from manifest') from exc
        if (lengths != info['original_clip_frames'] + info['appended_clip_frames']
                or sum(lengths[:len(before)]) != info['original_frames']
                or sum(lengths[len(before):]) != info['appended_frames']
                or len(before) != info['original_clip_count']
                or len(after)-len(before) != info['appended_clip_count']):
            raise ValueError('Dataset transition frame/clip counts differ')
    old_floor, new_floor = (_checked(document[k]) for k in ('parent_contact_floor', 'contact_floor'))
    if (old_floor['prepared_manifest_sha256'] != document['parent_manifest']['sha256']
            or new_floor['prepared_manifest_sha256'] != document['manifest']['sha256']
            or new_floor['clips'][:len(old_floor['clips'])] != old_floor['clips']):
        raise ValueError('Dataset transition changed original contact calibration')
    _checked(document['approved_target']['review_manifest'])
    if document['approved_target']['frame'] != 11446 or document['approved_target']['fps'] != 50:
        raise ValueError('Dataset transition differs from approved standing target')
    return document


def extend_inventory_state(checkpoint, library, command, info):
    """Keep learned tensors untouched; copy old counters and sampler into their prefix."""
    n = len(info['original_clips'])
    if (library.clip_names != info['original_clips'] + info['appended_clips']
            or library.clip_len.cpu().tolist() != info['original_clip_frames'] + info['appended_clip_frames']
            or checkpoint.get('coverage', {}).get('clip_names') != info['original_clips']):
        raise ValueError('Checkpoint inventory is not the verified original prefix')
    result = dict(checkpoint)
    coverage = copy.deepcopy(checkpoint['coverage']); coverage['clip_names'] = list(library.clip_names)
    for stage, current in library.usage.items():
        saved = coverage['counts'][stage]
        if saved.shape != (n,) or saved.dtype != torch.long or bool((saved < 0).any()):
            raise ValueError('Invalid original coverage counters')
        expanded = torch.zeros_like(current, device='cpu'); expanded[:n] = saved.cpu()
        coverage['counts'][stage] = expanded
    source = coverage['source']
    if (type(source.get('since_iteration')) is not int or source['since_iteration'] < 0
            or set(source['counts']) != set(library.source_usage)
            or set(coverage['counts']) != set(library.usage)):
        raise ValueError('Invalid original source coverage schema/epoch')
    for key, current in (('start', library.source_start), ('stop', library.source_stop), ('has_phases', library.source_has_phases)):
        values = current.cpu().tolist()
        if source[key] != values[:n]:
            raise ValueError('Original source coverage windows changed')
        source[key] = values
    for stage, current in library.source_usage.items():
        saved = source['counts'][stage]
        if len(saved) != n or any(type(v) is not int or v < 0 for v in saved):
            raise ValueError('Invalid original source coverage counters')
        source['counts'][stage] = saved + [0] * (len(current)-n)
    result['coverage'] = coverage
    if checkpoint.get('human_coverage') is not None:
        from types import SimpleNamespace
        previous = human_coverage_state(checkpoint['human_coverage'],
            SimpleNamespace(clip_names=info['original_clips'], device=torch.device('cpu')),
            checkpoint['iteration'])
        expanded = dict(previous, clip_names=list(library.clip_names))
        for key in ('counts', 'source_counts'):
            values = torch.zeros(len(library.clip_names), dtype=torch.long)
            values[:n] = previous[key]
            expanded[key] = values
        result['human_coverage'] = expanded
    sampler = dict(checkpoint['sampler']); old_bins = int((library.bin_start < info['original_frames']).sum())
    for key, current in (('G_b', command.G_b), ('past_softmax', command.past_softmax)):
        saved = sampler[key]
        if saved.shape != (old_bins,) or not bool(torch.isfinite(saved).all()) or bool((saved < 0).any()):
            raise ValueError('Original sampler bins differ from append-only prefix')
        expanded = torch.zeros_like(current, device='cpu')
        if key == 'past_softmax':
            if not torch.isclose(saved.sum(), saved.new_tensor(1.), atol=1e-4):
                raise ValueError('Original sampler distribution is not normalized')
            expanded.fill_(1 / len(expanded)); expanded[:old_bins] = saved.cpu() * old_bins / len(expanded)
        else:
            expanded[:old_bins] = saved.cpu()
        sampler[key] = expanded
    if 'probe_probabilities' in sampler:
        saved = sampler['probe_probabilities']
        if saved.shape != (old_bins,):
            raise ValueError('Original probe bins differ')
        expanded = torch.zeros_like(command.G_b, device='cpu'); expanded[:old_bins] = saved.cpu()
        sampler['probe_probabilities'] = expanded
    result['sampler'] = sampler
    return result


def human_coverage_state(state, library, iteration):
    """Legacy checkpoints have no human source counters; start an explicit new epoch."""
    if state is None:
        return dict(clip_names=list(library.clip_names), since_iteration=int(iteration),
                    counts=torch.zeros(len(library.clip_names), dtype=torch.long, device=library.device),
                    source_counts=torch.zeros(len(library.clip_names), dtype=torch.long, device=library.device))
    if (state.get('clip_names') != library.clip_names or type(state.get('since_iteration')) is not int
            or not 0 <= state['since_iteration'] <= iteration):
        raise ValueError('Human coverage inventory or epoch differs')
    result = dict(state)
    for key in ('counts', 'source_counts'):
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != (len(library.clip_names),) or value.dtype != torch.long or bool((value < 0).any()):
            raise ValueError('Invalid human coverage counters')
        result[key] = value.to(library.device).clone()
    if bool((result['source_counts'] > result['counts']).any()):
        raise ValueError('Human source counts exceed total counts')
    return result
