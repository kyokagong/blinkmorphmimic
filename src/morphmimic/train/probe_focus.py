"""Optional robot-local reset focus from budgeted probes of training clips only."""
from __future__ import annotations

import math
import random
import re

import torch
from torch import Tensor

SCHEMA_VERSION = 1
LOOKBACK_SECONDS = 10
ROBOTS = ('q1', 'bumi', 'g1')


def focus_probabilities(document, library, robot: str, expected_subset_hash: str | None = None) -> Tensor:
    """Count complete bins within ten seconds before each local failure frame.

    Counts accumulate for duplicate failures. Partial boundary bins are excluded so
    focused starts remain before failure and inside the clip. No eligible bin yields
    zeros, which the command treats as the unchanged base sampler. Pass the current
    training YAML's SHA256 to reject focus from a different training subset.
    """
    if not isinstance(document, dict):
        raise ValueError('probe focus document must be an object')
    if type(document.get('schema_version')) is not int or document['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported probe focus schema')
    if robot not in ROBOTS or document.get('robot') != robot:
        raise ValueError('probe focus robot does not match this rank')
    interface=document.get('interface')
    if interface not in ('motion','combined') or document.get('fps') != library.fps:
        raise ValueError('probe focus interface/fps does not match the motion library')
    weights=document.get('interface_weights')
    if interface=='combined':
        if (not isinstance(weights,dict) or set(weights)!={'motion','human-five','human-full'}
            or any(type(w) not in (int,float) or not math.isfinite(w) or w<0 for w in weights.values())
            or not math.isfinite(sum(weights.values())) or sum(weights.values())<=0):
            raise ValueError('invalid combined interface weights')
    for key in ('checkpoint_sha256', 'training_subset_sha256'):
        if not isinstance(document.get(key), str) or re.fullmatch(r'[a-f0-9]{64}', document[key]) is None:
            raise ValueError(f'probe focus has invalid {key}')
    if expected_subset_hash is not None and document['training_subset_sha256'] != expected_subset_hash:
        raise ValueError('probe focus training subset fingerprint mismatch')
    failures = document.get('failures')
    if not isinstance(failures, list):
        raise ValueError('probe focus failures must be a list')
    names = {name: i for i, name in enumerate(library.clip_names)}
    if len(names) != len(library.clip_names):
        raise ValueError('probe focus requires unique clip names')
    counts = torch.zeros(library.num_bins, dtype=torch.float32, device=library.bin_start.device)
    for record in failures:
        if not isinstance(record, dict) or not isinstance(record.get('clip'), str):
            raise ValueError('probe failure must identify a clip by name')
        if record.get('robot') != robot or not isinstance(record.get('source_id'), str) or not record['source_id']:
            raise ValueError('probe failure has invalid robot/source identity')
        clip = names.get(record.get('clip'))
        if clip is None:
            raise ValueError(f'probe failure names unknown clip: {record.get("clip")!r}')
        frame = record.get('fail_frame')
        if type(frame) is not int or not 0 <= frame < int(library.clip_len[clip]):
            raise ValueError('probe failure frame is outside its clip')
        upper = library.clip_start[clip] + frame
        lower = torch.maximum(library.clip_start[clip], upper - LOOKBACK_SECONDS * library.fps)
        eligible = ((library.bin_clip == clip) & (library.bin_start >= lower)
                    & (library.bin_start + library.bin_len <= upper))
        weight=1.0
        if interface=='combined':
            name=record.get('interface');weight=record.get('weight')
            if (name not in weights or type(weight) not in (int,float) or not math.isfinite(weight)
                or weight<0 or weight!=weights[name]):
                raise ValueError('invalid combined failure interface or weight')
            weight /= sum(weights.values())
        counts.add_(eligible,alpha=weight)
    return counts / counts.sum().clamp_min(1)


def normalize_probe_focus(probabilities: Tensor, num_bins: int, rho: float) -> Tensor | None:
    """Validate once when installing focus, leaving the reset hot path sync-free."""
    if not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError('probe reset share must be finite and in [0, 1), preserving base exploration')
    if not isinstance(probabilities, Tensor) or probabilities.ndim != 1 or probabilities.numel() != num_bins:
        raise ValueError('probe probabilities must have one entry per local reset bin')
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('probe probabilities must be finite and nonnegative')
    mass = probabilities.sum()
    if not torch.isfinite(mass):
        raise ValueError('probe probability sum is nonfinite')
    return None if mass == 0 else (probabilities / mass).detach().clone()


def mix_probe_focus(base: Tensor, focus: Tensor | None, rho: float) -> Tensor:
    return base if focus is None or rho == 0 else (1 - rho) * base + rho * focus


def select_probe_clips(clips, max_clips: int, max_frames: int, seed: int):
    """Seeded whole-clip selection under a strict reference-transition budget."""
    if type(max_clips) is not int or not 1 <= max_clips <= 256 or type(max_frames) is not int or max_frames < 1:
        raise ValueError('probe needs 1..256 clips and a positive frame budget')
    order = list(clips)
    random.Random(seed).shuffle(order)
    selected = []
    remaining = max_frames
    for clip in order:
        frames = clip.get('frames')
        if type(frames) is not int or frames < 2:
            raise ValueError('probe clip must contain at least two frames')
        cost = frames - 1
        if cost <= remaining:
            selected.append(clip)
            remaining -= cost
            if len(selected) == max_clips:
                break
    if not selected:
        raise ValueError('no complete training clip fits the probe frame budget')
    return selected
