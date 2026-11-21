"""Small stateful training optimizations with explicit resume and I/O boundaries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import shutil

import torch


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def _write_checkpoint(path, snapshot, keep=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(snapshot, temporary)
    os.replace(temporary, path)
    latest = path.with_name('latest.pt')
    temporary = latest.with_name(latest.name + '.tmp')
    if keep:
        temporary.unlink(missing_ok=True)
        os.link(path, temporary)
    else:
        shutil.copyfile(path, temporary)
    os.replace(temporary, latest)
    if keep:
        checkpoints = sorted(
            (p for p in path.parent.glob('ckpt_*.pt') if p.stem.removeprefix('ckpt_').isdigit()),
            key=lambda p: int(p.stem.removeprefix('ckpt_')))
        for expired in checkpoints[:-keep]:
            expired.unlink()
        return
    excluded = {'optim', 'rng', 'rng_ranks', 'sampler', 'coverage'}
    light = {key: value for key, value in snapshot.items() if key not in excluded}
    light['inference_only'] = True
    target = path.with_name('policy_' + path.name.removeprefix('ckpt_'))
    temporary = target.with_name(target.name + '.tmp')
    torch.save(light, temporary)
    os.replace(temporary, target)


class AsyncCheckpointWriter:
    """One bounded writer; GPU-to-CPU copying finishes before the worker starts."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint')
        self.pending = None

    def flush(self):
        if self.pending is not None:
            self.pending.result()
            self.pending = None

    def submit(self, path, state, *, keep=0):
        self.flush()
        snapshot = cpu_snapshot(state)
        self.pending = self.executor.submit(_write_checkpoint, Path(path), snapshot, keep)

    def close(self):
        try:
            self.flush()
        finally:
            self.executor.shutdown(wait=True)


@dataclass
class ActorLRSchedule:
    multiplier: float = 1.
    threshold: float = .75
    decay_steps: int = 300
    triggered_at: int | None = None

    def __post_init__(self):
        if not math.isfinite(self.multiplier) or self.multiplier < 1 or self.decay_steps < 1:
            raise ValueError('Actor LR multiplier must be finite >=1 and decay steps positive')
        if not math.isfinite(self.threshold):
            raise ValueError('Actor LR threshold must be finite')

    def observe(self, iteration, training_return):
        if self.triggered_at is None and math.isfinite(training_return) and training_return >= self.threshold:
            self.triggered_at = iteration

    def factor(self, iteration):
        if self.triggered_at is None:
            return self.multiplier
        fraction = min(1., max(0., (iteration - self.triggered_at) / self.decay_steps))
        return self.multiplier + (1. - self.multiplier) * fraction

    def state_dict(self):
        return asdict(self)


@dataclass
class WarmupGate:
    minimum: int = 100
    threshold: float = .2
    ready_at: int | None = None
    maximum: int = 1000

    def __post_init__(self):
        if self.minimum < 1 or self.maximum < self.minimum or not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError('Warmup minimum and finite error threshold must be positive')

    def observe(self, iteration, robot_errors):
        if self.ready_at is not None:
            return True
        if iteration >= self.minimum and robot_errors and all(math.isfinite(x) and x <= self.threshold for x in robot_errors):
            self.ready_at = iteration
        if self.ready_at is None and iteration >= self.maximum:
            raise RuntimeError(f'World model warmup failed by iteration {iteration}: errors={robot_errors}, threshold={self.threshold}')
        return self.ready_at is not None
