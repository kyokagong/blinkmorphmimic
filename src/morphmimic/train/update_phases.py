"""Optimizer boundaries shared by ordinary and serial equal-robot training."""
from dataclasses import dataclass
from typing import Callable
import torch
from torch import nn


@dataclass
class UpdatePhase:
    name: str
    parameters: tuple
    optimizer: object
    finish: Callable


def run_phases(phases):
    """Ordinary Trainer: finish each backward immediately, preserving old order."""
    result = None
    while True:
        try:
            phase = phases.send(result)
        except StopIteration as done:
            return done.value
        result = phase.finish()


def run_equal_phases(generators, accumulators=None, on_step=None):
    """Same-weight backward from each robot, then ONE optimizer step per phase.

    Accumulators own storage outside graph pools. Backward may zero/rebind p.grad;
    copy its contribution before advancing the next robot. Never average updates.
    """
    if not generators:
        raise ValueError('At least one robot phase stream is required')
    accumulators = {} if accumulators is None else accumulators
    result = None
    while True:
        phases, finished = [], []
        expected = None
        used = None
        for generator in generators:
            try:
                phase = generator.send(result)
            except StopIteration as done:
                finished.append(done.value)
                continue
            if finished:
                raise ValueError('Robot phase streams have different lengths')
            signature = (phase.name, id(phase.optimizer), tuple(id(p) for p in phase.parameters))
            present = tuple(p.grad is not None for p in phase.parameters)
            if expected is None:
                expected, used = signature, present
                if signature not in accumulators:
                    accumulators[signature] = [torch.zeros_like(p) for p in phase.parameters]
                sums = accumulators[signature]
                for value in sums:
                    value.zero_()
            elif signature != expected or present != used:
                raise ValueError('Robot phase/parameter/optimizer/gradient ownership differs')
            with torch.no_grad():
                for value, p in zip(sums, phase.parameters, strict=True):
                    if p.grad is not None:
                        value.add_(p.grad)
            phases.append(phase)
        if finished:
            if phases:
                raise ValueError('Robot phase streams have different lengths')
            return finished
        owner = phases[0]
        with torch.no_grad():
            for value, p, present in zip(sums, owner.parameters, used, strict=True):
                if present:
                    # Stable optimizer-facing storage, independent of local graph outputs.
                    p.grad.copy_(value / len(generators))
        result = owner.finish()
        if on_step is not None:
            on_step(owner.name)


def share_parameters(module: nn.Module, owner: nn.Module):
    """Alias Parameters by exact name before building optimizers; never alias buffers."""
    local, shared = dict(module.named_parameters()), dict(owner.named_parameters())
    if local.keys() != shared.keys():
        raise ValueError('Shared parameter names differ')
    for name, p in local.items():
        q = shared[name]
        if (p.shape, p.dtype, p.device, p.requires_grad) != (q.shape, q.dtype, q.device, q.requires_grad):
            raise ValueError(f'Shared parameter signature differs: {name}')
    for name, p in shared.items():
        parent, _, leaf = name.rpartition('.')
        setattr(module.get_submodule(parent) if parent else module, leaf, p)
