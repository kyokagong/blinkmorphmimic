"""Hidden-matrix Muon and parameter-only policy EMA; physics buffers are copied."""
from __future__ import annotations
import copy
import torch
from torch import nn


class PolicyEMA:
    def __init__(self, policy: nn.Module, decay: float):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.decay = decay
        self.module = copy.deepcopy(policy).requires_grad_(False).eval()
        self.updates = 0

    @torch.no_grad()
    def update_parameters(self, policy: nn.Module):
        torch._foreach_lerp_(list(self.module.parameters()), list(policy.parameters()), 1-self.decay)
        self.updates += 1

    @torch.no_grad()
    def copy_buffers(self, policy: nn.Module):
        for target, source in zip(self.module.buffers(), policy.buffers(), strict=True):
            target.copy_(source)

    def update(self, policy: nn.Module):
        self.update_parameters(policy)
        self.copy_buffers(policy)


class MuonAdam:
    def __init__(self, params, mlps, lr, betas, fused, capturable=False):
        params = list(params)
        hidden = set()
        for mlp in mlps:
            layers = [layer for layer in mlp.modules() if isinstance(layer, nn.Linear)]
            hidden.update(id(layer.weight) for layer in layers[1:-1])
        matrix = [p for p in params if id(p) in hidden]
        other = [p for p in params if id(p) not in hidden]
        if not matrix or not other:
            raise ValueError('Muon requires internal matrices and Adam parameter groups')
        if not hasattr(torch.optim, 'Muon'):
            raise RuntimeError('Muon requires the pinned Docker PyTorch runtime')
        self.muon = torch.optim.Muon(matrix, lr=lr, weight_decay=0, adjust_lr_fn='match_rms_adamw')
        self.adam = torch.optim.Adam(other, lr=lr, betas=betas, fused=fused, capturable=capturable)

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adam.param_groups

    @property
    def state(self):
        return dict(self.muon.state) | dict(self.adam.state)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return dict(muon=self.muon.state_dict(), adam=self.adam.state_dict())

    def load_state_dict(self, state):
        if set(state) != {'muon', 'adam'}:
            raise ValueError('checkpoint optimizer differs: expected Muon+Adam')
        self.muon.load_state_dict(state['muon'])
        self.adam.load_state_dict(state['adam'])


def make_optimizer(params, mlps, *, kind='adam', lr=.0002, betas=(.9,.999), fused=False, capturable=False):
    if kind == 'adam':
        return torch.optim.Adam(params, lr=lr, betas=betas, fused=fused, capturable=capturable)
    if kind == 'muon':
        return MuonAdam(params, mlps, lr, betas, fused, capturable=capturable)
    raise ValueError(f'Unknown optimizer: {kind}')


def restore_optimizer_runtime(opt, *, fused: bool, capturable: bool):
    """Restore this run's Adam mode and device-resident graph learning rates after load."""
    if not isinstance(opt, (torch.optim.Adam, MuonAdam)):
        raise TypeError('Expected Adam or MuonAdam')
    if not isinstance(fused, bool) or not isinstance(capturable, bool):
        raise TypeError('Optimizer runtime flags must be booleans')
    adam = opt.adam if isinstance(opt, MuonAdam) else opt
    if isinstance(opt, MuonAdam):
        for group in opt.muon.param_groups:
            group.pop('fused', None)
            group.pop('capturable', None)
    for group in opt.param_groups:
        params = group['params']
        if not params:
            raise ValueError('Optimizer group has no parameters')
        if capturable:
            device = params[0].device
            if device.type != 'cuda' or any(p.device != device for p in params):
                raise ValueError('Capturable optimizer group needs parameters on one CUDA device')
            group['lr'] = torch.as_tensor(group['lr'], device=device)
    for group in adam.param_groups:
        group['fused'] = fused
        group['capturable'] = capturable
        for param in group['params']:
            state = adam.state.get(param, {})
            if 'step' in state:
                device = param.device if fused or capturable else torch.device('cpu')
                state['step'] = torch.as_tensor(state['step'], device=device)
