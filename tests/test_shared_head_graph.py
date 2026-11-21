"""Alternating motion/human optimizers may share a head and its gradient storage."""
import copy

import pytest
import torch
from torch import nn

from morphmimic.train.cuda_graphs import CapturedBackward, CapturedOptimizerStep
from morphmimic.train.imagination import frozen, guarded_clip_grad_norm_


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('fused', [False, True])
def test_alternating_shared_head_matches_eager_for_three_updates(fused):
    torch.manual_seed(17)
    eager = nn.ModuleDict({name: nn.Linear(4, 4) for name in ('motion', 'human', 'head')}).cuda()
    captured = copy.deepcopy(eager)
    models = (eager, captured)
    names = ('motion', 'human')
    parameters = [[list(model[name].parameters()) + list(model['head'].parameters())
                   for name in names] for model in models]
    optimizers = [[torch.optim.Adam(params, lr=torch.tensor(.01, device='cuda'),
                                   capturable=True, fused=fused)
                   for params in model_params] for model_params in parameters]

    def rollout(model, name, x, target):
        return (model['head'](torch.tanh(model[name](x))) - target).square().mean()

    backwards = [CapturedBackward(
        lambda x, target, name=name: rollout(captured, name, x, target), params)
        for name, params in zip(names, parameters[1])]
    steps = [CapturedOptimizerStep(opt, params,
             lambda params=params: guarded_clip_grad_norm_(params, .25))
             for opt, params in zip(optimizers[1], parameters[1])]

    for iteration, lr in enumerate((.01, .02, .005)):
        x, target, reconstruction = (torch.randn(8, 4, device='cuda') for _ in range(3))
        for phase, name in enumerate(names):
            for opts in optimizers:
                opts[phase].param_groups[0]['lr'].fill_(lr / (phase + 1))
                opts[phase].zero_grad(set_to_none=False)
            for model_index, model in enumerate(models):
                # The head participates in the human forward but is frozen until reconstruction.
                with frozen(*(model['head'],) if name == 'human' else ()):
                    if model_index:
                        backwards[phase](x, target)
                    else:
                        rollout(model, name, x, target).backward()
                if name == 'human':
                    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0
                               for p in model['head'].parameters())
                (.3 * rollout(model, name, x, reconstruction)).backward()

            for left, right in zip(parameters[0][phase], parameters[1][phase]):
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
            original_gradients = [p.grad for p in parameters[1][phase]]
            # Single-rank equivalent: all-reduce is a no-op, before clipping/step.
            eager_clip = guarded_clip_grad_norm_(parameters[0][phase], .25)
            optimizers[0][phase].step()
            graph_clip = steps[phase].step()
            for left, right in zip(eager_clip, graph_clip):
                torch.testing.assert_close(left, right)
            for left, right, original, staged in zip(
                    parameters[0][phase], parameters[1][phase], original_gradients,
                    steps[phase].grads):
                assert right.grad is original
                torch.testing.assert_close(left.grad, staged, atol=2e-6, rtol=2e-5)
            for left, right in zip(eager.parameters(), captured.parameters()):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
            for opt_index in range(2):
                left_opt, right_opt = optimizers[0][opt_index], optimizers[1][opt_index]
                for left, right in zip(parameters[0][opt_index], parameters[1][opt_index]):
                    left_state, right_state = left_opt.state.get(left, {}), right_opt.state.get(right, {})
                    assert left_state.keys() == right_state.keys()
                    for key in left_state:
                        torch.testing.assert_close(left_state[key], right_state[key], atol=2e-6, rtol=2e-5)
                    if right_state:
                        assert right_state['step'].item() == iteration + (opt_index <= phase)

    assert all(opt.state[captured['head'].weight]['step'].item() == 3
               for opt in optimizers[1])
    assert optimizers[1][0].state[captured['head'].weight]['exp_avg'].data_ptr() != \
        optimizers[1][1].state[captured['head'].weight]['exp_avg'].data_ptr()
