"""The optional compact dynamics path must support policy gradients on CUDA."""
from dataclasses import replace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('robot', ['q1', 'bumi', 'g1'])
def test_compact_native_fk_and_contact_reward_capture_action_gradients(robot):
    from morphmimic.model.kinematics import RobotKinematics, calibrate_foot_heights
    from morphmimic.model.rewards_torch import RewardCfg
    from morphmimic.model.state import StateNormalizer
    from morphmimic.model.world_model import WorldModel, WorldModelCfg
    from morphmimic.robots import get_robot
    from morphmimic.train.cuda_graphs import CapturedBackward

    spec = get_robot(robot)
    native = spec.entity_cfg_fn().spec_fn().compile()
    fk = RobotKinematics.from_mujoco(native, spec).cuda()
    mask = torch.tensor(spec.translator.joint_mask, device='cuda')
    q = torch.zeros(2, 29, device='cuda')
    root = torch.zeros(2, 3, device='cuda'); root[:, 2] = .8
    quat = torch.zeros(2, 4, device='cuda'); quat[:, 0] = 1
    state = fk(q, q, root, quat, torch.zeros_like(root), torch.zeros_like(root))
    reward = replace(RewardCfg(), w_contact_agreement=.1, w_foot_slip=-.001,
        foot_height=calibrate_foot_heights(native, spec))
    model = WorldModel(WorldModelCfg(hidden=(16, 16), state_mode='compact_fk',
        joint_dynamics='semi_implicit'), StateNormalizer(),
        torch.tensor([[-3., 3.]]).expand(29, 2), torch.zeros(29),
        joint_mask=mask.cpu(), kinematics=fk, reward_cfg=reward).cuda().requires_grad_(False)
    with torch.no_grad():
        model.dynamics_mlp[-1].weight.normal_(std=.01)
    action = torch.zeros(2, 29, device='cuda', requires_grad=True)
    context = torch.zeros(2, 33, device='cuda')
    floor = torch.zeros(2, device='cuda')
    valid = torch.tensor([True, False], device='cuda')

    def objective(a):
        predicted = model.step(state, a, context)
        score = model.reward(predicted, state, a, torch.zeros_like(a),
            reference_floor=floor, reference_floor_valid=valid)
        return predicted.square().mean() - score.mean(), predicted

    graph = CapturedBackward(lambda: objective(action), [action])
    for amplitude in (.1, -.2):
        with torch.no_grad():
            action.fill_(amplitude)
        eager = action.detach().clone().requires_grad_(True)
        expected, expected_state = objective(eager)
        expected_grad, = torch.autograd.grad(expected, eager)
        loss, predicted = graph()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(predicted, expected_state)
        torch.testing.assert_close(action.grad, expected_grad, atol=2e-6, rtol=2e-5)
        assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
        assert torch.count_nonzero(action.grad[:, ~mask]) == 0
