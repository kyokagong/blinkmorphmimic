import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_reference_skip_internal_and_batched_morph_are_capture_safe():
    from morphmimic.model.policy import Policy,PolicyCfg
    p=Policy(PolicyCfg(enc_hidden=(8,8),dec_hidden=(8,8),human=True,
        morph_dim=154,film=True,ref_reconstruction_skip=True)).cuda()
    morph=torch.zeros(1,154,device='cuda');morph[:,-29:]=1
    p.set_morphology(morph[0]);p.set_reference_controls(morph,torch.zeros(1,29,device='cuda'),torch.ones(1,29,device='cuda'))
    p.set_reference_stats(torch.zeros(670,device='cuda'),torch.ones(670,device='cuda'))
    history=torch.randn(2,930,device='cuda');goal=torch.randn(2,670,device='cuda')
    for condition in (None,morph.expand(2,-1)):
        expected=p(history,goal,condition)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):actual=p(history,goal,condition)
        graph.replay()
        torch.testing.assert_close(actual,expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_human_heading_constants_are_capture_safe():
    from morphmimic.model.human import heading_features
    robot = torch.tensor([[1., 0., 0., 0.]], device='cuda')
    human = robot[:, None, None].expand(1, 10, 24, 4).clone()
    points = torch.ones(1, 24, dtype=torch.bool, device='cuda')
    expected = heading_features(robot, human, points)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = heading_features(robot, human, points)
    graph.replay()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_five_point_mask_sampling_is_capture_safe_and_causal():
    from morphmimic.model.human import sample_human_masks
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        points, frames, heading = sample_human_masks(4, 'five', torch.device('cuda'), fixed_horizon=1)
    graph.replay()
    assert points[0].nonzero().flatten().tolist() == [0, 7, 8, 20, 21]
    assert frames[:, 0].all() and not frames[:, 1:].any()
    assert heading.all()

