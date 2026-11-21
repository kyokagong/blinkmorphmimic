"""Opt-in actual simulator qualification. Set CONTROLLER_NATIVE_SUBSETS to robot/path JSON.

No default GPU work; parent schedules this test after CPU review. Source and all
original clips must be mounted read-only. These are small qualification fixtures,
not claims about the completed 5000-step experiment or large-batch capacity.
"""
import json
import os
import pytest
import torch

SUBSETS=os.environ.get('CONTROLLER_NATIVE_SUBSETS')
pytestmark=pytest.mark.skipif(not SUBSETS or not torch.cuda.is_available(),reason='scheduled native GPU fixture required')


@pytest.mark.parametrize('robot',['q1','bumi','g1'])
@pytest.mark.parametrize('captured',[False,True])
def test_actual_native_controller_before_after_targets_and_reset_clock(robot,captured):
    from morphmimic.env.env_cfg import morphmimic_env_cfg
    from morphmimic.env.morphmimic_env import MorphMimicEnv
    from morphmimic.env.controller_recording import NativeControllerRecorder,read_delay_state
    from morphmimic.env.masked_reset import MaskedResetGraph
    from morphmimic.train.collector import Collector
    from morphmimic.train.replay import ReplayBuffer,ReplayCfg
    from morphmimic.robots import get_robot
    from morphmimic.model.controller import ControllerState
    paths=json.loads(open(SUBSETS).read())
    cfg=morphmimic_env_cfg(robot,subset_yaml=paths[robot],num_envs=4,play=True,eval=False,
                          human_root=os.environ.get('CONTROLLER_HUMAN_ROOT'),actuator_delay=True)
    cfg.commands['motion'].allow_missing_human=True
    device='cuda:0';stream=torch.cuda.Stream() if captured else None
    env=MorphMimicEnv(cfg,device=device,graph_stream=stream)
    try:
        rec=NativeControllerRecorder(env,get_robot(robot));env.controller_recording=rec
        from morphmimic.env.controller_recording import controller_from_config
        expected=controller_from_config(dict(robot=robot,controller_mode='native_delay_v1',
            actuator_delay=True,actuator_delay_max_ms=20.,actuator_delay_fixed=True))
        assert rec.controller.identity()==expected.identity()
        replay=ReplayBuffer(ReplayCfg(4,16,device,controller_horizon=4))
        collector=Collector(env,replay,lambda h,g:torch.zeros(4,29,device=device),physics_graph=captured)
        collector.reset_all()
        reset=MaskedResetGraph(env) if captured else None
        for j,bits in enumerate(([False]*4,[True,False,False,False],[True]*4,[False,True,False,False])):
            ctx=env.obs_buf['context'].clone()
            action=torch.linspace(-.05,.05,29,device=device).expand(4,-1)*(j+1)
            if captured:result=collector._physics_graph.step(action)
            else:result=env.step(action)
            collector._check_graph_overflow('controller qualification step')
            before=ControllerState(*(x.clone() for x in rec.before));after=ControllerState(*(x.clone() for x in rec.after))
            targets=rec.targets.clone();lags=rec.lags.clone()
            mask=torch.tensor(bits,device=device)|(result[2]|result[3])
            rng=torch.cuda.get_rng_state();fresh=read_delay_state(rec.delay,rec.canonicalize)
            assert torch.equal(rng,torch.cuda.get_rng_state())
            for a,b in zip(after,fresh):torch.testing.assert_close(a,b,rtol=0,atol=0)
            if captured:reset(mask)
            elif bool(mask.any()):env.reset(env_ids=mask.nonzero().flatten())
            collector._check_graph_overflow('controller qualification reset')
            following=read_delay_state(rec.delay,rec.canonicalize)
            innovations=torch.cat((lags[:,:cfg.decimation],following.meta[:,:1]),-1)
            pure=rec.controller.transition(before,action,ctx[:,1:30],innovations,mask.any().expand(4))
            torch.testing.assert_close(pure.targets,targets,rtol=2e-6,atol=3e-7)
            for a,b in zip(pure.after,after):torch.testing.assert_close(a,b,rtol=2e-6,atol=3e-7)
            for a,b in zip(pure.next,following):torch.testing.assert_close(a[~mask],b[~mask],rtol=2e-6,atol=3e-7)
        assert env.physics_diagnostics.report()['overflow_bits']==0
        # Real collector publication and sampling also use the observed state.
        collector.reset_all();collector.step(5)
        batch=replay.sample_windows(4,2);start=replay.sample_starts(4,1)
        assert batch.controller is not None and start.controller is not None
        assert start.reset_clock.shape==(4,4)
        assert torch.isfinite(batch.controller.targets).all()
    finally:env.close()
