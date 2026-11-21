import torch
import pytest
from morphmimic.model.controller import ControllerCfg,ControllerState,NativeDelayController


def controller(decimation=2,max_lag=3,period=100):
    return NativeDelayController(ControllerCfg(max_lag=max_lag,decimation=decimation,update_period=period,physics_dt=.01),
        torch.zeros(29),torch.ones(29),torch.ones(29,dtype=torch.bool))


def test_native_delay_backfill_clamp_and_global_reset_clock_are_explicit():
    c=controller();state=c.empty_state(1);a=torch.ones(1,29)
    result=c.transition(state,a,torch.zeros_like(a),torch.full((1,3),3),torch.tensor([False]))
    torch.testing.assert_close(result.targets[:,:2],torch.ones(1,2,29))
    assert result.after.meta.tolist()==[[3,2,2,0]]
    assert result.next.meta.tolist()==[[3,2,2,0]]
    second=c.transition(result.next,2*a,torch.zeros_like(a),torch.full((1,3),3),torch.tensor([True]))
    torch.testing.assert_close(second.targets[:,:2],torch.ones(1,2,29))
    assert second.after.meta.tolist()==[[3,4,4,0]]
    assert second.next.meta.tolist()==[[3,5,5,0]]
    noreset=c.transition(result.next,2*a,torch.zeros_like(a),torch.full((1,3),3),torch.tensor([False]))
    future=c.transition(second.next,3*a,torch.zeros_like(a),torch.full((1,3),3),torch.tensor([False]))
    other=c.transition(noreset.next,3*a,torch.zeros_like(a),torch.full((1,3),3),torch.tensor([False]))
    assert not torch.equal(future.targets,other.targets)


def test_history_is_markov_given_innovations_and_gradient_selects_delayed_action():
    c=controller(decimation=1,max_lag=1,period=0)
    old=torch.randn(1,29,requires_grad=True);new=torch.randn(1,29,requires_grad=True)
    first=c.transition(c.empty_state(1),old,torch.zeros_like(old),torch.zeros(1,2,dtype=torch.long),torch.tensor([False]))
    result=c.transition(first.next,new,torch.zeros_like(new),torch.ones(1,2,dtype=torch.long),torch.tensor([False]))
    torch.testing.assert_close(result.targets[:,0],old)
    result.targets.sum().backward();torch.testing.assert_close(old.grad,torch.ones_like(old));torch.testing.assert_close(new.grad,torch.zeros_like(new))
    restored=ControllerState(*(x.detach().clone() for x in first.next))
    repeat=c.transition(restored,new.detach(),torch.zeros_like(new),torch.ones(1,2,dtype=torch.long),torch.tensor([False]))
    torch.testing.assert_close(result.targets,repeat.targets)


def test_mapping_clip_bias_and_partial_reset():
    c=NativeDelayController(ControllerCfg(3,2,100,.01),torch.full((29,),.5),torch.full((29,),.2),
        torch.ones(29,dtype=torch.bool),torch.full((29,),-.2),torch.full((29,),.6))
    a=torch.ones(2,29);bias=torch.full_like(a,.1)
    torch.testing.assert_close(c.target(a,bias),torch.full_like(a,.5))
    state=c.transition(c.empty_state(2),a,bias,torch.zeros(2,3,dtype=torch.long),torch.zeros(2,dtype=torch.bool)).next
    reset=c.reset_state(state,torch.tensor([True,False]))
    assert reset.queue[0].count_nonzero()==0 and reset.meta[0].count_nonzero()==0
    torch.testing.assert_close(reset.queue[1],state.queue[1])
    with pytest.raises(ValueError,match='clock'):c.transition(state,a,bias,torch.zeros(2,3,dtype=torch.long),None)

@pytest.mark.parametrize('decimation,max_lag,dt', [(10,10,.002),(4,4,.005)])
@pytest.mark.parametrize('period', [0,7,2**40])
def test_native_delay_buffer_exact_substeps_partial_resets_and_global_clock(decimation,max_lag,dt,period):
    from mjlab.utils.buffers import DelayBuffer
    from morphmimic.env.controller_recording import read_delay_state
    native=DelayBuffer(max_lag=max_lag,batch_size=3,update_period=period,per_env_phase=True)
    c=NativeDelayController(ControllerCfg(max_lag,decimation,period,dt,per_env_phase=True),torch.zeros(29),torch.ones(29),torch.ones(29,dtype=torch.bool))
    # First reset-style enqueue is an actual native write, not an invented history.
    native.append(torch.randn(3,29));native.compute()
    for t in range(7):
        before=read_delay_state(native,lambda x:x)
        action=torch.randn(3,29);targets=[];lags=[]
        for _ in range(decimation):
            native.append(action);targets.append(native.compute().clone());lags.append(native.current_lags.clone())
        after=read_delay_state(native,lambda x:x)
        clock=t%2==0
        extra=native.current_lags.clone()
        if clock:
            native.reset([0]);native.append(action);native.compute();extra=native.current_lags.clone()
        expected=read_delay_state(native,lambda x:x)
        candidates=torch.stack(lags+[extra],-1)
        result=c.transition(before,action,torch.zeros_like(action),candidates,torch.full((3,),clock))
        torch.testing.assert_close(result.targets[:,:decimation],torch.stack(targets,1),rtol=0,atol=0)
        for a,b in zip(result.after,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Own reset world is initialized by native; only surviving worlds use conditional continuation.
        for a,b in zip(result.next,expected):torch.testing.assert_close(a[1:],b[1:],rtol=0,atol=0)


def test_transition_fullgraph_compile_preserves_history_and_gradients():
    c=controller(decimation=2,max_lag=3,period=7)
    fn=torch.compile(c.transition,backend='eager',fullgraph=True)
    s=c.empty_state(2);a=torch.randn(2,29,requires_grad=True);bias=torch.zeros_like(a)
    lags=torch.tensor([[2,1,3],[1,3,2]]);clock=torch.tensor([True,False])
    actual=fn(s,a,bias,lags,clock);expected=c.transition(s,a,bias,lags,clock)
    torch.testing.assert_close(actual.targets,expected.targets)
    for x,y in zip(actual.next,expected.next):torch.testing.assert_close(x,y)
    actual.targets.sum().backward();assert torch.isfinite(a.grad).all()

@pytest.mark.parametrize('robot',['q1','bumi','g1'])
def test_real_robot_mapping_bias_clip_and_native_joint_order(robot):
    import re
    from morphmimic.robots import get_robot
    spec=get_robot(robot);tr=spec.translator;cfg=spec.entity_cfg_fn()
    def resolve(table,name,default=None):
        if isinstance(table,(int,float)):return table
        matches=[v for pattern,v in table.items() if re.fullmatch(pattern,name)]
        assert len(matches)<=1 and (matches or default is not None),(name,matches)
        return matches[0] if matches else default
    import mujoco
    native_model=cfg.spec_fn().compile()
    defaults=[float(native_model.qpos0[native_model.jnt_qposadr[mujoco.mj_name2id(native_model,mujoco.mjtObj.mjOBJ_JOINT,n)]]) for n in spec.joint_names]
    q0=torch.tensor([resolve(cfg.init_state.joint_pos,n,default) for n,default in zip(spec.joint_names,defaults)])
    scale=torch.tensor([resolve(spec.action_scale,n) for n in spec.joint_names])
    lower=q0-.15;upper=q0+.25;limits=tr.limits_to_canonical(torch.stack((lower,upper),-1))
    c=NativeDelayController(ControllerCfg(round(.02/spec.physics_timestep),spec.physics_decimation,2**40,spec.physics_timestep),
        tr.to_canonical(q0),tr.to_canonical_delta(scale).abs(),torch.tensor(tr.joint_mask),limits[:,0],limits[:,1])
    action=torch.randn(4,29)*3;native_bias=torch.randn(4,tr.num_native)*.01
    native=(q0+tr.to_native_delta(action)*scale).clamp(lower,upper)-native_bias
    torch.testing.assert_close(c.target(action,tr.to_canonical_delta(native_bias)),tr.to_canonical(native),rtol=1e-6,atol=2e-7)
    assert c.cfg.decimation*spec.physics_timestep==pytest.approx(.02)


def test_no_clip_does_not_replace_native_overflow_with_finite_saturation():
    c=controller();c.scale.fill_(2.)
    out=c.target(torch.full((1,29),torch.finfo(torch.float32).max),torch.zeros(1,29))
    assert out.isinf().all()
    assert not c.clip_enabled and torch.isfinite(c.lower).all() and torch.isfinite(c.upper).all()
