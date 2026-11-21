import copy
import functools
from types import SimpleNamespace
import torch
import pytest
from morphmimic.constants import GOAL_DIM, GOAL_K, HUMAN_FEAT_DIM, HUMAN_HEADING_DIM
from morphmimic.model.controller import ControllerCfg, NativeDelayController
from morphmimic.model.state import StateNormalizer
from morphmimic.model.world_model import WorldModel, WorldModelCfg
from morphmimic.model.policy import Policy, PolicyCfg, Critic, CriticCfg
from morphmimic.train.normalizers import RunningNormalizer
from morphmimic.train.imagination import imagine_rollout, ImaginationCfg
from morphmimic.train.replay import StartBatch
from tests.test_model_optimizations import state


def model():
    c=NativeDelayController(ControllerCfg(1,2,0,.01),torch.zeros(29),torch.ones(29),torch.ones(29,dtype=torch.bool))
    wm=WorldModel(WorldModelCfg(hidden=(16,),controller_mode='native_delay_v1'),StateNormalizer(),torch.tensor([[-3.,3.]]).expand(29,2),torch.zeros(29),controller=c)
    with torch.no_grad():wm.dynamics_mlp[-1].weight.normal_(std=.01)
    return wm


def test_conditioned_world_model_requires_actual_targets_and_uses_them():
    wm=model();s=state();a=torch.zeros(2,29);ctx=torch.zeros(2,33)
    with pytest.raises(ValueError,match='targets'):wm.step(s,a,ctx)
    targets=torch.randn(2,10,29,requires_grad=True)
    result=wm.step(s,a,ctx,controller_targets=targets)
    result.square().sum().backward()
    assert targets.grad[:,:2].abs().sum()>0 and targets.grad[:,2:].count_nonzero()==0
    torch.testing.assert_close(wm.rollout(s,a[:,None].expand(-1,2,-1),ctx,controller_targets=targets[:,None].expand(-1,2,-1,-1))[:,0],result)


def test_controller_checkpoint_identity_rejects_mode_and_plant_drift():
    from morphmimic.model.controller import validate_controller_checkpoint
    wm=model();ckpt={'cfg':{'controller_mode':'native_delay_v1'},'controller_identity':wm.controller.identity(),'wm':copy.deepcopy(wm.state_dict())}
    validate_controller_checkpoint(wm,ckpt)
    with pytest.raises(ValueError,match='controller'):validate_controller_checkpoint(wm,{'cfg':{}})
    ckpt['controller_identity']['timing']['physics_dt']=.02
    with pytest.raises(ValueError,match='identity'):validate_controller_checkpoint(wm,ckpt)

@pytest.mark.parametrize('interface',['motion','human'])
def test_real_policy_rollout_threads_queue_and_explicit_clock_with_gradients(interface):
    from morphmimic.train.imagination import imagine_step,imagine_step_human,frozen
    wm=model();p=Policy(PolicyCfg(enc_hidden=(16,),dec_hidden=(16,),hist_len=2,human=True))
    obs=RunningNormalizer(93);goal=RunningNormalizer(670)
    human=RunningNormalizer(HUMAN_FEAT_DIM);heading=RunningNormalizer(HUMAN_HEADING_DIM)
    lib=SimpleNamespace(clip_end=lambda f:torch.full_like(f,100),remaining=lambda f:torch.full_like(f,100),
                        state=lambda f:state(1)[0].expand(*f.shape,240))
    # Test starts from an explicit, already appended reset target, as native reset does.
    c0=wm.controller.transition(wm.controller.empty_state(2),torch.ones(2,29),torch.zeros(2,29),torch.zeros(2,3,dtype=torch.long),torch.zeros(2,dtype=torch.bool)).after
    start=StartBatch(state(),torch.zeros(2,2,93),torch.zeros(2,33),torch.zeros(2,dtype=torch.long),c0,torch.tensor([[True,False,True],[False,True,False]]))
    cfg=ImaginationCfg(horizon=3);seen=[]
    if interface=='motion':base=functools.partial(imagine_step,wm,p,obs,goal);extras=None
    else:
        base=functools.partial(imagine_step_human,wm,p,obs,goal,human,heading,'return')
        poses=torch.zeros(2,GOAL_K,24,7);poses[...,3]=1
        extras=lambda t:(torch.ones(2,24,dtype=torch.bool),torch.ones(2,GOAL_K,dtype=torch.bool),torch.ones(2,dtype=torch.bool),torch.zeros(2,GOAL_K,24,6),poses)
    def step(*args,**kw):
        before=kw['controller_state'];result=base(*args,**kw);seen.append((before,result.controller));return result
    with frozen(wm):
        ro=imagine_rollout(wm,p,None,lib,start,torch.zeros(2,5,93),cfg,obs,goal,step=step,extras=extras)
        ro.states[:,-1].square().sum().backward()
    assert len(seen)==3
    for t,(before,after) in enumerate(seen):
        torch.testing.assert_close(after.meta[:,1],before.meta[:,1]+2+start.reset_clock[:,t])
        if t:torch.testing.assert_close(before.queue,seen[t-1][1].queue)
    enc=p.encoder if interface=='motion' else p.human_encoder
    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in enc.parameters())
    assert all(x.grad is None for x in wm.parameters())
    with pytest.raises(ValueError,match='real replay history'):
        imagine_rollout(wm,p,None,lib,start._replace(controller=None),torch.zeros(2,5,93),cfg,obs,goal)


def test_fit_uses_logged_effective_targets_without_resampling_history():
    from morphmimic.train.replay import WindowBatch
    from morphmimic.model.controller import ControllerRecord
    wm=model();s=state();a=torch.randn(2,2,29);ctx=torch.zeros(2,33)
    c=wm.controller.empty_state(2)
    record=ControllerRecord(c,c,torch.randn(2,2,10,29),torch.zeros(2,2,10,dtype=torch.long),torch.zeros(2,2,dtype=torch.bool))
    b=WindowBatch(s[:,None].expand(-1,3,-1),a,torch.zeros(2,2),torch.zeros(2,2,dtype=torch.bool),ctx,torch.zeros(2,3,dtype=torch.long),record)
    rng=torch.random.get_rng_state().clone();loss,info=wm.fit_loss(b,None,refs=s[:,None].expand(-1,2,-1));loss.backward()
    assert torch.equal(rng,torch.random.get_rng_state()) and torch.isfinite(loss)
    assert wm.dynamics_mlp[0].weight.grad[:, -290:].abs().sum()>0


def test_delayed_action_to_world_model_gradient_has_exact_cadence_and_finite_difference():
    c=NativeDelayController(ControllerCfg(2,1,100,.01),torch.zeros(29),torch.ones(29),torch.ones(29,dtype=torch.bool))
    wm=WorldModel(WorldModelCfg(hidden=(),controller_mode='native_delay_v1'),StateNormalizer(),torch.tensor([[-3.,3.]]).expand(29,2),torch.zeros(29),controller=c)
    with torch.no_grad():
        # A raw-action shortcut would produce a premature derivative of 3.
        wm.dynamics_mlp[-1].weight[0,268+17]=3.
        wm.dynamics_mlp[-1].weight[0,-290]=.1
    def run(value):
        s=state(1);zero=torch.zeros(1,29);lags=torch.full((1,2),2);clock=torch.zeros(1,dtype=torch.bool)
        queue=c.transition(c.empty_state(1),zero,zero,lags,clock).after
        outputs=[]
        direction=torch.zeros(1,29);direction[:,0]=1
        for t in range(4):
            action=value*direction if t==0 else zero
            transition=c.transition(queue,action,zero,lags,clock)
            s=wm.step(s,action,torch.zeros(1,33),controller_targets=transition.targets)
            outputs.append(s[0,0]);queue=transition.next
        return torch.stack(outputs)
    value=torch.tensor(.4,requires_grad=True);out=run(value)
    derivative=torch.stack([torch.autograd.grad(x,value,retain_graph=True)[0] for x in out])
    torch.testing.assert_close(derivative,torch.tensor([0.,0.,.1,.1]),rtol=0,atol=1e-7)
    fd=(run(value.detach()+.001)-run(value.detach()-.001))/.002
    torch.testing.assert_close(derivative,fd,rtol=2e-4,atol=2e-6)


def test_mirror_transforms_controller_history_and_targets_only_on_selected_rows():
    from morphmimic.model.controller import ControllerRecord
    from morphmimic.train.mirror import RobotMirror
    from morphmimic.train.replay import WindowBatch
    mirror=RobotMirror(torch.arange(29),-torch.ones(29),torch.ones(29))
    c=model().controller;history=c.empty_state(2)._replace(queue=torch.randn(2,3,11,29),meta=torch.randint(0,10,(2,3,4)))
    record=ControllerRecord(history,history,torch.randn(2,3,10,29),torch.randint(0,3,(2,3,10)),torch.rand(2,3)>.5)
    batch=WindowBatch(state()[:,None].expand(-1,4,-1),torch.randn(2,3,29),torch.zeros(2,3),torch.zeros(2,3,dtype=torch.bool),torch.zeros(2,33),torch.zeros(2,4,dtype=torch.long),record)
    result,refs=mirror.window(batch,batch.state[:,1:],torch.tensor([True,False]))
    torch.testing.assert_close(result.controller.before.queue[0],-history.queue[0])
    torch.testing.assert_close(result.controller.before.queue[1],history.queue[1])
    torch.testing.assert_close(result.controller.targets[0],-record.targets[0])
    assert torch.equal(result.controller.lags,record.lags) and torch.equal(result.controller.clock,record.clock)
    restored,_=mirror.window(result,refs,torch.tensor([True,False]))
    torch.testing.assert_close(restored.controller.targets,record.targets)


def test_same_mode_network_optimizer_roundtrip_and_legacy_state_dict(tmp_path):
    from morphmimic.model.controller import validate_controller_checkpoint
    a=model();opt=torch.optim.Adam(a.parameters(),lr=.001)
    a.step(state(),torch.randn(2,29),torch.zeros(2,33),controller_targets=torch.randn(2,10,29)).square().sum().backward();opt.step()
    ckpt=dict(cfg={'controller_mode':'native_delay_v1'},controller_identity=a.controller.identity(),wm=a.state_dict(),optim=opt.state_dict())
    path=tmp_path/'ckpt.pt';torch.save(ckpt,path);saved=torch.load(path,weights_only=False)
    b=model();other=torch.optim.Adam(b.parameters(),lr=.001);validate_controller_checkpoint(b,saved);b.load_state_dict(saved['wm']);other.load_state_dict(saved['optim'])
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert all(int(v['step'])==1 for v in other.state.values())
    saved['wm']['controller.scale'][0]=5
    with pytest.raises(ValueError,match='buffers'):validate_controller_checkpoint(b,saved)
    legacy=WorldModel(WorldModelCfg(hidden=(16,)),StateNormalizer(),torch.tensor([[-3.,3.]]).expand(29,2),torch.zeros(29))
    assert legacy.dynamics_mlp[0].in_features==347
    assert not any(k.startswith('controller.') for k in legacy.state_dict())
    validate_controller_checkpoint(legacy,dict(cfg={}))
    legacy.load_state_dict(copy.deepcopy(legacy.state_dict()),strict=True)


def test_train_cli_controller_option_is_explicit_and_legacy_default():
    import tyro
    from morphmimic.scripts.train import TrainArgs
    assert TrainArgs().controller_mode=='legacy'
    args=tyro.cli(TrainArgs,args=['--controller-mode','native_delay_v1','--actuator-delay'])
    assert args.controller_mode=='native_delay_v1' and args.actuator_delay


def test_unqualified_controller_combinations_fail_before_environment_construction():
    from morphmimic.train.trainer import Trainer,TrainCfg
    with pytest.raises(ValueError,match='CPU adapter'):
        Trainer(TrainCfg(controller_mode='native_delay_v1'),'cpu')
    with pytest.raises(ValueError,match='delayed builtin'):
        Trainer(TrainCfg(controller_mode='native_delay_v1',actuator_delay=False),'cuda:0')
    with pytest.raises(ValueError,match='warm-start'):
        Trainer(TrainCfg(controller_mode='native_delay_v1',initial_checkpoint_root='/unused'),'cuda:0')
