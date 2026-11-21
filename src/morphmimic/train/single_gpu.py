"""One real process/device, robot-local simulations, equal-gradient shared updates.

This is an alternative executor, not the executor used by the formal 3-GPU run.
"""
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import time
import torch
from morphmimic.train.checkpoint_set import read_checkpoint_set, write_checkpoint_set, sha256
from morphmimic.train.optimizations import AsyncCheckpointWriter, cpu_snapshot
from morphmimic.train.update_phases import run_equal_phases


def validate_config(cfg):
    if cfg.executor != 'single_gpu_sequential':raise ValueError('Wrong executor selection')
    if not cfg.robots or len(set(cfg.robots))!=len(cfg.robots):raise ValueError('Distinct robots required')
    if not cfg.subset_yamls or len(cfg.subset_yamls)!=len(cfg.robots):raise ValueError('One subset per robot required')
    if cfg.freeze_receipt and (cfg.executor_subset_sha256 is None or len(cfg.executor_subset_sha256)!=len(cfg.robots)):
        raise ValueError('Frozen local executor requires executor_subset_sha256 for each robot')
    if cfg.gpu_ids is not None and len(cfg.gpu_ids)!=1:raise ValueError('Single-GPU executor requires exactly one GPU')
    if cfg.share!='all' or not cfg.pool_norms or cfg.policy_goal!='motion' or not cfg.kp_interface or cfg.kp_student!='return':
        raise ValueError('Single-GPU executor supports shared-all, pooled norms, motion+human return training')
    if cfg.latent_align_weight!=0:raise ValueError('Local executor requires explicit latent_align_weight=0')
    if cfg.initial_checkpoint_root or getattr(cfg,'human_teacher_root',''):
        raise ValueError('Cross-executor initialization/teacher import is not supported')
    if cfg.wandb:
        raise ValueError('Local executor currently logs each robot to TensorBoard; use wandb=False')
    if cfg.compile=='reduce-overhead':
        raise ValueError('Use compile=none or default; local executor supports explicit graphs, not automatic graph trees')


def _same(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(_same(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return isinstance(right,(list,tuple)) and len(left)==len(right) and all(_same(a,b) for a,b in zip(left,right))
    return left==right


def _validate_optimizer(opt,saved):
    if hasattr(opt,'muon'):
        if set(saved)!={'muon','adam'}:raise ValueError('Optimizer kind differs')
        _validate_optimizer(opt.muon,saved['muon']);_validate_optimizer(opt.adam,saved['adam']);return
    groups=saved.get('param_groups',[])
    if len(groups)!=len(opt.param_groups):raise ValueError('Optimizer groups differ')
    params={}
    for group,current in zip(groups,opt.param_groups):
        if len(group['params'])!=len(current['params']):raise ValueError('Optimizer parameter counts differ')
        params.update(zip(group['params'],current['params']))
    for index,state in saved.get('state',{}).items():
        if index not in params:raise ValueError('Unknown optimizer parameter state')
        for key,value in state.items():
            if isinstance(value,torch.Tensor):
                if not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite optimizer state')
                if key=='step':
                    if value.numel()!=1 or bool(value<0):raise ValueError('Invalid optimizer step')
                elif value.shape!=params[index].shape:
                    raise ValueError('Optimizer moment shape differs')


class SingleGpuMultiRobotExecutor:
    def __init__(self,cfg,device,*,resume=False):
        from morphmimic.train.distributed import rank_and_world_size
        from morphmimic.train.trainer import Trainer
        validate_config(cfg)
        if rank_and_world_size()!=(0,1):raise ValueError('Local executor cannot join a process group')
        if torch.device(device).type!='cuda':raise ValueError('Native local executor requires one CUDA device')
        if not cfg.run_name:cfg=replace(cfg,run_name=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.cfg,self.device=cfg,device
        self.subset_hashes={r:sha256(p) for r,p in zip(cfg.robots,cfg.subset_yamls)}
        if cfg.executor_subset_sha256 is not None and list(cfg.executor_subset_sha256)!=list(self.subset_hashes.values()):
            raise ValueError('Local executor subset SHA differs from configuration')
        self.run_dir=Path(cfg.log_dir)/cfg.run_name
        if not resume and self.run_dir.exists():raise FileExistsError('Fresh local run directory already exists')
        self.trainers=[];self.accumulators={};self.writer=None;self.stage_updates={}
        try:
            for index,_ in enumerate(cfg.robots):
                t=Trainer(cfg,device,resume=resume,local_robot_index=index,
                          parameter_owner=self.trainers[0] if self.trainers else None)
                self.trainers.append(t)
                # Same gate and LR trigger for every body, but local loss scales/replay.
                t._actor_lr=self.owner._actor_lr;t._warmup_gate=self.owner._warmup_gate
            self.writer=AsyncCheckpointWriter() if cfg.async_checkpoint else None
        except BaseException:
            self.close();raise

    @property
    def owner(self):return self.trainers[0]
    @property
    def it(self):return self.owner.it

    def identity(self):
        return dict(schema=1,executor='single_gpu_sequential',robots=list(self.cfg.robots),
                    world_size=1,robot_count=len(self.trainers),weighting='equal_robot',
                    num_envs_per_robot=self.cfg.num_envs,total_envs=self.cfg.num_envs*len(self.trainers),
                    parameter_owner=self.cfg.robots[0],iteration=self.it,training_subsets=self.subset_hashes,
                    stage_order=['wm','policy','critic','human','critic_kp'])

    @torch.no_grad()
    def collect(self,n):
        totals=[dict(steps=0,episodes_finished=0,failures=0,reward=0.,length=0.,returns=0.) for _ in self.trainers]
        for _ in range(n):
            inputs=[t.collection_normalizer_inputs(t.collector.history.flat(),t.collector.obs) for t in self.trainers]
            for name in inputs[0]:getattr(self.owner,name).update_many([v[name] for v in inputs])
            for t,total in zip(self.trainers,totals):
                t._collection_norms_preupdated=True
                try:stats=t.collector.step(1)
                finally:t._collection_norms_preupdated=False
                total['steps']+=stats['steps'];total['episodes_finished']+=stats['episodes_finished'];total['failures']+=stats['failures']
                total['reward']+=stats['mean_reward']*stats['steps']
                if stats['episodes_finished']:
                    total['length']+=stats['mean_episode_length']*stats['episodes_finished']
                    total['returns']+=stats['mean_norm_return']*stats['episodes_finished']
        return [dict(steps=x['steps'],episodes_finished=x['episodes_finished'],failures=x['failures'],
                     mean_reward=x['reward']/x['steps'] if x['steps'] else math.nan,
                     mean_episode_length=x['length']/x['episodes_finished'] if x['episodes_finished'] else math.nan,
                     mean_norm_return=x['returns']/x['episodes_finished'] if x['episodes_finished'] else math.nan) for x in totals]

    def iteration(self):
        from morphmimic.train.trainer import _PhaseTimer, cosine_to
        cfg=self.cfg;lr=cosine_to(self.it,cfg.lr_decay_iters,cfg.lr_final_frac)
        for opt,base in self.owner._base_lr:
            factor=self.owner._actor_lr.factor(self.it) if opt in (self.owner.policy_opt,self.owner.kp_opt) else 1.
            for group in opt.param_groups:
                value=base*lr*factor
                if isinstance(group['lr'],torch.Tensor):group['lr'].fill_(value)
                else:group['lr']=value
        # Interleaved learning has one global interval, not per-robot phase times.
        timer=_PhaseTimer(True,names=("global_collect","global_learning"))
        timer.mark()
        counts=[t._collect_steps() for t in self.trainers]
        if len(set(counts))!=1:raise ValueError('Robot collection schedules differ')
        collected=self.collect(counts[0]) if counts[0] else [None]*len(self.trainers)
        timer.mark()
        results=run_equal_phases([t.learning_phases(update_targets=False) for t in self.trainers],self.accumulators,self._record_step)
        ready=[value[1] for value in results]
        if len(set(ready))!=1:raise RuntimeError('Robot readiness gates diverged')
        if ready[0]:
            self.owner._update_target()
            if self.owner.policy_avg is not None:
                self.owner.policy_avg.update_parameters(self.owner.policy)
                for t in self.trainers:
                    t.policy_avg.copy_buffers(t.policy)
                    t.policy_avg.updates=self.owner.policy_avg.updates
        timer.mark()
        metrics=[]
        for t,(out,_),collect in zip(self.trainers,results,collected):
            t.it+=1
            out.update({'lr/mult':lr,'lr/actor_multiplier':t._actor_lr.factor(t.it),
                        'optimization/policy_updated':float(ready[0]),'_collect':collect,'_phases':timer,
                        'executor/physical_world_size':1.,'executor/robot_count':float(len(self.trainers))})
            metrics.append(out)
        return metrics

    def _record_step(self,name):
        self.stage_updates[name]=self.stage_updates.get(name,0)+1

    def _log(self,windows):
        # Schedule reductions must precede each local logger; no robot-order trigger.
        key='collect/mean_norm_return';weight=sum(w.weights.get(key,0) for w in windows)
        if any(w.policy_iterations for w in windows):
            pooled=sum(w.sums.get(key,0) for w in windows)/weight if weight else math.nan
            self.owner._actor_lr.observe(self.it,float(pooled))
        validation=[]
        if self.cfg.adaptive_warmup and self.owner._warmup_gate.ready_at is None:
            with torch.no_grad():
                for t in self.trainers:
                    start=t.cfg.num_envs-t.cfg.wm_validation_envs
                    batch=t.replay.sample_windows(64,t.cfg.window_k,env_range=(start,t.cfg.num_envs))
                    _,info=t.wm.fit_loss(batch,t.library)
                    value=float(info['pred_err_8step']);validation.append(value if math.isfinite(value) else math.inf)
            ready=self.owner._warmup_gate.observe(self.it,validation)
            for w in windows:
                w.mean('wm/validation_pred_rms_worst',max(validation));w.mean('optimization/warmup_ready',float(ready))
        for t,w in zip(self.trainers,windows):t._log(w,observe_schedules=False)

    def train(self, *, stop_at=None):
        from morphmimic.train.trainer import _Window, check_residual_scale
        cfg=self.cfg
        end=cfg.iterations if stop_at is None else stop_at
        if not self.it < end <= cfg.iterations:
            raise ValueError("Stop boundary must be after current iteration and within frozen target")
        for t in self.trainers:t._open_loggers()
        try:
            for t in self.trainers:t.collector.reset_all()
            started=time.perf_counter();self.collect(cfg.warmup_env_steps)
            print(f'[single_gpu] {len(self.trainers)} robots x {cfg.num_envs} envs on {self.device}; warmup {time.perf_counter()-started:.2f}s',flush=True)
            for t in self.trainers:
                if cfg.residual_scale=='step':
                    if self.it==0 and t.resumed_from is None:t.wm.set_residual_scale(t.estimate_residual_scale())
                    check_residual_scale(t.wm.residual_scale,0,1,same_across_ranks=False)
                # Reject a requested human stage with no valid starts before learning.
                if not bool(t._human_start_filter(t.library.clip_start).any()):
                    raise ValueError('No eligible human training starts')
            if self.it==0 and self.owner.resumed_from is None:self.save_checkpoint()
            windows=[_Window() for _ in self.trainers]
            while self.it<end:
                for t,w,m in zip(self.trainers,windows,self.iteration()):t._accumulate(w,m)
                if self.it%cfg.log_every==0:
                    self._log(windows);windows=[_Window() for _ in self.trainers]
                if self.it%cfg.ckpt_every==0:self.save_checkpoint()
            if any(w.iterations for w in windows):self._log(windows)
            if self.it%cfg.ckpt_every:self.save_checkpoint()
            if self.writer:self.writer.flush()
        finally:
            for t in self.trainers:t._close_loggers()

    def save_checkpoint(self):
        from morphmimic.train.trainer import rng_state
        if len({t.it for t in self.trainers})!=1:raise ValueError('Robot iterations differ')
        identity=self.identity()
        states={t.cfg.robot:dict(t.checkpoint(),local_executor=identity) for t in self.trainers}
        coordinator=dict(iteration=self.it,local_executor=identity,config=asdict(self.cfg),rng=rng_state(),stage_updates=dict(self.stage_updates))
        snapshot=cpu_snapshot({'robots':states,'coordinator':coordinator})
        if self.writer:
            self.writer.flush()
            self.writer.pending=self.writer.executor.submit(write_checkpoint_set,self.run_dir,snapshot)
        else:write_checkpoint_set(self.run_dir,snapshot)
        return self.run_dir/'commits'/f'iteration_{self.it}.json'

    def load(self,path,explicit=()):
        from morphmimic.train.trainer import set_rng_state
        from morphmimic.constants import GOAL_VERSION, KP_GOAL_VERSION
        source=Path(path)
        source_root=source if source.is_dir() else source.parent.parent
        if source_root.resolve()!=self.run_dir.resolve():
            raise ValueError("Resume root must match the original local executor configuration")
        states,coordinator=read_checkpoint_set(path)
        canonical=lambda v:json.dumps(v,sort_keys=True)
        if canonical(coordinator['config'])!=canonical(asdict(self.cfg)):
            raise ValueError('Local executor resume requires the exact original configuration')
        expected=dict(self.identity(),iteration=coordinator['iteration'])
        if coordinator['local_executor']!=expected:raise ValueError('Executor ownership identity differs')
        owner=states[self.cfg.robots[0]]
        modules=('wm','policy','critic','critic_kp','target_critic','target_critic_kp')
        # Validate ALL files and sharing before mutating any module or optimizer.
        for t in self.trainers:
            state=states[t.cfg.robot]
            if state.get('inference_only'):raise ValueError('Inference-only checkpoint cannot resume training')
            if (t.cfg.controller_mode != 'legacy' or state.get('cfg',{}).get('controller_mode','legacy') != 'legacy'
                or state.get('controller_identity') is not None):
                from morphmimic.model.controller import validate_controller_checkpoint
                validate_controller_checkpoint(t.wm,state)
            if canonical(state['cfg'])!=canonical(asdict(t.cfg)):raise ValueError('Local robot configuration differs')
            if state.get('goal_version') != GOAL_VERSION or state.get('kp_goal_version') != KP_GOAL_VERSION:
                raise ValueError('Checkpoint goal input version differs')
            if state.get('imu_body') != t.command.spec.imu_body:
                raise ValueError('Checkpoint IMU body differs')
            if state.get('schedule', {}).get('world_size') != 1:
                raise ValueError('Checkpoint physical world size differs')
            for name in modules:
                module=getattr(t,name,None)
                if module is None:continue
                actual=module.state_dict();saved=state[name]
                if actual.keys()!=saved.keys() or any(actual[k].shape!=saved[k].shape or actual[k].dtype!=saved[k].dtype for k in actual):
                    raise ValueError(f'Local module signature differs: {name}')
                if any(not bool(torch.isfinite(v).all()) for v in saved.values()):
                    raise ValueError(f'Nonfinite local module state: {name}')
                for key in dict(module.named_parameters()):
                    if not torch.equal(saved[key],owner[name][key]):raise ValueError(f'Shared parameters differ: {name}.{key}')
            for key,name in (('wm','wm_opt'),('policy','policy_opt'),('kp','kp_opt'),('critic','critic_opt'),('critic_kp','critic_kp_opt')):
                _validate_optimizer(getattr(t,name),state['optim'][key])
            for name in ('optim','obs_norm','goal_norm','human_norm','heading_norm','optimization_state','ema_updates'):
                if not _same(state.get(name),owner.get(name)):raise ValueError(f'Shared state differs: {name}')
            for name in ('obs_norm','goal_norm','human_norm','heading_norm','kp_loss_scales'):
                module=getattr(t,name,None)
                if module is None:continue
                actual=module.state_dict();saved=state.get(name,{})
                if actual.keys()!=saved.keys() or any(actual[k].shape!=saved[k].shape or actual[k].dtype!=saved[k].dtype or not bool(torch.isfinite(saved[k]).all()) for k in actual):
                    raise ValueError(f'Local normalization signature differs: {name}')
            coverage=state.get('coverage',{})
            if coverage.get('clip_names')!=t.library.clip_names or set(coverage.get('counts',{}))!=set(t.library.usage):
                raise ValueError('Checkpoint local motion inventory differs')
            for name,value in t.library.usage.items():
                saved=coverage['counts'][name]
                if saved.shape!=value.shape or saved.dtype!=value.dtype or bool((saved<0).any()):
                    raise ValueError('Checkpoint coverage counts differ')
            from morphmimic.train.dataset_transition import human_coverage_state
            human_coverage_state(state.get('human_coverage'), t.library, int(state['iteration']))
            t._validate_sampler_state(state,strict=True)
            source=coverage.get('source');current_source=t.library.source_coverage_state()
            if current_source:
                if not isinstance(source,dict) or any(source.get(k)!=current_source[k] for k in ('start','stop','has_phases')):
                    raise ValueError('Checkpoint source windows differ')
                if set(source.get('counts',{}))!=set(current_source['counts']) or type(source.get('since_iteration')) is not int or source['since_iteration']<0:
                    raise ValueError('Invalid source coverage state')
                for key,counts in source['counts'].items():
                    value=torch.as_tensor(counts)
                    if value.shape!=torch.as_tensor(current_source['counts'][key]).shape or value.dtype!=torch.long or bool((value<0).any()):
                        raise ValueError('Invalid source coverage counts')
            if t.policy_avg:
                current=t.policy_avg.module.state_dict();saved=state.get('ema_policy',{})
                if current.keys()!=saved.keys() or any(current[k].shape!=saved[k].shape or current[k].dtype!=saved[k].dtype or not bool(torch.isfinite(saved[k]).all()) for k in current):
                    raise ValueError('EMA state signature differs')
                for key in dict(t.policy_avg.module.named_parameters()):
                    if not torch.equal(state['ema_policy'][key],owner['ema_policy'][key]):raise ValueError('Shared EMA differs')
            if t.cfg.optimization_schema:
                t._manifest_identity.verify_checkpoint(state,t.cfg.freeze_receipt)
                if state['optimized_training']['contact_floor_sha256']!=t._contact_floor_sha256:raise ValueError('Floor identity differs')
        for index,t in enumerate(self.trainers):
            checkpoint=self.run_dir/t.cfg.robot/f"ckpt_{coordinator['iteration']}.pt"
            t.load(checkpoint,explicit=explicit,checkpoint_state=states[t.cfg.robot],restore_shared=index==0,restore_rng=False)
            t._actor_lr=self.owner._actor_lr;t._warmup_gate=self.owner._warmup_gate
        self.stage_updates=dict(coordinator['stage_updates'])
        set_rng_state(coordinator['rng'])
        return ['committed robot set','shared optimizers','local buffers','single-process RNG']

    def close(self):
        try:
            if self.writer:self.writer.close()
        finally:
            for t in self.trainers:t.close()
