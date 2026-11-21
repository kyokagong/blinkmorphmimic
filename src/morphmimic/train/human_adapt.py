"""Train a human encoder against an immutable, pinned motion teacher."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import torch
from morphmimic.train.optim import PolicyEMA
from morphmimic.train.trainer import TrainCfg,Trainer

PARENT_SHA256={
    'q1':'eef7a4d7b97186c26e766a49a5d6b86746314cfff7cd0516ab802728847847a1',
    'bumi':'a3fe26ee66127e5c103a34e8c44f13b5847161f46e8eb24bde8b5f6b40a286c7',
    'g1':'ecda4b965b87df618271cea374f4f8c69f0618a27d2765585904d6a46460d213',
}


def verify_adaptation_identity(ckpt,*,robot,update,data_sha256):
    cfg=ckpt.get('cfg',{});adapt=ckpt.get('human_adaptation',{});parent=adapt.get('parent',{})
    if (robot not in PARENT_SHA256 or cfg.get('robot')!=robot or ckpt.get('iteration')!=update
        or adapt.get('human_updates')!=update or parent.get('iteration')!=30000
        or parent.get('sha256')!=PARENT_SHA256[robot] or parent.get('weights','ema')!='ema'
        or not isinstance(data_sha256,str) or len(data_sha256)!=64
        or adapt.get('paired_manifest_sha256')!=data_sha256 or cfg.get('kp_fixed_horizon')!=1
        or not cfg.get('kp_interface') or cfg.get('policy_goal','motion')!='motion'):
        raise ValueError('Human checkpoint identity differs from its fixed parent, update or paired data')


def _raise_collective(error,context):
    errors=[error];dist=torch.distributed
    if dist.is_initialized():
        errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
    if any(errors):raise ValueError(context+': '+'; '.join(e for e in errors if e))


def _verify_distributed_pairs(root,robot,subset,expected_sha256=None,subset_sha256=None):
    from morphmimic.data.human_prepare import verify_prepared_subset
    digest=None;error=None
    try:
        if subset_sha256 is not None and file_sha(subset)!=subset_sha256:
            raise ValueError('Training subset SHA differs')
        digest=verify_prepared_subset(root,subset,robot=robot,split='train',expected_sha256=expected_sha256)
    except Exception as exc:error=f'{robot}: {type(exc).__name__}: {exc}'
    _raise_collective(error,'Human pair verification failed')
    return digest


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def teacher_fingerprint(trainer):
    groups={'wm':trainer.wm,'motion_encoder':trainer.policy.encoder,'decoder':trainer.policy.decoder,
            'critic':trainer.critic,'obs_norm':trainer.obs_norm,'goal_norm':trainer.goal_norm}
    result={}
    for name,module in groups.items():
        digest=hashlib.sha256()
        for key,tensor in sorted(module.state_dict().items()):
            digest.update(f'{key}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        result[name]=digest.hexdigest()
    return result


def initialize_teacher(trainer,path,expected_sha256):
    path=Path(path).absolute()
    if path.is_symlink() or path.resolve()!=path or path.name!='ckpt_30000.pt' or path.parent.name!=trainer.cfg.robot:
        raise ValueError('Teacher path identity does not match the robot')
    actual=file_sha(path)
    if actual!=expected_sha256:raise ValueError('Teacher SHA differs from pinned release')
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if ckpt['iteration']!=30000 or ckpt['cfg'].get('robot')!=trainer.cfg.robot or ckpt['cfg'].get('policy_goal','motion')!='motion':
        raise ValueError('Teacher identity/interface differs')
    state=ckpt.get('ema_policy',ckpt['policy'])
    if any(k.startswith('human_encoder.') for k in state):raise ValueError('Expected the original motion-only teacher')
    result=trainer.policy.load_state_dict(state,strict=False)
    if result.unexpected_keys or any(not k.startswith('human_encoder.') for k in result.missing_keys):
        raise ValueError(f'Teacher policy architecture differs: {result}')
    if trainer.policy.human_encoder is None:raise ValueError('Human encoder is required')
    for name,param in trainer.policy.named_parameters():param.requires_grad_(name.startswith('human_encoder.'))
    for name in ('wm','critic'):
        module=getattr(trainer,name);module.load_state_dict(ckpt[name]);module.requires_grad_(False)
    for name in ('obs_norm','goal_norm'):
        module=getattr(trainer,name);module.load_state_dict(ckpt[name]);module.eval()
    for name,key in (('target_critic','target_critic'),('critic_kp','critic'),('target_critic_kp','critic')):
        module=getattr(trainer,name,None)
        if module is not None:module.load_state_dict(ckpt.get(key,ckpt['critic']))
    trainer.policy_avg=PolicyEMA(trainer.policy,trainer.cfg.policy_ema) if trainer.cfg.policy_ema is not None else None
    trainer.it=0
    if file_sha(path)!=actual:raise ValueError('Teacher changed while loading')
    trainer.teacher_identity=dict(path=str(path),iteration=30000,sha256=actual,weights='ema')
    trainer.teacher_baseline=teacher_fingerprint(trainer)
    return trainer.teacher_identity


@dataclass
class HumanAdaptCfg(TrainCfg):
    human_teacher_root: str = ''


class HumanAdaptTrainer(Trainer):
    def __init__(self,cfg,device,*,resume=False):
        if not cfg.kp_interface or cfg.policy_goal!='motion' or cfg.kp_fixed_horizon!=1:
            raise ValueError('Human adaptation requires the motion teacher and causal human interface')
        dist=torch.distributed
        distributed=dist.is_initialized()
        rank=dist.get_rank() if distributed else 0
        robot=cfg.robots[rank] if cfg.robots else cfg.robot
        self.human_subset_path=Path(cfg.subset_yamls[rank] if cfg.robots else cfg.subset_yaml)
        self.human_manifest_path=Path(cfg.human_root).parent/'manifest.json'
        self.human_manifest_sha256=_verify_distributed_pairs(self.human_manifest_path.parent,robot,self.human_subset_path)
        self.human_subset_sha256=file_sha(self.human_subset_path)
        super().__init__(cfg,device,resume=resume)
        error=None
        try:
            root=Path(cfg.human_teacher_root)
            release=json.loads((root/'release_manifest.json').read_text())
            robot=self.cfg.robot;path=root/robot/'ckpt_30000.pt'
            if release['models'][robot]['checkpoint_sha256']!=PARENT_SHA256[robot]:
                raise ValueError('Parent release differs from the preserved 30000 teacher')
            initialize_teacher(self,path,release['models'][robot]['checkpoint_sha256'])
            if not bool(self.library.human_joint_valid[:,[0,7,8,20,21]].all()):
                raise ValueError('Every training clip must have all five tracked joints')
        except Exception as exc:error=f'{robot}: {type(exc).__name__}: {exc}'
        try:_raise_collective(error,'Teacher initialization failed')
        except Exception:self.close();raise

    @torch.no_grad()
    def estimate_residual_scale(self):
        # Parent warm-up calls this at update zero; the loaded teacher scale stays fixed.
        return self.wm.residual_scale.detach().clone()

    def _policy_fn(self,obs_hist_flat,goals):
        from morphmimic.model.human import human_features_from_goal
        available=self.library.human_valid_at(self.command.global_frame)
        features=human_features_from_goal(goals['goal_kp'],self.kp_point_mask)
        self.human_norm.train()
        try:self.human_norm.update(features[available.all(-1)])
        finally:self.human_norm.eval()
        return super()._policy_fn(obs_hist_flat,goals)

    def iteration(self):
        from morphmimic.train.trainer import _PhaseTimer
        from morphmimic.train.schedules import cosine_to
        if self.it==0:self.save_checkpoint()
        cfg=self.cfg
        if self.steps.cudagraphs:torch.compiler.cudagraph_mark_step_begin()
        lr_mult=cosine_to(self.it,cfg.lr_decay_iters,cfg.lr_final_frac)
        for opt,base in self._base_lr:
            for group in opt.param_groups:group['lr']=base*lr_mult
        phases=_PhaseTimer(self._cuda);phases.mark()
        n=self._collect_steps();collect=self.collector.step(n) if n>0 else None
        phases.mark();phases.mark()
        ro,metrics=self.improve_policy_kp()
        if self.critic_kp is not None:
            metrics|=self._fit_critic('critic_kp',self.critic_kp,self.critic_kp_opt,ro,
                                      self.steps.critic_targets_kp,self.steps.critic_loss_kp)
        if self.target_critic_kp is not None:
            with torch.no_grad():
                target=list(self.target_critic_kp.parameters());online=list(self.critic_kp.parameters())
                torch._foreach_lerp_(target,online,1-cfg.target_critic_alpha)
        if self.policy_avg is not None:self.policy_avg.update(self.policy)
        phases.mark();self.it+=1
        metrics.update({'lr/mult':lr_mult,'_collect':collect,'_phases':phases})
        return metrics

    def checkpoint(self,*args,**kwargs):
        if file_sha(self.human_manifest_path)!=self.human_manifest_sha256:
            raise ValueError('Paired manifest SHA differs')
        fingerprint=teacher_fingerprint(self)
        if fingerprint!=self.teacher_baseline:raise RuntimeError('Frozen teacher changed; checkpoint refused')
        result=super().checkpoint(*args,**kwargs)
        result['human_adaptation']=dict(parent=self.teacher_identity,human_updates=self.it,
            teacher_fingerprint=fingerprint,human_norm_sources='fully_calibrated_clips_only',
            available_joint_mask=True,primary_points='five',horizon=1)
        result['human_adaptation']['paired_manifest_sha256']=self.human_manifest_sha256
        result['human_adaptation']['train_subset_sha256']=self.human_subset_sha256
        return result

    def save_checkpoint(self):
        _verify_distributed_pairs(self.human_manifest_path.parent,self.cfg.robot,self.human_subset_path,self.human_manifest_sha256,self.human_subset_sha256)
        return super().save_checkpoint()

    def _accumulate(self,window,metrics):
        updated='kp/grad_norm' in metrics
        super()._accumulate(window,metrics)
        if updated:window.policy_iterations+=1

    def _add_perf(self,out,window,wall,ranks):
        super()._add_perf(out,window,wall,ranks)
        out['perf/imag_steps_per_s']=window.policy_iterations*self.cfg.kp_imag_batch*self.cfg.horizon*ranks/wall

    def _print_line(self,out):
        import math
        def metric(key):return out.get(key,math.nan)
        print(f"[morphmimic] human update {self.it} (parent 30000) | {metric('perf/it_per_s'):.2f} it/s"
              f" | J {metric('kp/return'):.4f} copy {metric('kp/action_err'):.4f} align {metric('kp/align_err'):.4f}"
              f" | nonfinite {metric('kp/nonfinite_steps'):.0f} | mem {metric('perf/gpu_mem_gb'):.2f} GB",flush=True)

    def _print_robot_line(self,robot,m):
        print(f"[morphmimic] {robot} | reward {m['collect/mean_reward']:.4f}"
              f" return {m['collect/mean_norm_return']:.3f} failures {m['collect/failures']:.0f}",flush=True)

    def load(self,path,**kwargs):
        error=None
        try:
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            verify_adaptation_identity(ckpt,robot=self.cfg.robot,update=ckpt['iteration'],data_sha256=self.human_manifest_sha256)
            if ckpt['human_adaptation'].get('train_subset_sha256')!=self.human_subset_sha256:
                raise ValueError('Resume training subset differs')
            if ckpt['cfg'].get('pool_norms',True)!=self.cfg.pool_norms:
                raise ValueError('Human resume must keep the pooled normalizers')
        except Exception as exc:error=f'{self.cfg.robot}: {type(exc).__name__}: {exc}'
        _raise_collective(error,'Human resume identity verification failed')
        del ckpt
        error=None;restored=None
        try:
            restored=super().load(path,**kwargs)
            if teacher_fingerprint(self)!=self.teacher_baseline:
                raise ValueError('Resume differs from the pinned adaptation teacher')
        except Exception as exc:error=f'{self.cfg.robot}: {type(exc).__name__}: {exc}'
        _raise_collective(error,'Human resume restoration failed')
        self.obs_norm.eval();self.goal_norm.eval()
        return restored
