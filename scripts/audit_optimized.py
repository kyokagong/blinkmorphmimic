"""Audit fixed joint checkpoints on CPU without creating simulation environments."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from morphmimic.constants import CRITIC_PRIV_DIM,NUM_JOINTS
from morphmimic.model.policy import Policy,PolicyCfg,Critic,CriticCfg
from morphmimic.model.state import StateNormalizer
from morphmimic.model.world_model import WorldModel,WorldModelCfg
from morphmimic.train.run_identity import _validate_receipt, read_receipt, _canonical
from scripts.evaluate_optimized import validate_selection
from scripts.prepare_optimized import ROBOTS,sha


def finite_tree(value,path='checkpoint'):
    if isinstance(value,torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f'nonfinite tensor: {path}')
    elif isinstance(value,dict):
        for key,child in value.items():finite_tree(child,f'{path}.{key}')
    elif isinstance(value,(list,tuple)):
        for index,child in enumerate(value):finite_tree(child,f'{path}[{index}]')


def corpus_identity(manifest_path):
    raw=Path(manifest_path).read_bytes();digest=hashlib.sha256(raw).hexdigest();manifest=json.loads(raw)
    if manifest.get('complete') is not True or not isinstance(manifest.get('clips'),list):
        raise ValueError('Incomplete composite manifest')
    root=Path(manifest_path).resolve().parent;subsets={};counts={}
    for robot in ROBOTS:
        rows=[r for r in manifest['clips'] if r.get('robot')==robot and r.get('split')=='train']
        families=Counter(r.get('dataset') for r in rows)
        if set(families)!={'seed','lafan1','motiondecode'}:raise ValueError(f'Missing training corpus for {robot}')
        subset=root/robot/'train.yaml'
        text=subset.read_text()
        try:spec=json.loads(text)
        except json.JSONDecodeError:
            import yaml
            spec=yaml.safe_load(text)
        clip_root=Path(spec['root']);clip_root=clip_root if clip_root.is_absolute() else subset.parent/clip_root
        paths=[str((clip_root/clip).resolve()) for clip in spec['clips']]
        expected=[str(Path(r['output']).resolve()) for r in rows]
        if not paths or len(set(paths))!=len(paths) or len(set(expected))!=len(expected) or set(paths)!=set(expected):
            raise ValueError(f'Training subset differs from manifest: {robot}')
        subsets[robot]=str(subset.resolve());counts[robot]=dict(families)
    return digest,subsets,counts


def construct_models(cfg,morph,*,controller_identity=None):
    """Build architecture from configuration, validating conditional plant identity on CPU.

    Legacy physical buffers retain the old audit placeholders. Controller buffers
    are reconstructed from the actual compiled plant and verified, never inferred
    from checkpoint weight widths or trusted metadata alone.
    """
    controller_mode=cfg.get('controller_mode','legacy')
    controller=None
    if controller_mode=='native_delay_v1':
        if controller_identity is None:
            raise ValueError('Missing controller identity for native_delay_v1')
        from morphmimic.env.controller_recording import controller_from_config
        controller=controller_from_config(cfg)
        if controller.identity()!=controller_identity:
            raise ValueError('Checkpoint controller identity differs from configured native plant')
    elif controller_mode!='legacy' or controller_identity is not None:
        raise ValueError('Invalid controller mode or unexpected legacy controller identity')
    actor_dim=int(morph['actor'].numel());wm_dim=int(morph['wm'].numel())
    policy=Policy(PolicyCfg(enc_hidden=tuple(cfg['policy_hidden']),dec_hidden=tuple(cfg['policy_hidden']),
        act=cfg['act'],hist_len=cfg['hist_len'],latent=cfg['latent'],human=cfg['kp_interface'],
        recon=cfg['policy_goal']=='human',ref_reconstruction_skip=cfg.get('ref_reconstruction_skip',False),
        bf16=cfg['bf16'],morph_dim=actor_dim,film=bool(morph['film'])))
    mode=cfg['wm_state_mode']
    wm=WorldModel(WorldModelCfg(hidden=tuple(cfg['wm_hidden']),act=cfg['act'],termination=cfg['termination'],
        layer_norm=cfg['wm_layer_norm'],bf16=cfg['bf16'],frame=cfg['wm_frame'],residual_scale=cfg['residual_scale'],
        morph_dim=wm_dim,joint_dynamics=cfg['wm_joint_dynamics'],state_mode=mode,
        **({} if controller is None else dict(controller_mode=controller_mode,
            dt=controller.cfg.physics_dt*controller.cfg.decimation))),
        StateNormalizer(),torch.zeros(NUM_JOINTS,2),torch.zeros(NUM_JOINTS),
        kinematics=nn.Identity() if mode=='compact_fk' else None,
        **({} if controller is None else dict(controller=controller)))
    critic=Critic(CriticCfg(hidden=tuple(cfg['critic_hidden']),act=cfg['act'],layer_norm=cfg['critic_layer_norm'],
        bf16=cfg['bf16'],morph_dim=wm_dim,priv_dim=CRITIC_PRIV_DIM if cfg['critic_anchor_pos'] else 0))
    return policy,wm,critic


def internal_parameters(module,networks):
    hidden=set()
    for network in networks:
        layers=[m for m in network.modules() if isinstance(m,nn.Linear)]
        hidden.update(id(layer.weight) for layer in layers[1:-1])
    return [(name,p) for name,p in module.named_parameters() if id(p) in hidden]


def require_muon(saved,named,step_count,label):
    if not named:raise ValueError(f'{label}: no internal Muon matrices')
    try:
        muon=saved['muon'];adam=saved['adam'];groups=muon['param_groups'];state=muon['state']
        ids=[i for group in groups for i in group['params']]
    except (KeyError,TypeError) as e:raise ValueError(f'{label}: missing Muon optimizer state') from e
    if len(ids)!=len(named) or len(set(ids))!=len(ids):raise ValueError(f'{label}: Muon internal group membership differs')
    momentum_norm=0.
    for index,(name,param) in zip(ids,named):
        entry=state.get(index,{})
        buffer=entry.get('momentum_buffer') if isinstance(entry,dict) else None
        if not isinstance(buffer,torch.Tensor) or buffer.shape!=param.shape or not bool(torch.isfinite(buffer).all()):
            raise ValueError(f'{label}: missing or invalid Muon momentum for {name}')
        momentum_norm+=float(torch.linalg.vector_norm(buffer.double()).item())
    if momentum_norm<=0:raise ValueError(f'{label}: Muon momentum is entirely zero')
    adam_states=adam.get('state',{}) if isinstance(adam,dict) else {}
    steps=[]
    for entry in adam_states.values():
        if isinstance(entry,dict) and 'step' in entry:
            value=entry['step'];steps.append(int(value.item() if isinstance(value,torch.Tensor) else value))
    if steps and (max(steps)>step_count or max(steps)<1):raise ValueError(f'{label}: Adam step state differs')
    return dict(internal_matrices=[name for name,_ in named],momentum_l2_sum=momentum_norm,
                adam_state_steps_min=min(steps) if steps else None,adam_state_steps_max=max(steps) if steps else None)


def require_adam(saved, named, step_count, label):
    """Verify every expected parameter has actual Adam history at the logged step."""
    try:
        ids=[i for group in saved['param_groups'] for i in group['params']]
        states=saved['state']
    except (KeyError,TypeError) as e:
        raise ValueError(f'{label}: missing Adam optimizer state') from e
    if not named or len(ids)!=len(named) or len(set(ids))!=len(ids) or set(states)!=set(ids):
        raise ValueError(f'{label}: Adam parameter membership or state differs')
    first_norm=second_norm=0.
    for index,(name,param) in zip(ids,named):
        state=states[index]
        step=state.get('step') if isinstance(state,dict) else None
        if isinstance(step,torch.Tensor):
            if step.numel()!=1 or not bool(torch.isfinite(step).all()):
                raise ValueError(f'{label}: invalid Adam step for {name}')
            step=step.item()
        if isinstance(step,bool) or not isinstance(step,(int,float)) or step!=step_count or step_count<1:
            raise ValueError(f'{label}: Adam step differs for {name}')
        for key in ('exp_avg','exp_avg_sq'):
            value=state.get(key)
            if not isinstance(value,torch.Tensor) or value.shape!=param.shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f'{label}: invalid Adam {key} for {name}')
        if bool((state['exp_avg_sq']<0).any()):
            raise ValueError(f'{label}: negative Adam second moment for {name}')
        first_norm+=float(torch.linalg.vector_norm(state['exp_avg'].double()))
        second_norm+=float(torch.linalg.vector_norm(state['exp_avg_sq'].double()))
    if first_norm<=0 or second_norm<=0:
        raise ValueError(f'{label}: Adam moments are entirely zero')
    return dict(parameters=[name for name,_ in named],state_steps=step_count,
                exp_avg_l2_sum=first_norm,exp_avg_sq_l2_sum=second_norm)


def _named_subset(module,params):
    names={id(p):name for name,p in module.named_parameters()}
    return [(names[id(p)],p) for p in params]


def experiment_contract(manifest_digest, frozen_config, freeze_receipt,
                        expected_num_envs, expected_optimizer):
    requested=any(x is not None for x in (frozen_config,freeze_receipt,expected_num_envs,expected_optimizer))
    if not requested:
        return 4096,'muon',None,None
    if frozen_config is None or freeze_receipt is None:
        raise ValueError('Explicit expectations require both frozen config and freeze receipt')
    path=Path(frozen_config)
    if path.is_symlink():raise ValueError('Frozen config must not be a symlink')
    cfg=json.loads(path.read_text())
    receipt=read_receipt(freeze_receipt,manifest_digest,config=cfg)
    envs=cfg.get('num_envs');optimizer=cfg.get('optimizer');iterations=cfg.get('iterations')
    if type(envs) is not int or envs<=0 or optimizer not in ('adam','muon') or type(iterations) is not int or iterations<=0:
        raise ValueError('Invalid frozen experiment configuration')
    if expected_num_envs is not None and (type(expected_num_envs) is not int or expected_num_envs!=envs):
        raise ValueError('expected_num_envs differs from frozen configuration')
    if expected_optimizer is not None and expected_optimizer!=optimizer:
        raise ValueError('expected_optimizer differs from frozen configuration')
    return envs,optimizer,cfg,receipt


def _check_frozen_config(actual,frozen,robot):
    # These are exactly the launch fields resolved by Trainer._resolve_cfg for
    # this joint-motion/human contract; all other saved fields must match.
    expected=dict(frozen,robot=robot,subset_yaml=frozen['subset_yamls'][ROBOTS.index(robot)],morphology=True)
    expected['decoder_film']=True if frozen.get('decoder_film') is None else frozen['decoder_film']
    weight=frozen.get('latent_align_weight')
    if weight is None:
        from morphmimic.train.trainer import MULTI_ROBOT_LATENT_ALIGN_WEIGHT
        weight=MULTI_ROBOT_LATENT_ALIGN_WEIGHT
    expected['latent_align_weight']=float(weight)
    if _canonical(actual)!=_canonical(expected):
        raise ValueError(f'Checkpoint configuration differs from frozen configuration: {robot}')


def _fixed_checkpoint(path):
    path=Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve()!=path:raise ValueError(f'Invalid fixed checkpoint: {path}')
    before=sha(path);data=torch.load(path,map_location='cpu',weights_only=False)
    from morphmimic.train.checkpoint_set import verify_checkpoint_member
    verify_checkpoint_member(path,data)
    if sha(path)!=before:raise ValueError(f'Checkpoint changed while reading: {path}')
    return data,before


def _parameter_state(ckpt,models):
    policy,wm,critic=models
    if ckpt.get("cfg",{}).get("controller_mode","legacy") != "legacy":
        from morphmimic.model.controller import validate_controller_checkpoint
        validate_controller_checkpoint(wm,ckpt)
    elif ckpt.get("controller_identity") is not None or any(k.startswith("controller.") for k in ckpt.get("wm",{})):
        raise ValueError("Legacy checkpoint unexpectedly carries controller buffers or identity")
    groups={'policy':policy,'ema_policy':policy,'wm':wm,'critic':critic}
    cfg=ckpt['cfg']
    if cfg.get('target_critic_alpha') is not None:groups['target_critic']=critic
    if cfg['kp_student']=='return':
        groups['critic_kp']=critic
        if cfg.get('target_critic_alpha') is not None:groups['target_critic_kp']=critic
    result={}
    for key,model in groups.items():
        saved=ckpt.get(key)
        if not isinstance(saved,dict):raise ValueError(f'Missing shared model: {key}')
        result[key]={}
        for name,param in model.named_parameters():
            tensor=saved.get(name)
            if not isinstance(tensor,torch.Tensor) or tensor.shape!=param.shape or tensor.dtype!=param.dtype:
                raise ValueError(f'Invalid parameter: {key}.{name}')
            result[key][name]=tensor
    for key in ('obs_norm','goal_norm','human_norm','heading_norm'):
        saved=ckpt.get(key)
        if not isinstance(saved,dict) or not {'mean','var','count'}.issubset(saved):raise ValueError(f'Missing pooled norm: {key}')
        result[key]={name:saved[name] for name in ('mean','var','count')}
        if any(not isinstance(x,torch.Tensor) for x in result[key].values()):raise ValueError(f'Invalid pooled norm: {key}')
    return result


def _compare(reference,current,robot,iteration):
    if set(reference)!=set(current):raise ValueError(f'Shared model groups differ: {robot}/{iteration}')
    for group in reference:
        if set(reference[group])!=set(current[group]):raise ValueError(f'Shared parameter names differ: {group}')
        for name,a in reference[group].items():
            b=current[group][name]
            if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):
                raise ValueError(f'Shared learned parameter or pooled norm differs: {robot}/{iteration}/{group}/{name}')


def audit(run_root,manifest_path,iterations=None,robots=None,*,frozen_config=None,freeze_receipt=None,
          expected_num_envs=None,expected_optimizer=None):
    iterations,robots=validate_selection(iterations,robots)
    digest,subsets,counts=corpus_identity(manifest_path)
    expected_envs,optimizer,experiment,expected_receipt=experiment_contract(
        digest,frozen_config,freeze_receipt,expected_num_envs,expected_optimizer)
    if experiment is not None and any(i>experiment['iterations'] for i in iterations):
        raise ValueError('Requested iteration exceeds frozen experiment iterations')
    requested=sorted(set([0,*iterations]));report=dict(schema_version=1,manifest_sha256=digest,training_corpora=counts,
        requested_iterations=iterations,robots=robots,checkpoints=[],deltas={},run_configs={})
    baseline={};receipt_identity=None;receipt_seen=False
    for iteration in requested:
        reference=None
        for robot in robots:
            path=Path(run_root)/robot/f'ckpt_{iteration}.pt';ckpt,checkpoint_sha=_fixed_checkpoint(path)
            finite_tree(ckpt)
            cfg=ckpt.get('cfg',{});meta=ckpt.get('optimized_training',{})
            if (ckpt.get('iteration')!=iteration or cfg.get('robot')!=robot or cfg.get('num_envs')!=expected_envs
                or cfg.get('optimizer')!=optimizer or tuple(cfg.get('robots',()))!=ROBOTS or cfg.get('share')!='all'
                or tuple(cfg.get('subset_yamls',()))!=tuple(subsets[r] for r in ROBOTS)
                or cfg.get('subset_yaml')!=subsets[robot] or str(Path(cfg.get('prepared_manifest','')).resolve())!=str(Path(manifest_path).resolve())
                or not cfg.get('kp_interface') or cfg.get('policy_goal')!='motion' or cfg.get('kp_fixed_horizon')!=1 or cfg.get('kp_masking')!='five'
                or meta.get('schema_version')!=1 or meta.get('iteration')!=iteration or meta.get('optimizer')!=optimizer
                or meta.get('num_envs')!=expected_envs or meta.get('joint_motion_human') is not True
                or meta.get('human_horizon')!=1 or meta.get('human_points')!='five' or meta.get('prepared_manifest_sha256')!=digest):
                raise ValueError(f'Checkpoint identity differs: {path}')
            if experiment is not None:_check_frozen_config(cfg,experiment,robot)
            physics=ckpt.get('physics')
            if not isinstance(physics,dict) or type(physics.get('overflow_bits')) is not int or physics['overflow_bits']!=0:
                raise ValueError(f'Physics overflow or missing report: {path}')
            if iteration>0 and (type(physics.get('checked_physics_steps')) is not int or physics['checked_physics_steps']<=0):
                raise ValueError(f'No physics overflow samples: {path}')
            steps=meta.get('optimizer_steps');other=ckpt.get('optimization_state',{}).get('optimizer_steps')
            if (not isinstance(steps,dict) or set(steps)!={'world_model','motion','human'} or steps!=other
                or any(type(x) is not int or x<0 or (iteration>0 and x==0) for x in steps.values())
                or (iteration==0 and any(steps.values())) or ckpt.get('ema_updates')!=steps['motion']):
                raise ValueError(f'Invalid optimizer or EMA counters: {path}')
            receipt=meta.get('freeze_receipt');reduced=None
            if receipt is not None:
                _validate_receipt(receipt,digest)
                reduced={k:receipt[k] for k in ('source_tree_sha256','runtime_image_id','config_sha256','data_manifest_sha256')}
            if expected_receipt is not None and receipt!=expected_receipt:
                raise ValueError('Checkpoint freeze receipt differs from expected receipt')
            if receipt_seen and receipt_identity!=reduced:raise ValueError('Freeze receipts differ')
            receipt_identity=reduced;receipt_seen=True
            config_bytes=json.dumps(cfg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            config_sha=hashlib.sha256(config_bytes).hexdigest();report['run_configs'][config_sha]=cfg
            models=construct_models(cfg,ckpt['morph'],controller_identity=ckpt.get('controller_identity'));policy,wm,_=models
            shared=_parameter_state(ckpt,models)
            if reference is None:reference=shared
            else:_compare(reference,shared,robot,iteration)
            names={'world_model':internal_parameters(wm,[wm.dynamics_mlp]),
                   'motion_encoder':internal_parameters(policy,[policy.encoder.net]),
                   'decoder':internal_parameters(policy,[policy.decoder.net]),
                   'human_encoder':internal_parameters(policy,[policy.human_encoder.net])}
            if any(not value for value in names.values()):raise ValueError(f'No internal matrices: {path}')
            selected={key:(value[0][0],ckpt['wm' if key=='world_model' else 'policy'][value[0][0]].clone()) for key,value in names.items()}
            proofs={}
            if iteration==0:baseline[robot]=selected
            else:
                for key,(name,current) in selected.items():
                    previous_name,previous=baseline[robot][key]
                    if previous_name!=name or previous.shape!=current.shape:raise ValueError(f'Baseline architecture differs: {key}')
                    delta=float(torch.linalg.vector_norm((current.double()-previous.double())).item())
                    if not delta>0:raise ValueError(f'Zero learned parameter delta: {robot}/{iteration}/{key}')
                    report['deltas'].setdefault(str(iteration),{}).setdefault(robot,{})[key]=dict(parameter=name,l2_delta=delta)
                optim=ckpt.get('optim',{})
                if optimizer=='muon':
                    proofs['world_model']=require_muon(optim.get('wm',{}),names['world_model'],steps['world_model'],'world_model')
                    proofs['motion']=require_muon(optim.get('policy',{}),internal_parameters(policy,[policy.encoder.net,policy.decoder.net]),steps['motion'],'motion')
                    proofs['human']=require_muon(optim.get('kp',{}),names['human_encoder'],steps['human'],'human')
                else:
                    proofs['world_model']=require_adam(optim.get('wm',{}),list(wm.named_parameters()),steps['world_model'],'world_model')
                    proofs['motion']=require_adam(optim.get('policy',{}),_named_subset(policy,policy.motion_parameters()),steps['motion'],'motion')
                    proofs['human']=require_adam(optim.get('kp',{}),_named_subset(policy,policy.human_parameters()),steps['human'],'human')
                    from morphmimic.train.imagination import ImaginationCfg
                    critic_cfg=ImaginationCfg()
                    critic_updates=critic_cfg.critic_epochs*critic_cfg.critic_minibatches
                    proofs['critic']=require_adam(optim.get('critic',{}),list(models[2].named_parameters()),steps['motion']*critic_updates,'critic')
                    if cfg['kp_student']=='return':
                        proofs['critic_kp']=require_adam(optim.get('critic_kp',{}),list(models[2].named_parameters()),steps['human']*critic_updates,'critic_kp')
            report['checkpoints'].append(dict(iteration=iteration,robot=robot,path=str(path.resolve()),sha256=checkpoint_sha,
                optimizer_steps=steps,ema_updates=ckpt['ema_updates'],config_sha256=config_sha,physics=physics,**{optimizer:proofs}))
    if sha(manifest_path)!=digest:raise ValueError('Prepared manifest changed during audit')
    for record in report['checkpoints']:
        checkpoint=Path(record['path'])
        if checkpoint.is_symlink() or not checkpoint.is_file() or sha(checkpoint)!=record['sha256']:
            raise ValueError(f'Checkpoint changed during audit: {checkpoint}')
    report['freeze_receipt']=receipt_identity
    if experiment is not None:report['experiment']=experiment
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--data-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--iterations',nargs='+',type=int);p.add_argument('--robots',nargs='+')
    p.add_argument('--frozen-config',type=Path);p.add_argument('--freeze-receipt',type=Path)
    p.add_argument('--expected-num-envs',type=int);p.add_argument('--expected-optimizer',choices=('adam','muon'))
    a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():raise FileExistsError(a.out)
    report=audit(a.run_root,a.data_manifest,a.iterations,a.robots,frozen_config=a.frozen_config,
                 freeze_receipt=a.freeze_receipt,expected_num_envs=a.expected_num_envs,expected_optimizer=a.expected_optimizer)
    name=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=a.out.parent,prefix='.audit-',delete=False) as stream:
            name=stream.name;json.dump(report,stream,indent=2,allow_nan=False)
        os.link(name,a.out)
    finally:
        if name is not None:os.unlink(name)
    print('Audited',len(report['checkpoints']),'fixed robot checkpoints')

if __name__=='__main__':main()
