"""Route independent benchmark environments to fixed human checkpoints."""
from pathlib import Path
import torch
from deploy.export_onnx import _build_policy
from morphmimic.constants import GOAL_VERSION,KP_GOAL_VERSION
from morphmimic.model.human import fixed_human_masks,human_features_from_goal,heading_from_goal,mask_human_goal
from morphmimic.scripts.eval import checkpoint_identity,verify_checkpoint
from morphmimic.train.human_adapt import verify_adaptation_identity
from morphmimic.train.normalizers import RunningNormalizer,normalize_history


class BatchHumanPolicy:
    def __init__(self,paths,robot,data_sha256,device,*,points='five'):
        if not paths:raise ValueError('At least one fixed checkpoint is required')
        if points not in ('five','full'):raise ValueError('Use five or full available points')
        self.points=points;self.encoders=[];self.human_norms=[];self.heading_norms=[];self.identities=[]
        first=None
        for path in paths:
            identity=checkpoint_identity(path)
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            update=int(Path(path).stem.removeprefix('ckpt_'))
            verify_adaptation_identity(ckpt,robot=robot,update=update,data_sha256=data_sha256)
            if ckpt.get('goal_version')!=GOAL_VERSION or ckpt.get('kp_goal_version')!=KP_GOAL_VERSION or ckpt['cfg'].get('goal_anchor','reference')!='reference':
                raise ValueError('Checkpoint goal interface differs')
            state=ckpt.get('ema_policy',ckpt['policy'])
            if first is None:
                first=ckpt;self.hist_len=int(ckpt['cfg']['hist_len'])
            else:
                original=first.get('ema_policy',first['policy'])
                if state.keys()!=original.keys() or any(not torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('human_encoder.')):
                    raise ValueError('Checkpoints differ in frozen policy layers')
                for group in ('obs_norm','goal_norm'):
                    if any(not torch.equal(v,ckpt[group][k]) for k,v in first[group].items()):
                        raise ValueError('Checkpoints differ in frozen normalization')
            policy=_build_policy(ckpt)
            if self.encoders and policy.cfg!=self.policy_cfg:raise ValueError('Checkpoint policy configuration differs')
            self.policy_cfg=policy.cfg
            if policy.human_encoder is None:raise ValueError('Checkpoint has no human encoder')
            self.encoders.append(policy.human_encoder.to(device).eval())
            if len(self.encoders)==1:
                self.decoder=policy.decoder.to(device).eval()
                self.obs_norm=RunningNormalizer(93);self.obs_norm.load_state_dict(ckpt['obs_norm']);self.obs_norm.to(device).eval()
            for group,width,dest in (('human_norm',2160,self.human_norms),('heading_norm',60,self.heading_norms)):
                norm=RunningNormalizer(width);norm.load_state_dict(ckpt[group]);dest.append(norm.to(device).eval())
            verify_checkpoint(identity);self.identities.append(dict(identity,human_updates=update))

    @torch.no_grad()
    def __call__(self,history,goals,checkpoint_ids):
        batch=len(history)
        if checkpoint_ids.shape!=(batch,) or checkpoint_ids.dtype!=torch.long or not bool(((checkpoint_ids>=0)&(checkpoint_ids<len(self.encoders))).all()):
            raise ValueError('Invalid per-environment checkpoint route')
        points,frames,heading_mask=fixed_human_masks(batch,self.points,1,history.device)
        points &= goals['human_joint_valid']
        raw=goals['goal_kp'];features=human_features_from_goal(raw,points);heading=heading_from_goal(raw,points)
        latents=[]
        for encoder,human_norm,heading_norm in zip(self.encoders,self.human_norms,self.heading_norms):
            goal=mask_human_goal(human_norm(features),points,frames,heading_norm(heading),heading_mask)
            latents.append(encoder(goal))
        latent=torch.stack(latents)[checkpoint_ids,torch.arange(batch,device=history.device)]
        action=self.decoder(latent,normalize_history(self.obs_norm,history))
        if not bool(torch.isfinite(action).all()):raise ValueError('Nonfinite batched checkpoint action')
        return action


def split_checkpoint_results(frame,names,lengths,count):
    if count<1 or len(frame)!=len(names)*count:
        raise ValueError('Checkpoint case count differs')
    expected_names=[name for name in names for _ in range(count)]
    expected_lengths=[int(length) for length in lengths for _ in range(count)]
    if frame['name'].tolist()!=expected_names or frame['frames'].tolist()!=expected_lengths:
        raise ValueError('Checkpoint case order or frame count differs')
    return [frame.iloc[index::count].reset_index(drop=True) for index in range(count)]
