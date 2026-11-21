"""Evaluate fixed human updates 0..10000 and the frozen teacher on paired validation."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from morphmimic.eval.benchmark import evaluate,summarize
from morphmimic.scripts.eval import checkpoint_identity,verify_checkpoint
from morphmimic.train.human_adapt import file_sha,verify_adaptation_identity,PARENT_SHA256

ROBOTS=('q1','bumi','g1')
EVALUATION_PROTOCOL='dataset-groups-v1'


def choose_candidate(rows):
    groups={}
    for row in rows:
        if row['human_updates']>0 and row['split']=='val' and row['points']=='five':
            group=groups.setdefault(row['human_updates'],{})
            if row['robot'] in group:raise ValueError('Duplicate checkpoint/robot validation')
            group[row['robot']]=row
    choices=[]
    for update,robots in groups.items():
        if set(robots)!=set(ROBOTS):continue
        score=dict(human_updates=update,failures=sum(r['failures'] for r in robots.values()),
                   mean_norm_return=float(np.mean([r['mean_norm_return'] for r in robots.values()])))
        if np.isfinite(score['mean_norm_return']):choices.append(score)
    return min(choices,key=lambda r:(r['failures'],-r['mean_norm_return'])) if choices else None


def plot_progress(rows,out,teacher=()):
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for col,points in enumerate(('five','full')):
        for robot,color in zip(ROBOTS,('tab:blue','tab:orange','tab:green')):
            selected=sorted((r for r in rows if r['robot']==robot and r['points']==points and r['split']=='val'),key=lambda r:r['human_updates'])
            if not selected:continue
            x=[r['human_updates'] for r in selected]
            axes[0,col].plot(x,[r['mean_norm_return'] for r in selected],'-o',label=robot,color=color)
            axes[1,col].plot(x,[r['failures'] for r in selected],'-o',label=robot,color=color)
            baseline=next((r for r in teacher if r['robot']==robot),None)
            if baseline is not None:
                axes[0,col].axhline(baseline['mean_norm_return'],linestyle='--',color=color,alpha=.6,label=robot+' motion teacher')
        axes[0,col].set_title(points+' / current frame (h=1)');axes[0,col].set_ylabel('Validation normalized return')
        axes[1,col].set_ylabel('Validation failures');axes[1,col].set_xlabel('Human updates after fixed motion teacher 30000')
        for ax in axes[:,col]:ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Human adaptation: same paired validation clips at every fixed checkpoint')
    fig.tight_layout();fig.savefig(out/'human-training.png',dpi=160);fig.savefig(out/'human-training.svg');plt.close(fig)


def evaluate_one(path,subset,data_hash,out,*,robot,split,points,update,num_envs,categories,human_root):
    from morphmimic.data.human_prepare import verify_prepared_subset
    verify_prepared_subset(Path(human_root).parent,subset,robot=robot,split=split,expected_sha256=data_hash)
    identity=checkpoint_identity(path);stem=f'{robot}_{split}_{points}_{update}'
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if points=='motion':
        if identity['checkpoint_sha256']!=PARENT_SHA256[robot] or ckpt['cfg']['robot']!=robot or ckpt['iteration']!=30000:
            raise ValueError('Motion baseline identity differs from the preserved teacher')
    else:verify_adaptation_identity(ckpt,robot=robot,update=update,data_sha256=data_hash)
    del ckpt
    csv_path=out/(stem+'.csv');json_path=out/(stem+'.json');subset_hash=file_sha(subset)
    if json_path.exists():
        saved=json.loads(json_path.read_text())
        if saved.get('evaluation_protocol')!=EVALUATION_PROTOCOL or saved['checkpoint_sha256']!=identity['checkpoint_sha256'] or saved['paired_manifest_sha256']!=data_hash or saved['subset_sha256']!=subset_hash or file_sha(csv_path)!=saved['per_clip_sha256']:
            raise ValueError('Evaluation resume identity differs')
        return saved
    from morphmimic.data.manifest import load_subset
    by_name=categories.drop_duplicates('name').set_index('name')['category'].to_dict()
    groups={}
    for clip in load_subset(str(subset)):
        groups.setdefault(by_name[Path(clip).stem],[]).append(Path(clip).name)
    frames=[];physics={};availability={};evaluated_subsets=[]
    folder=out/'subsets';folder.mkdir(exist_ok=True)
    for family,names in sorted(groups.items()):
        if family not in ('seed','lafan1'):raise ValueError('Unknown paired human dataset')
        part=folder/f'{robot}_{split}_{family}.yaml'
        part.write_text(yaml.safe_dump(dict(root=str(Path(human_root).parent/robot),clips=names)))
        frame=evaluate(str(path),str(part),'cuda:0',num_envs=num_envs,library_dtype='float32',
            interface='motion' if points=='motion' else 'human',human_points='five' if points=='motion' else points,
            kp_horizon=1,human_root=human_root,verbose=True)
        frames.append(frame);physics[family]=frame.attrs['physics']
        for count,clips in (frame.attrs.get('available_human_joints') or {}).items():
            availability[count]=availability.get(count,0)+clips
        evaluated_subsets.append(dict(dataset=family,clips=len(names),subset_yaml=str(part),subset_sha256=file_sha(part)))
    df=pd.concat(frames,ignore_index=True);bits=0
    for report in physics.values():bits|=int(report.get('overflow_bits',0))
    df.attrs.update(mode=frames[0].attrs['mode'],physics=dict(overflow_bits=bits,by_dataset=physics),available_human_joints=availability or None)
    verify_checkpoint(identity)
    verify_prepared_subset(Path(human_root).parent,subset,robot=robot,split=split,expected_sha256=data_hash)
    summary=summarize(df,categories)
    summary.update(**identity,robot=robot,split=split,points=points,human_updates=update,parent_iteration=30000,
        paired_manifest_sha256=data_hash,subset_yaml=str(subset),subset_sha256=subset_hash,
        iteration=update,mode=df.attrs['mode'],available_human_joints=df.attrs.get('available_human_joints'),physics=df.attrs['physics'])
    if points=='motion':summary['human_updates']=0
    summary.update(evaluation_protocol=EVALUATION_PROTOCOL,evaluation_subsets=evaluated_subsets)
    temporary=csv_path.with_suffix('.csv.tmp');df.to_csv(temporary,index=False);temporary.replace(csv_path)
    summary['per_clip_sha256']=file_sha(csv_path)
    temporary=json_path.with_suffix('.json.tmp');temporary.write_text(json.dumps(summary,indent=2));temporary.replace(json_path)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','data-root','teacher-root','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--follow',action='store_true');p.add_argument('--final-update',type=int,default=10000)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    from morphmimic.scripts.gpu_guard import select_gpu
    from mjlab.utils.torch import configure_torch_backends
    select_gpu(args.gpu,True);configure_torch_backends()
    manifest_path=args.data_root/'manifest.json';data_hash=file_sha(manifest_path);manifest=json.loads(manifest_path.read_text())
    if manifest.get('complete') is not True:raise ValueError('Formal evaluation requires the complete paired source manifest')
    categories=pd.DataFrame([dict(name=r['name'],category=r['dataset']) for r in manifest['clips'] if r['robot']=='g1'])
    probes=args.out/'subsets';probes.mkdir(exist_ok=True);subsets={}
    for robot in ROBOTS:
        selected=[]
        for dataset in ('seed','lafan1'):
            selected.extend([r['name']+'.npz' for r in manifest['clips'] if r['robot']==robot and r['split']=='train' and r['dataset']==dataset][:16])
        probe=probes/(robot+'_train_probe.yaml');probe.write_text(yaml.safe_dump(dict(root=str(args.data_root/robot),clips=selected)))
        subsets[robot]={'train_probe':probe,'val':args.data_root/robot/'val.yaml'}
    rows=[];teacher=[]
    human_root=str(args.data_root/'human50')
    for robot in ROBOTS:
        teacher.append(evaluate_one(args.teacher_root/robot/'ckpt_30000.pt',subsets[robot]['val'],data_hash,args.out,
            robot=robot,split='val',points='motion',update=30000,num_envs=args.num_envs,categories=categories,human_root=human_root))
        torch.cuda.empty_cache()
    (args.out/'teacher-baseline.json').write_text(json.dumps(teacher,indent=2))
    for update in range(0,args.final_update+1,1000):
        paths={r:args.run_root/r/f'ckpt_{update}.pt' for r in ROBOTS}
        while not all(path.is_file() for path in paths.values()):
            if not args.follow:
                print('Not yet complete:',update,flush=True);break
            if file_sha(manifest_path)!=data_hash:raise ValueError('Paired manifest changed')
            time.sleep(10)
        else:
            for robot in ROBOTS:
                for split,subset in subsets[robot].items():
                    for points in ('five','full'):
                        row=evaluate_one(paths[robot],subset,data_hash,args.out,robot=robot,split=split,points=points,
                            update=update,num_envs=args.num_envs,categories=categories,human_root=human_root)
                        rows.append(row);torch.cuda.empty_cache()
            pd.DataFrame([{k:r[k] for k in ('human_updates','robot','split','points','failures','mean_norm_return','checkpoint_sha256')} for r in rows]).to_csv(args.out/'progress.csv',index=False)
            plot_progress(rows,args.out,teacher)
            candidate=choose_candidate(rows)
            (args.out/'candidate.json').write_text(json.dumps(candidate,indent=2))
            print('Validated human update',update,'candidate',candidate,flush=True)
            continue
        break
    if file_sha(manifest_path)!=data_hash:raise ValueError('Paired manifest changed')


if __name__=='__main__':main()
