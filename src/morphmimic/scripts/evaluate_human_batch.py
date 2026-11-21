"""Complete official human benchmark with independent environments per checkpoint."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
from morphmimic.data.human_prepare import verify_prepared_subset
from morphmimic.data.manifest import load_subset
from morphmimic.env.env_cfg import COMMAND_NAME
from morphmimic.eval.benchmark import make_eval_env,run_benchmark,require_human_sidecars,summarize
from morphmimic.eval.checkpoint_batch import BatchHumanPolicy,split_checkpoint_results
from morphmimic.scripts.eval import verify_checkpoint
from morphmimic.train.human_adapt import file_sha

PROTOCOL='checkpoint-batch-v2'


def configure_full_precision():
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    torch.backends.cudnn.conv.fp32_precision='ieee'
    torch.backends.cudnn.rnn.fp32_precision='ieee'
    if set(precision_report().values())!={'ieee'}:raise RuntimeError('Evaluation requires IEEE float32 operators')


def precision_report():
    return dict(cuda_matmul=torch.backends.cuda.matmul.fp32_precision,
                cudnn_conv=torch.backends.cudnn.conv.fp32_precision,
                cudnn_rnn=torch.backends.cudnn.rnn.fp32_precision)


def save_group_result(path,frame,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);report=path.with_suffix('.json')
    if path.exists() or report.exists():raise FileExistsError('Fixed raw evaluation exists: '+str(path))
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('x') as stream:frame.to_csv(stream,index=False)
    temporary.replace(path)
    metadata=dict(metadata,per_clip_sha256=file_sha(path))
    temporary=report.with_suffix('.json.tmp')
    with temporary.open('x') as stream:json.dump(metadata,stream,indent=2)
    temporary.replace(report)


def write_case_yaml(path,clips):
    clips=[Path(p).resolve() for p in clips]
    if not clips or len({p.parent for p in clips})!=1:raise ValueError('Case clips need one absolute robot root')
    content=yaml.safe_dump(dict(root=str(clips[0].parent),clips=[p.name for p in clips]))
    if path.exists():
        if path.read_text()!=content:raise FileExistsError('Refusing to change fixed case YAML '+str(path))
    else:
        with path.open('x') as stream:stream.write(content)


def evaluate_versions(paths,subset,data_root,out,*,robot,split,num_envs=1024):
    data_root=Path(data_root).resolve();out=Path(out).resolve();subset=Path(subset).resolve()
    if any((out/'raw').glob(f'{robot}_{split}_*')):raise FileExistsError('Partial raw evaluation exists; recover it or use a new output directory')
    for path in paths:
        update=int(Path(path).stem.removeprefix('ckpt_'))
        for points in ('five','full'):
            for suffix in ('.csv','.json'):
                target=out/(f'{robot}_{split}_{points}_{update}'+suffix)
                if target.exists():raise FileExistsError('Refusing to overwrite fixed evaluation '+str(target))
    out.mkdir(parents=True,exist_ok=True)
    digest=file_sha(data_root/'manifest.json');manifest=json.loads((data_root/'manifest.json').read_text())
    if manifest.get('complete') is not True:raise ValueError('Incomplete human manifest')
    verify_prepared_subset(data_root,subset,robot=robot,split=split,expected_sha256=digest)
    subset_hash=file_sha(subset);original=[Path(p) for p in load_subset(str(subset))]
    records={r['name']:r for r in manifest['clips'] if r['robot']==robot}
    categories=pd.DataFrame([dict(name=n,category=r['dataset']) for n,r in records.items()])
    groups={}
    for path in original:groups.setdefault(records[path.stem]['dataset'],[]).append(path)
    result={points:[[] for _ in paths] for points in ('five','full')};physics={};availability={};parts=[]
    policy=BatchHumanPolicy(paths,robot,digest,'cuda:0');count=len(paths)
    updates=[r['human_updates'] for r in policy.identities]
    if len(set(updates))!=count:raise ValueError('Duplicate checkpoint version')
    folder=out/'subsets';folder.mkdir(exist_ok=True)
    for family,clips in sorted(groups.items()):
        if family not in ('seed','lafan1'):raise ValueError('Unknown human dataset')
        part=folder/f'{robot}_{split}_{family}_batch.yaml'
        write_case_yaml(part,[p for p in clips for _ in paths])
        env=make_eval_env(str(part),min(num_envs,len(clips)*count),'cuda:0','float32',robot=robot,human_root=str(data_root/'human50'))
        try:
            require_human_sidecars(env,str(part),str(data_root/'human50'))
            command=env.command_manager.get_term(COMMAND_NAME);lib=command.library
            names=[p.stem for p in clips];lengths=[records[n]['frames'] for n in names]
            for points in result:
                policy.points=points
                def act(history,goals):
                    return policy(history,goals,lib.frame_clip[command.global_frame]%count)
                frame=run_benchmark(env,act,policy.hist_len,verbose=True)
                separated=split_checkpoint_results(frame,names,lengths,count)
                for index,row in enumerate(separated):result[points][index].append(row)
                report=env.physics_diagnostics.report()
                if report['overflow_bits']:raise RuntimeError(f'Invalid evaluation physics: {report}')
                physics[family+'_'+points]=report
                save_group_result(out/'raw'/f'{robot}_{split}_{family}_{points}.csv',frame,
                    dict(robot=robot,split=split,dataset=family,points=points,checkpoints=policy.identities,
                         paired_manifest_sha256=digest,subset_sha256=subset_hash,case_yaml=str(part),
                         case_yaml_sha256=file_sha(part),actual_num_envs=env.num_envs,seed=0,
                         physics=report,precision=precision_report(),evaluation_protocol=PROTOCOL))
            counts,totals=lib.human_joint_valid[::count].sum(-1).unique(return_counts=True)
            for n,total in zip(counts.tolist(),totals.tolist()):availability[str(n)]=availability.get(str(n),0)+int(total)
            parts.append(dict(dataset=family,actual_num_envs=env.num_envs,seed=0,source_clips=len(clips),checkpoint_cases=len(clips)*count,subset_yaml=str(part),subset_sha256=file_sha(part)))
        finally:env.close()
        torch.cuda.empty_cache()
    verify_prepared_subset(data_root,subset,robot=robot,split=split,expected_sha256=digest)
    for identity in policy.identities:verify_checkpoint(identity)
    rows=[]
    expected={p.stem for p in original}
    for points,versions in result.items():
        for identity,frames in zip(policy.identities,versions):
            df=pd.concat(frames,ignore_index=True)
            if len(df)!=len(original) or set(df.name)!=expected or df.name.duplicated().any():raise ValueError('Incomplete source cases')
            row=summarize(df,categories);update=identity['human_updates']
            row.update(**identity,robot=robot,split=split,points=points,parent_iteration=30000,iteration=update,
                paired_manifest_sha256=digest,subset_yaml=str(subset),subset_sha256=subset_hash,
                mode='human-'+points+'-h1',available_human_joints=availability,
                physics=dict(overflow_bits=0,by_dataset_and_points=physics),evaluation_protocol=PROTOCOL,matmul_precision="float32-highest",tf32_matmul=precision_report()['cuda_matmul']=='tf32',tf32_cudnn=any(precision_report()[key]=='tf32' for key in ('cudnn_conv','cudnn_rnn')),precision=precision_report(),route="frame_clip[command.global_frame] % checkpoint_count",
                evaluation_subsets=parts,checkpoint_batch=policy.identities)
            stem=f'{robot}_{split}_{points}_{update}';csv_path=out/(stem+'.csv');json_path=out/(stem+'.json')
            if csv_path.exists() or json_path.exists():raise FileExistsError('Refusing to overwrite fixed evaluation '+stem)
            tmp=csv_path.with_suffix('.csv.tmp');df.to_csv(tmp,index=False);tmp.replace(csv_path)
            row['per_clip_sha256']=file_sha(csv_path);tmp=json_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(json_path)
            rows.append(row)
    print('Completed checkpoint batch',robot,split,len(rows),'summaries',flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','data-root','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--robot',choices=('q1','bumi','g1'),required=True);parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--num-envs',type=int,default=1024)
    args=parser.parse_args()
    args.data_root=args.data_root.resolve();args.out=args.out.resolve();args.run_root=args.run_root.resolve()
    from morphmimic.scripts.gpu_guard import select_gpu
    select_gpu(args.gpu,True);configure_full_precision()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data_root/'manifest.json').read_text());selected=[]
    for family in ('seed','lafan1'):
        selected.extend([r['name']+'.npz' for r in manifest['clips'] if r['robot']==args.robot and r['split']=='train' and r['dataset']==family][:16])
    probe=args.out/(args.robot+'_train_probe.yaml');write_case_yaml(probe,[args.data_root/args.robot/name for name in selected])
    paths=[args.run_root/args.robot/f'ckpt_{update}.pt' for update in range(0,10001,1000)]
    for split,subset in (('val',args.data_root/args.robot/'val.yaml'),('train_probe',probe)):
        evaluate_versions(paths,subset,args.data_root,args.out,robot=args.robot,split=split,num_envs=args.num_envs)


if __name__=='__main__':main()
