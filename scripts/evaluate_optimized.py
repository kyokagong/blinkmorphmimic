"""Evaluate fixed joint checkpoints by robot, dataset, and causal interface."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from scripts.prepare_optimized import ROBOTS, sha
from morphmimic.constants import R_MAX


def checkpoint_iterations():
    return range(0,5001,1000)


def validate_selection(iterations,robots):
    iterations=list(checkpoint_iterations()) if iterations is None else list(iterations)
    robots=list(ROBOTS) if robots is None else list(robots)
    if not iterations or len(set(iterations))!=len(iterations) or any(type(i) is not int or i not in range(0,10001,1000) for i in iterations):
        raise ValueError('Iterations must be unique fixed checkpoints 0..10000 in steps of 1000')
    if not robots or len(set(robots))!=len(robots) or any(r not in ROBOTS for r in robots):
        raise ValueError('Robots must be unique values from q1, bumi, g1')
    return iterations,robots


def flatten_summary(summary):
    result=dict(summary)
    drift=summary['drift']
    for key in ('mean_xy','final_xy','mean_abs_yaw_deg'):
        result['drift_'+key]=drift[key]
    return result


def cases(rows, robot):
    result={}
    for row in rows:
        if row['robot'] != robot or row['split'] != 'val': continue
        for interface in (('motion','five','full') if row['human_available'] else ('motion',)):
            result.setdefault((row['dataset'],interface),[]).append(row)
    return result


def audit_results(rows, expected):
    indexed={}
    for row in rows:
        key=tuple(row[k] for k in ('iteration','robot','dataset','interface'))
        if key in indexed:raise ValueError(f'Duplicate evaluation result: {key}')
        indexed[key]=row
    unexpected=set(indexed)-set(expected)
    if unexpected:raise ValueError(f'Unexpected evaluation cases: {sorted(unexpected)}')
    return [indexed.get(k,dict(zip(('iteration','robot','dataset','interface'),k),status='missing')) for k in expected]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','data-root','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--iterations',type=int,nargs='+');p.add_argument('--robots',nargs='+')
    a=p.parse_args()
    try:iterations,robots=validate_selection(a.iterations,a.robots)
    except ValueError as error:p.error(str(error))
    if a.num_envs<1:p.error('--num-envs must be positive')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True); manifest_path=a.data_root/'manifest.json'; data_hash=sha(manifest_path)
    manifest=json.loads(manifest_path.read_text());rows=manifest['clips']
    if not manifest.get('complete'):raise ValueError('Incomplete data manifest')
    from morphmimic.scripts.gpu_guard import select_gpu
    select_gpu(a.gpu,True)
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    from morphmimic.eval.benchmark import evaluate,summarize
    from morphmimic.scripts.eval import checkpoint_identity,verify_checkpoint
    import pandas as pd
    import torch
    results=[]
    for iteration in iterations:
        for robot in robots:
            checkpoint=a.run_root/robot/f'ckpt_{iteration}.pt'
            for (dataset,interface),group in sorted(cases(rows,robot).items()):
                result=dict(iteration=iteration,robot=robot,dataset=dataset,interface=interface,
                            split='val',status='error',clips=len(group),actual_num_envs=min(a.num_envs,len(group)),manifest_sha256=data_hash,
                            metric_definition='legacy_tracking_reward_v1',normalization_r_max=R_MAX,control_graph_requested=True)
                stem=f'{robot}_{dataset}_{interface}_{iteration}'
                try:
                    identity=checkpoint_identity(checkpoint)
                    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    meta=ckpt.get('optimized_training',{})
                    if ckpt.get('iteration')!=iteration or ckpt.get('cfg',{}).get('robot')!=robot or meta.get('prepared_manifest_sha256')!=data_hash:
                        raise ValueError('Checkpoint iteration, robot, or dataset identity differs')
                    if sha(manifest_path)!=data_hash:raise ValueError('Dataset manifest changed')
                    for row in group:
                        if sha(row['output'])!=row['output_sha256']:raise ValueError('Motion hash differs: '+row['output'])
                        if interface!='motion' and sha(row['human_sidecar'])!=row['human_sha256']:
                            raise ValueError('Human hash differs: '+row['human_sidecar'])
                    subset=a.out/(stem+'.yaml');subset.write_text(json.dumps(dict(root='/',clips=[r['output'] for r in group])))
                    frame=evaluate(str(checkpoint),str(subset),'cuda:0',num_envs=result['actual_num_envs'],
                        interface='motion' if interface=='motion' else 'human',
                        human_points='five' if interface=='motion' else interface,kp_horizon=1,
                        human_root=manifest['human_root'],robot=robot,control_graph=True)
                    if len(frame)!=len(group) or set(frame['name'])!={r['name'] for r in group}:
                        raise ValueError('Evaluation clip membership differs')
                    physics=frame.attrs.get('physics',{})
                    if physics.get('overflow_bits',0):raise ValueError(f'Evaluation physics overflow: {physics}')
                    verify_checkpoint(identity)
                    if sha(manifest_path)!=data_hash:raise ValueError('Dataset manifest changed')
                    summary=flatten_summary(summarize(frame,pd.DataFrame([dict(name=r['name'],category=dataset) for r in group])))
                    csv=a.out/(stem+'.csv');frame.to_csv(csv,index=False)
                    result.update(summary,**identity,status='ok',subset_sha256=sha(subset),per_clip_sha256=sha(csv),physics=physics,control_graph=True)
                except Exception as e:
                    result['error']=f'{type(e).__name__}: {e}'
                results.append(result)
                tmp=a.out/'results.json.tmp';tmp.write_text(json.dumps(results,indent=2));tmp.replace(a.out/'results.json')
                print(stem,result['status'],result.get('error',''),flush=True)
    if any(r['status']!='ok' for r in results):raise SystemExit('Evaluation has missing or failed cases; see results.json')

if __name__=='__main__':main()
