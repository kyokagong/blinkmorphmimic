"""Persistent, budgeted train-only probes on one dedicated GPU."""
from __future__ import annotations
import argparse
import fcntl
import math
from datetime import datetime,timezone
import json
import os
import random
from pathlib import Path
import signal
import time

from morphmimic.train.probe_worker import INTERFACES,aggregate_failures,budget_remaining,latest_completed,validate_training_rows,needs_probe
from morphmimic.scripts.probe import file_hash,checkpoint_identity,validate_probe_clips
from morphmimic.train.probe_focus import select_probe_clips

ROBOTS=('q1','bumi','g1')
WEIGHTS={'motion':.5,'human-five':.25,'human-full':.25}


def save_atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)


def stratified_pool(rows,pool_size,pool_frames,seed):
    if type(pool_size) is not int or not 1<=pool_size<=256 or type(pool_frames) is not int or pool_frames<1:
        raise ValueError('Invalid probe pool size or frame budget')
    if any(type(r.get('frames')) is not int or r['frames']<2 for r in rows):
        raise ValueError('Invalid probe clip length')
    if pool_size<3:return select_probe_clips(rows,pool_size,pool_frames,seed)
    rng=random.Random(seed);families=('seed','lafan1','motiondecode')
    groups={family:[r for r in rows if r.get('dataset')==family] for family in families}
    if any(not groups[family] for family in families):raise ValueError('Probe pool requires all three training corpora')
    for group in groups.values():rng.shuffle(group)
    minimum={family:min(r['frames']-1 for r in groups[family]) for family in families}
    if sum(minimum.values())>pool_frames:raise ValueError('Probe pool frame budget cannot fit all corpora')
    selected=[];remaining=pool_frames
    for index,family in enumerate(families):
        reserve=sum(minimum[x] for x in families[index+1:])
        item=next(r for r in groups[family] if r['frames']-1<=remaining-reserve)
        selected.append(item);remaining-=item['frames']-1
    selected_ids={id(r) for r in selected};rest=[r for r in rows if id(r) not in selected_ids]
    rng.shuffle(rest)
    for row in rest:
        if len(selected)>=pool_size:break
        if row['frames']-1<=remaining:selected.append(row);remaining-=row['frames']-1
    return selected



def select_checkpoint_clips(clips,interface,max_clips,max_frames,iteration,window_transitions=None):
    """Cover available datasets when feasible, rotating priority under tight bounds."""
    if interface not in INTERFACES or type(iteration) is not int or iteration<0:
        raise ValueError('Invalid probe interface or checkpoint iteration')
    if type(max_clips) is not int or not 1<=max_clips<=256 or type(max_frames) is not int or max_frames<1:
        raise ValueError('Invalid probe clip or frame budget')
    families=('seed','lafan1','motiondecode') if interface=='motion' else ('seed','lafan1')
    eligible=[r for r in clips if r.get('dataset') in families and (interface=='motion' or r.get('human_available') is True)]
    if any(type(r.get('frames')) is not int or r['frames']<2 or not isinstance(r.get('name'),str) for r in eligible):
        raise ValueError('Invalid probe clip identity or length')
    if len({r['name'] for r in eligible})!=len(eligible):raise ValueError('Duplicate probe clip identity')
    if window_transitions is not None and window_transitions not in (256,512):
        raise ValueError('Window must have 256 or 512 transitions')
    short=[r for r in eligible if window_transitions is not None and r['frames']<=window_transitions]
    eligible=[r for r in eligible if r not in short]
    cost=lambda r: r['frames']-1 if window_transitions is None else window_transitions
    groups={family:[r for r in eligible if r['dataset']==family] for family in families}
    for family,group in groups.items():random.Random(f'{interface}:{iteration}:{family}').shuffle(group)
    available=[family for family in families if groups[family]]
    if not available:raise ValueError(f'No eligible clips for {interface}')
    shift=(iteration//1000)%len(available);order=available[shift:]+available[:shift]
    minimum={family:min(cost(r) for r in groups[family]) for family in order}
    all_feasible=len(order)<=max_clips and sum(minimum.values())<=max_frames
    selected=[];remaining=max_frames
    for index,family in enumerate(order):
        if len(selected)>=max_clips:break
        reserve=sum(minimum[x] for x in order[index+1:]) if all_feasible else 0
        item=next((r for r in groups[family] if cost(r)<=remaining-reserve),None)
        if item is not None:selected.append(item);remaining-=cost(item)
    chosen={r['name'] for r in selected};rest=[r for r in eligible if r['name'] not in chosen]
    random.Random(f'{interface}:{iteration}:extra').shuffle(rest)
    for row in rest:
        if len(selected)>=max_clips:break
        if cost(row)<=remaining:selected.append(row);remaining-=cost(row)
    if not selected:raise ValueError(f'No whole {interface} clip fits the frame budget')
    detail=dict(required_datasets=list(families),available_datasets=available,
                unavailable_datasets=[f for f in families if not groups[f]],
                over_budget_clips=[dict(name=r['name'],dataset=r['dataset'],frames=r['frames'])
                                   for r in eligible if cost(r)>max_frames])
    detail['ineligible_short_clips']=[dict(name=r['name'],dataset=r['dataset'],frames=r['frames']) for r in short]
    return selected,detail


def choose_window(clip,interface,iteration,transitions):
    if transitions not in (256,512) or clip['frames']<=transitions:
        raise ValueError('Clip cannot fit requested window')
    start=random.Random(f"window:{interface}:{iteration}:{clip['name']}").randrange(clip['frames']-transitions)
    return start,start+transitions



def prepare_pool(data,robot,out,pool_size,pool_frames,seed):
    from morphmimic.data.manifest import load_subset
    manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if not manifest.get('complete'):raise ValueError('Incomplete composite manifest')
    subset=data/robot/'train.yaml';rows=validate_training_rows(manifest['clips'],robot,load_subset(str(subset)))
    clips=stratified_pool(rows,pool_size,pool_frames,seed)
    for row in clips:
        if file_hash(row['output'])!=row['output_sha256']:raise ValueError('Motion SHA mismatch: '+row['output'])
        if row['human_available'] and file_hash(row['human_sidecar'])!=row['human_sha256']:
            raise ValueError('Human SHA mismatch: '+row['human_sidecar'])
    validate_probe_clips([dict(r,path=r['output']) for r in clips])
    pool=out/(robot+'_pool.yaml');pool.write_text(json.dumps(dict(root='/',clips=[r['output'] for r in clips])))
    return clips,pool,dict(training_subset=str(subset.resolve()),training_subset_sha256=file_hash(subset),
        prepared_manifest_sha256=file_hash(manifest_path),pool_path=str(pool.resolve()),pool_sha256=file_hash(pool),human_root=manifest['human_root'])


def check_proof(data,checkpoint,proof,clips):
    if file_hash(data/'manifest.json')!=proof['prepared_manifest_sha256'] or file_hash(proof['training_subset'])!=proof['training_subset_sha256']:
        raise ValueError('Training manifest or subset changed')
    if file_hash(proof['pool_path'])!=proof['pool_sha256']:raise ValueError('Probe pool changed')
    if file_hash(checkpoint['checkpoint'])!=checkpoint['checkpoint_sha256']:raise ValueError('Checkpoint changed during probe')
    for row in clips:
        if file_hash(row['output'])!=row['output_sha256']:raise ValueError('Motion changed during probe')
        if row['human_available'] and file_hash(row['human_sidecar'])!=row['human_sha256']:
            raise ValueError('Human sidecar changed during probe')


def probe_one(env,clips,checkpoint,proof,data,robot,iteration,max_clips,max_frames,deadline,window_transitions=None):
    import torch
    from morphmimic.eval.benchmark import run_benchmark
    from morphmimic.eval.policy_loader import load_policy
    ckpt=torch.load(checkpoint['checkpoint'],map_location='cpu',weights_only=False)
    if ckpt.get('iteration')!=iteration or ckpt.get('ema_policy') is None:
        raise ValueError('Probe requires the matching fixed checkpoint with EMA weights')
    if ckpt.get('optimized_training',{}).get('prepared_manifest_sha256')!=proof['prepared_manifest_sha256']:
        raise ValueError('Checkpoint data manifest differs')
    rho=ckpt['cfg'].get('probe_reset_share',.15)
    if not math.isfinite(rho) or not 0<=rho<1:raise ValueError('Probe focus must retain base exploration')
    if checkpoint['goal_anchor']!='reference':raise ValueError('Probe pool requires reference-anchored checkpoints')
    del ckpt
    records={i:[] for i in INTERFACES};coverage={};selected_rows=[]
    for interface in INTERFACES:
        selected,selection=select_checkpoint_clips(clips,interface,max_clips,max_frames,iteration,window_transitions)
        selected_rows.extend(selected)
        ids=[clips.index(row) for row in selected]
        policy,meta=load_policy(checkpoint['checkpoint'],'cuda:0',interface='motion' if interface=='motion' else 'human',
            human_points='five' if interface in ('motion','human-five') else 'full',kp_horizon=1,goal_anchor=checkpoint['goal_anchor'])
        windows=[choose_window(r,interface,iteration,window_transitions) for r in selected] if window_transitions is not None else None
        frame=run_benchmark(env,policy,meta.hist_len,clip_ids=ids,deadline=deadline,windows=windows)
        physics=env.physics_diagnostics.report()
        if physics.get('overflow_bits',0):raise RuntimeError(f'Probe physics overflow: {physics}')
        if frame['name'].tolist()!=[r['name'] for r in selected]:raise ValueError('Probe result clip membership differs')
        for index,(row,clip) in enumerate(zip(frame.to_dict('records'),selected)):
            if row['failed']:
                fail=int(row['fail_frame'])
                if not 0<=fail<clip['frames']:raise ValueError('Invalid probe failure frame')
                failure=dict(robot=robot,source_id=clip['source_id'],dataset=clip['dataset'],clip=clip['name'],fail_frame=fail)
                if windows is not None:
                    start,end=windows[index]
                    if not start < fail <= end:raise ValueError('Failure frame outside probe window')
                    failure.update(metric_scope='probe_window',window_start_frame_50hz=start,window_end_frame_50hz=end,output_sha256=clip['output_sha256'])
                records[interface].append(failure)
        window_records=[]
        if windows is not None:
            for row,clip,(start,end) in zip(frame.to_dict('records'),selected,windows):
                if not math.isfinite(float(row['window_norm_return'])):raise ValueError('Nonfinite window return')
                window_records.append(dict(clip=clip['name'],dataset=clip['dataset'],source_id=clip['source_id'],
                    metric_scope='probe_window',window_start_frame_50hz=start,window_end_frame_50hz=end,
                    window_transitions=end-start,actual_steps=int(row['steps']),window_completed=bool(row['window_completed']),
                    failed=bool(row['failed']),fail_frame=int(row['fail_frame']) if row['failed'] else None,
                    window_norm_return=float(row['window_norm_return']),output_sha256=clip['output_sha256'],
                    human_sha256=clip.get('human_sha256') if clip.get('human_available') else None,
                    robot_source_sha256=clip.get('original_sha256'),human_source_sha256=(clip.get('pair_source') or {}).get('sha256')))
        coverage[interface]=dict(windows=window_records,clips=[r['name'] for r in selected],datasets=[r['dataset'] for r in selected],
            selection=selection,active_steps=int(frame['steps'].sum()),failures=len(records[interface]),physics=physics)
    check_proof(data,checkpoint,proof,selected_rows)
    return records,coverage,rho


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--training-start-epoch',type=float,required=True)
    p.add_argument('--training-target',type=int,choices=(5000,10000),default=5000,
        help='Final training iteration; use 10000 explicitly for legacy runs')
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--num-envs',type=int,default=32)
    p.add_argument('--pool-size',type=int,default=128);p.add_argument('--pool-frames',type=int,default=1000000)
    p.add_argument('--max-clips',type=int,default=8);p.add_argument('--max-frames',type=int,default=50000)
    p.add_argument('--window-transitions',type=int,choices=(256,512),default=512)
    p.add_argument('--whole-clips',action='store_true',help='Use complete clips instead of bounded probe windows')
    p.add_argument('--min-budget-seconds',type=float,default=20.);p.add_argument('--poll-seconds',type=float,default=10.)
    args=p.parse_args()
    if (args.gpu!=1 or not 1<=args.num_envs<=256 or not 1<=args.pool_size<=256 or not 1<=args.max_clips<=args.pool_size
        or not math.isfinite(args.min_budget_seconds) or not math.isfinite(args.poll_seconds)
        or args.min_budget_seconds<=0 or args.poll_seconds<1):
        p.error('Invalid GPU 1, pool, clip or timing bounds')
    if not 0<args.training_start_epoch<=time.time():p.error('Actual training start epoch required')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'worker.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another probe worker owns this budget ledger')
    ledger_path=args.output/'budget.json'
    if not ledger_path.exists() and any(p.name!='worker.lock' for p in args.output.iterdir()):
        raise ValueError('Budget ledger missing from nonempty probe output; restore its original ledger or use a new empty output directory')
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(training_start_epoch=args.training_start_epoch,training_target=args.training_target,used_seconds=0.,completed={})
    if ledger['training_start_epoch']!=args.training_start_epoch:raise ValueError('Budget ledger training start differs')
    if type(ledger.get('training_target')) is not int or ledger['training_target']!=args.training_target:
        raise ValueError('Budget ledger training target missing or differs; retain the original target or use a new output directory for an unbound legacy ledger')
    if any(type(it) is not int or not 0<=it<=args.training_target for it in ledger['completed'].values()):
        raise ValueError('Budget ledger completion exceeds training target')
    save_atomic(ledger_path,ledger)
    from morphmimic.scripts.gpu_guard import select_gpu
    select_gpu(args.gpu,True)
    pools={};configured=False
    def timeout(signum,frame):raise TimeoutError('Probe GPU budget exhausted')
    signal.signal(signal.SIGALRM,timeout)
    while True:
        made_progress=False
        for robot in ROBOTS:
            ckpt=latest_completed(args.run_root/robot)
            if ckpt is None:continue
            iteration=int(ckpt.stem.removeprefix('ckpt_'))
            if iteration>args.training_target:raise ValueError('Checkpoint exceeds requested training target')
            if not needs_probe(ledger['completed'],robot,iteration):continue
            checkpoint_time=ckpt.stat().st_mtime
            if checkpoint_time<args.training_start_epoch:raise ValueError('Checkpoint predates training start')
            training_seconds=checkpoint_time-args.training_start_epoch
            remaining=budget_remaining(training_seconds,ledger['used_seconds'])
            if remaining<args.min_budget_seconds:continue
            made_progress=True
            before=ledger['used_seconds'];ledger['used_seconds']=before+remaining
            save_atomic(ledger_path,ledger)  # Reserve the full allowance before touching CUDA.
            started=time.monotonic();signal.setitimer(signal.ITIMER_REAL,remaining)
            try:
                import torch
                from mjlab.utils.torch import configure_torch_backends
                from morphmimic.eval.benchmark import make_eval_env
                if not configured:
                    configure_torch_backends();configured=True
                if robot not in pools:
                    clips,pool,proof=prepare_pool(args.data_root,robot,args.output,args.pool_size,args.pool_frames,13)
                    env=make_eval_env(str(pool),args.num_envs,'cuda:0',library_dtype='float16',robot=robot,
                        human_root=proof['human_root'],allow_missing_human=True,control_graph=True)
                    pools[robot]=(env,clips,proof)
                env,clips,proof=pools[robot];checkpoint=checkpoint_identity(ckpt,robot)
                records,coverage,rho=probe_one(env,clips,checkpoint,proof,args.data_root,robot,iteration,
                    args.max_clips,args.max_frames,started+remaining,None if args.whole_clips else args.window_transitions)
                torch.cuda.synchronize()
                elapsed=time.monotonic()-started
                if elapsed>remaining:raise TimeoutError('Probe exceeded available budget')
                document=dict(schema_version=1,robot=robot,interface='combined',fps=50,created_utc=datetime.now(timezone.utc).isoformat(),
                    lookback_seconds=10,iteration=iteration,interface_weights=WEIGHTS,failures_by_interface=records,
                    probe_reset_share=rho,base_exploration_share=1-rho,
                    failures=aggregate_failures(records,WEIGHTS),coverage=coverage,**checkpoint,**proof,
                    training_seconds=training_seconds,probe_seconds=elapsed,control_graph=True)
                save_atomic(args.output/(robot+'.json'),document)
                ledger['completed'][robot]=iteration
            except BaseException as error:
                ledger['last_error']=dict(robot=robot,iteration=iteration,error=f'{type(error).__name__}: {error}')
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                ledger['used_seconds']=before+time.monotonic()-started
                save_atomic(ledger_path,ledger)
        if all(ledger['completed'].get(r,0)>=args.training_target for r in ROBOTS):break
        if not made_progress:
            latest=[latest_completed(args.run_root/r) for r in ROBOTS]
            if all(p is not None and int(p.stem.removeprefix('ckpt_'))>=args.training_target for p in latest):
                ledger['incomplete_robots']=[r for r in ROBOTS if ledger['completed'].get(r,0)<args.training_target]
                ledger['stop_reason']='Final checkpoints exist but no probe fits available budget'
                save_atomic(ledger_path,ledger);break
            time.sleep(args.poll_seconds)

if __name__=='__main__':main()
