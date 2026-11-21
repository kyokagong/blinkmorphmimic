"""Plot observed training scalars and fixed evaluation cases without interpolation."""
from __future__ import annotations
import argparse
import csv
import math
import json
import re
from pathlib import Path
from scripts.evaluate_optimized import cases, audit_results, checkpoint_iterations, flatten_summary, validate_selection
from scripts.prepare_optimized import ROBOTS, sha
from morphmimic.constants import R_MAX


def load_results(inputs, expected_manifest_sha):
    rows=[];checkpoint_hashes={};verified={}
    for value in inputs:
        path=Path(value);path=path/'results.json' if path.is_dir() else path
        if path.is_symlink() or not path.is_file():raise ValueError(f'Invalid evaluation result path: {path}')
        data=json.loads(path.read_text())
        if not isinstance(data,list) or not data:raise ValueError(f'Empty or invalid evaluation results: {path}')
        for row in data:
            if not isinstance(row,dict) or row.get('manifest_sha256')!=expected_manifest_sha:
                raise ValueError('Evaluation manifest SHA differs')
            if row.get('status') not in ('ok','error'):raise ValueError('Invalid evaluation status')
            if row['status']=='ok':
                if row.get('metric_definition')!='legacy_tracking_reward_v1' or row.get('normalization_r_max')!=R_MAX:
                    raise ValueError('Evaluation metric definition differs')
                digest=row.get('checkpoint_sha256');ckpt=Path(row['checkpoint'])
                if not isinstance(digest,str) or re.fullmatch('[a-f0-9]{64}',digest) is None:
                    raise ValueError('Invalid evaluation checkpoint SHA')
                group=(row['iteration'],row['robot'])
                if group in checkpoint_hashes and checkpoint_hashes[group]!=digest:raise ValueError('Evaluation checkpoint hashes differ')
                checkpoint_hashes[group]=digest
                if ckpt.is_symlink() or not ckpt.is_file() or ckpt.name!=f"ckpt_{row['iteration']}.pt":
                    raise ValueError('Invalid fixed evaluation checkpoint')
                if ckpt not in verified:verified[ckpt]=sha(ckpt)
                if verified[ckpt]!=digest:raise ValueError('Evaluation checkpoint changed')
                stem=f"{row['robot']}_{row['dataset']}_{row['interface']}_{row['iteration']}"
                for suffix,key in (('.csv','per_clip_sha256'),('.yaml','subset_sha256')):
                    artifact=path.parent/(stem+suffix)
                    if artifact.is_symlink() or not artifact.is_file() or sha(artifact)!=row.get(key):
                        raise ValueError(f'Evaluation artifact changed: {artifact}')
            rows.append(row)
    return rows


def series(rows, metric):
    rows=sorted(rows,key=lambda r:r['iteration'])
    return [r['iteration'] for r in rows], [float(r[metric]) if r['status']=='ok' and metric in r else float('nan') for r in rows]


def normalize_scalar_events(events):
    points=[]
    for event in events:
        value=float(event.value);finite=math.isfinite(value)
        points.append(dict(iteration=int(event.step),value=value if finite else None,
                           status='ok' if finite else 'nonfinite'))
    return points


def training_checkpoint_rows(train, iterations=None):
    iterations, _ = validate_selection(iterations, None)
    tags=sorted({tag for robot in ROBOTS for tag in train.get(robot,{}).get('tags',{})})
    rows=[]
    for robot in ROBOTS:
        for tag in tags:
            points=train.get(robot,{}).get('tags',{}).get(tag,[])
            for iteration in iterations:
                selected=[p for p in points if p['iteration']==iteration]
                if not selected:status,value='missing',None
                elif len(selected)>1:status,value='ambiguous',None
                else:status,value=selected[0]['status'],selected[0]['value']
                rows.append(dict(robot=robot,tag=tag,iteration=iteration,status=status,value=value))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,nargs='+')
    a=p.parse_args()
    try:iterations,_=validate_selection(a.iterations,None)
    except ValueError as error:p.error(str(error))
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    manifest_path=a.data_root/'manifest.json';manifest_sha=sha(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    results=load_results(a.evaluation,manifest_sha)
    results=[flatten_summary(r) if r.get('status')=='ok' and 'drift' in r else r for r in results]
    expected=[(it,r,d,i) for it in iterations for r in ROBOTS for d,i in cases(manifest['clips'],r)]
    results=audit_results(results,expected)
    (a.out/'evaluation-audit.json').write_text(json.dumps(results,indent=2))
    groups=sorted({(r['dataset'],r['interface']) for r in results})
    for metric in ('mean_norm_return','failure_rate','drift_mean_xy','drift_final_xy','drift_mean_abs_yaw_deg'):
        fig,axes=plt.subplots(len(groups),1,figsize=(9,3*len(groups)),squeeze=False,sharex=True)
        for ax,(dataset,interface) in zip(axes[:,0],groups):
            for robot in ROBOTS:
                selected=[r for r in results if r['robot']==robot and r['dataset']==dataset and r['interface']==interface]
                if selected:
                    x,y=series(selected,metric);ax.plot(x,y,'o-',label=robot)
            ax.set_title(f'{dataset} / {interface}');ax.set_ylabel(metric);ax.grid(alpha=.25);ax.legend()
        axes[-1,0].set_xlabel('Completed iteration');fig.tight_layout()
        fig.savefig(a.out/f'eval_{metric}.png',dpi=160);fig.savefig(a.out/f'eval_{metric}.svg');plt.close(fig)
    train={}
    for robot in ROBOTS:
        events=list((a.run_root/robot).glob('events.out.tfevents.*'))
        if not events:
            train[robot]={'status':'missing events'};continue
        accumulator=EventAccumulator(str(a.run_root/robot),size_guidance={'scalars':0});accumulator.Reload()
        train[robot]={'status':'ok','tags':{tag:normalize_scalar_events(accumulator.Scalars(tag)) for tag in accumulator.Tags()['scalars']}}
    (a.out/'training-scalars.json').write_text(json.dumps(train,indent=2,allow_nan=False))
    checkpoint_rows=training_checkpoint_rows(train, iterations=iterations)
    (a.out/'training-checkpoints.json').write_text(json.dumps(checkpoint_rows,indent=2,allow_nan=False))
    with (a.out/'training-checkpoints.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=('robot','tag','iteration','status','value'))
        writer.writeheader();writer.writerows(checkpoint_rows)
    tags=sorted({tag for data in train.values() for tag in data.get('tags',{})})
    for index,tag in enumerate(tags):
        fig,ax=plt.subplots(figsize=(9,4))
        for robot in ROBOTS:
            points=train[robot].get('tags',{}).get(tag,[])
            if points:
                line,=ax.plot([x['iteration'] for x in points],[x['value'] if x['value'] is not None else float('nan') for x in points],'.-',label=robot)
                marked=[r for r in checkpoint_rows if r['robot']==robot and r['tag']==tag and r['status']=='ok']
                if marked:ax.scatter([r['iteration'] for r in marked],[r['value'] for r in marked],
                    marker='o',s=45,facecolors='none',edgecolors=line.get_color(),zorder=3)
        ax.set_title(tag);ax.set_xlabel('Completed iteration');ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(a.out/f'train_{index:03}.png',dpi=160);plt.close(fig)
    (a.out/'training-tags.json').write_text(json.dumps(dict(enumerate(tags)),indent=2))
    if sha(manifest_path)!=manifest_sha:raise ValueError('Prepared manifest changed while plotting')
    if any(r['status']!='ok' for r in results) or any(x['status']!='ok' for x in train.values()):
        raise SystemExit('Missing or failed data recorded in plot audit')

if __name__=='__main__':main()
