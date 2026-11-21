"""Prepare traceable, body-only three-robot human pairs from original motion capture."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,time
from pathlib import Path
import numpy as np
from morphmimic.constants import HUMAN_JOINTS
from morphmimic.data.human_convert import resample_poses,soma_slot_poses
from morphmimic.data.human_prepare import seed_index,write_pairs
from morphmimic.data.lafan_convert import lafan_slot_poses


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare_source(task):
    path=Path(task['source']);dataset=task['dataset'];records=task['records']
    if not path.is_file():raise FileNotFoundError(path)
    if dataset=='seed':
        pos,quat,dt=soma_slot_poses(path);valid=np.ones(24,dtype=bool)
    else:pos,quat,dt,valid=lafan_slot_poses(path)
    fps=round(1/dt);n=len(pos)
    if fps!=task['fps'] or n!=task['source_frames']:raise ValueError(f'{path}: original frame metadata differs ({n}, {fps})')
    pos,quat=resample_poses(pos,quat,fps,50)
    human=dict(human_pos_w=pos,human_quat_w=quat,human_joint_names=np.array(HUMAN_JOINTS),
               human_joint_valid=valid,fps=np.array(50),source_frames=np.array(n))
    robots={}
    for record in records:
        robot=record['robot'];p=Path(task['parent_root'])/robot/(record['name']+'.npz')
        if sha(p)!=record['output_sha256']:raise ValueError(f'Parent reference changed: {p}')
        with np.load(p,allow_pickle=False) as arrays:robots[robot]={k:arrays[k] for k in arrays.files}
    source=dict(path=str(path),sha256=sha(path),fps=fps,frames=n,
                rest_convention='SOMA_SMPL' if dataset=='seed' else 'LAFAN_GMR_calibrated',
                world_yaw_degrees=0 if dataset=='seed' else 90)
    return write_pairs(task['output_root'],records,robots,human,source=source,resume=task['resume'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-root',type=Path,default=Path('/data/kk/data/blinkmorph_human_raw/seed_selected'))
    parser.add_argument('--metadata',type=Path,default=Path('/data/kk/data/bones-seed-human/metadata/seed_metadata_v004.csv'))
    parser.add_argument('--lafan-root',type=Path,default=Path('/data/kk/data/blinkmorph_human_raw/lafan1'))
    parser.add_argument('--datasets',nargs='+',choices=('seed','lafan1'),default=['seed','lafan1'])
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.parent_root/'manifest.json';parent=json.loads(manifest_path.read_text());groups={}
    for row in parent['clips']:
        if row['dataset'] in args.datasets:groups.setdefault((row['dataset'],row['name']),[]).append(row)
    originals={key:next(r for r in rows if r['robot']=='q1')['upstream_record']['original_identity']['filename']
               for key,rows in groups.items() if key[0]=='seed'}
    index=seed_index(args.metadata,set(originals.values())) if originals else {}
    tasks=[]
    for (dataset,name),records in sorted(groups.items()):
        if {r['robot'] for r in records}!={'q1','bumi','g1'}:raise ValueError(f'Incomplete robot triplet: {name}')
        if dataset=='seed':
            source=index[originals[(dataset,name)]];path=args.seed_root/source['path'];frames=source['source_frames'];fps=120
        else:
            path=args.lafan_root/(name+'.bvh');frames=next(r for r in records if r['robot']=='g1')['source_frames'];fps=30
        tasks.append(dict(dataset=dataset,source=str(path),source_frames=frames,fps=fps,records=records,
                          parent_root=str(args.parent_root),output_root=str(args.output),resume=args.resume))
    if args.limit:tasks=tasks[:args.limit]
    report=dict(schema_version=1,parent_manifest_sha256=sha(manifest_path),human_data=True,alignment_enabled=True,
                expected_source_count=11046,datasets=args.datasets,diagnostic_subset=bool(args.limit),
                five_point_indices=[0,7,8,20,21],clips=[],errors=[],complete=False)
    package=Path(__file__).resolve().parents[1]
    report['conversion_sources']={str(path.relative_to(package)):sha(path) for path in
        [Path(__file__),*(package/'data'/name for name in ('bvh.py','human_convert.py','human_prepare.py','lafan_convert.py'))]}
    existing=args.output/'manifest.json'
    if existing.exists() and not args.resume:raise FileExistsError(existing)
    def save():
        temp=existing.with_suffix('.json.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(existing)
    started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_tasks={pool.submit(prepare_source,t):t for t in tasks}
        for i,future in enumerate(concurrent.futures.as_completed(future_tasks),1):
            task=future_tasks[future]
            try:
                rows=future.result()
                for row in rows:
                    row['output']=str(Path(row['output']).relative_to(args.output))
                    row['human_sidecar']=str(Path(row['human_sidecar']).relative_to(args.output))
                report['clips'].extend(rows)
            except Exception as error:
                report['errors'].append(dict(source=task['source'],error=f'{type(error).__name__}: {error}'))
            if i%20==0 or i==len(tasks):
                save();print('prepared',i,'/',len(tasks),'errors',len(report['errors']),'seconds',round(time.monotonic()-started,1),flush=True)
    report['clips'].sort(key=lambda r:(r['robot'],r['dataset'],r['name']))
    report['complete']=not report['errors'] and len(tasks)==report['expected_source_count'] and not args.limit
    report['converted_source_count']=len(report['clips'])//3
    for robot in ('q1','bumi','g1'):
        folder=args.output/robot;folder.mkdir(exist_ok=True)
        for split in ('train','val','all'):
            names=[row['name']+'.npz' for row in report['clips'] if row['robot']==robot and (split=='all' or row['split']==split)]
            (folder/(split+'.yaml')).write_text('root: .\nclips:\n'+''.join('  - '+n+'\n' for n in names))
    save();print('Human preparation sources',report['converted_source_count'],'complete',report['complete'],flush=True)
    if report['errors']:raise SystemExit(1)


if __name__=='__main__':main()
