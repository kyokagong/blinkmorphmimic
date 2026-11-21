"""Rebuild legacy Bumi MotionDecode references into a new, fully checked catalog."""
from __future__ import annotations
import argparse,concurrent.futures,copy,json,shutil,sys,time
from pathlib import Path

import numpy as np
from morphmimic.data.local_prepare import sha256
from morphmimic.data.motiondecode import CONVERSION_VERSION,convert_reference


def _rebuild_one(task):
    row,source_root,native_root,output,repo,source_hash=task
    if repo not in sys.path:sys.path.insert(0,repo)
    from supermimic.bumi import JOINT_NAMES as SDK_JOINT_NAMES
    source=Path(native_root)/(row['name']+'.csv')
    if sha256(source)!=source_hash:raise ValueError('Native source changed before conversion')
    target=Path(output)/row['output']
    stats=convert_reference(source,target,'bumi')
    with np.load(Path(source_root)/row['output']) as old,np.load(target) as new:
        permutation=[SDK_JOINT_NAMES.index(n) for n in new['joint_names'].tolist()]
        if not np.allclose(new['joint_pos'],old['joint_pos'][:,permutation],atol=2e-7,rtol=0):
            raise ValueError('Rebuild must only permute legacy SDK joint values')
        for field in ('body_pos_w','body_quat_w'):
            if not np.array_equal(new[field][:,0],old[field][:,0]):raise ValueError('Rebuild changed legacy base root trajectory')
    for key in ('frames','source_frames','source_duration_seconds','duration_seconds'):
        if abs(stats[key]-row[key])>1e-9:raise ValueError(f'Rebuild changed time axis: {key}')
    if sha256(source)!=source_hash:raise ValueError('Native source changed during conversion')
    return dict(row,**stats,output_sha256=sha256(target),conversion_version=CONVERSION_VERSION)


def rebuild(source_root,native_root,output,retarget_repo,*,workers=8):
    source_root=Path(source_root).resolve(strict=True);native_root=Path(native_root).resolve(strict=True)
    output=Path(output).absolute();repo=str(Path(retarget_repo).resolve(strict=True))
    if output.resolve().is_relative_to(source_root) or source_root.is_relative_to(output.resolve()):
        raise ValueError('Rebuild output must be separate from the legacy dataset')
    if workers<1:raise ValueError('workers must be positive')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Rebuild requires a new or empty output directory')
    manifest=source_root/'manifest.json';digest=sha256(manifest);baseline=json.loads(manifest.read_text())
    if baseline.get('conversion_version',1)>=CONVERSION_VERSION:raise ValueError('Expected the legacy joint-order dataset')
    if baseline.get('source_fps')!=120 or baseline.get('source_fps_requires_confirmation') is not False:
        raise ValueError('Expected the user-confirmed 120 Hz MotionDecode dataset')
    rows=baseline['clips'];keys=set();hashes={};csv_hashes={}
    for row in rows:
        key=(row['robot'],row['name']);relative=Path(row['output'])
        if key in keys or row['robot'] not in ('q1','bumi','g1') or relative!=Path(row['robot'])/(row['name']+'.npz') or len(relative.parts)!=2:
            raise ValueError('Duplicate or invalid legacy catalog path')
        keys.add(key);file=source_root/relative
        if file.is_symlink() or sha256(file)!=row['output_sha256']:raise ValueError('Legacy NPZ SHA differs from manifest')
        hashes[str(relative)]=row['output_sha256']
        if row['robot']=='bumi':
            source=native_root/(row['name']+'.csv')
            if source.is_symlink() or not source.is_file():raise FileNotFoundError(f'Missing regular native CSV: {source}')
            csv_hashes[row['name']]=sha256(source)
    if not csv_hashes:raise ValueError('No Bumi references to rebuild')
    output.mkdir(parents=True,exist_ok=True)
    result=copy.deepcopy(baseline);result['clips']=[]
    for row in rows:
        if row['robot']=='bumi':continue
        target=output/row['output'];target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source_root/row['output'],target)
        if sha256(target)!=row['output_sha256']:raise ValueError('Peer reference changed during copy')
        result['clips'].append(copy.deepcopy(row))
    tasks=[(r,str(source_root),str(native_root),str(output),repo,csv_hashes[r['name']]) for r in rows if r['robot']=='bumi']
    started=time.monotonic()
    def accept(row):
        result['clips'].append(row)
        count=sum(r['robot']=='bumi' for r in result['clips'])
        (output/'rebuild-progress.json').write_text(json.dumps(dict(validated_clips=count,expected_clips=len(tasks),complete=False)))
        if count%20==0 or count==len(tasks):print('Bumi validated',count,'/',len(tasks),'seconds',round(time.monotonic()-started,1),flush=True)
    if workers==1:
        for task in tasks:accept(_rebuild_one(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(_rebuild_one,t) for t in tasks]):accept(future.result())
    for relative,expected in hashes.items():
        if sha256(source_root/relative)!=expected:raise ValueError('Legacy data changed during rebuild')
    for name,expected in csv_hashes.items():
        if sha256(native_root/(name+'.csv'))!=expected:raise ValueError('Native CSV changed during rebuild')
    if sha256(manifest)!=digest:raise ValueError('Legacy manifest changed during rebuild')
    result['clips'].sort(key=lambda r:(r['robot'],r['name']))
    common=set.intersection(*({r['name'] for r in rows if r['robot']==robot} for robot in ('q1','bumi','g1')))
    for name in common:
        if len({r['source_id'] for r in rows if r['name']==name})!=1:raise ValueError('Comparison source identity differs')
    converted=[r['converted_validation'] for r in result['clips'] if r['robot']=='bumi']
    result.update(schema_version=2,conversion_version=CONVERSION_VERSION,common_three_robot_source_count=len(common),
        bumi_rebuild=dict(complete=True,expected_clips=len(tasks),validated_clips=len(converted),validated_frames=sum(r['frames'] for r in converted),
            source_dataset=str(source_root),source_manifest_sha256=digest,native_root=str(native_root),
            native_csv_sha256=csv_hashes,unchanged_peer_npzs=len(rows)-len(tasks),
            min_sole_height_m=min(r['min_sole_height_m'] for r in converted),min_self_contact_m=min(r['min_self_contact_m'] for r in converted),
            max_joint_speed_ratio=max(r['max_joint_speed_ratio'] for r in converted),
            fk_max_position_error_m=max(r['fk_max_position_error_m'] for r in converted)))
    # Keep original exclusions and completeness: repairing Bumi does not recover rejected sources.
    temporary=output/'manifest.json.tmp';temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(output/'manifest.json')
    (output/'rebuild-progress.json').write_text(json.dumps(dict(validated_clips=len(converted),expected_clips=len(tasks),complete=True)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--retarget-repo',default='/data/kk/supermimic_vibed')
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    result=rebuild(a.source_root,a.native_root,a.output,a.retarget_repo,workers=a.workers)
    print('COMPLETE',result['bumi_rebuild']['validated_clips'],'Bumi clips',result['bumi_rebuild']['validated_frames'],'frames',flush=True)


if __name__=='__main__':main()
