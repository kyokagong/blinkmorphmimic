"""Prepare MotionDecode baji/dance robot references; no human ground truth is invented."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,re,sys,time
from functools import cache
from pathlib import Path
from types import ModuleType
import numpy as np
from morphmimic.data.motiondecode import native_rows,convert_reference,CONVERSION_VERSION,VALIDATION_SCHEMA
from morphmimic.data.local_prepare import _retarget_function,sha256,partition


@cache
def q1_retarget(repo):
    sys.path.insert(0,repo)
    path=Path(repo)/'tools/retarget_q1.py';source=path.read_text()
    old='from supermimic.robot import G1_JOINT_NAMES'
    if source.count(old)!=1:raise ValueError('Upstream Q1 metadata import differs')
    module=ModuleType('blinkmorph_q1_retarget');module.__file__=str(path)
    exec(compile(source.replace(old,'from morphmimic.constants import JOINT_NAMES as G1_JOINT_NAMES'),str(path),'exec'),module.__dict__)
    return module.retarget


def prepare_one(task):
    import torch
    torch.set_num_threads(1)
    source,rel,output,repo,fps=task;source=Path(source);output=Path(output)
    kind='baji' if source.stem=='WP_Baji_Quan' else 'dance'
    name='motiondecode_'+kind+'_'+re.sub('[^A-Za-z0-9_-]','_',source.stem)+'_'+hashlib.sha256(rel.encode()).hexdigest()[:8]
    receipt=output/'receipts'/(name+'.json')
    identity=dict(original_sha256=sha256(source),original_fps=fps,relative_source=rel,conversion_version=CONVERSION_VERSION)
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['identity'].get('conversion_version')!=CONVERSION_VERSION:
            raise ValueError('Legacy MotionDecode conversion receipt; rebuild in a new output directory')
        if saved['identity']!=identity:raise ValueError('Existing MotionDecode source identity differs')
        for row in saved['clips']:
            if sha256(output/row['robot']/(name+'.npz'))!=row['output_sha256']:raise ValueError('Prepared output changed')
            if row['robot']=='bumi' and row.get('converted_validation',{}).get('schema')!=VALIDATION_SCHEMA:
                raise ValueError('Legacy Bumi conversion validation; rebuild in a new output directory')
        return saved
    rows,info=native_rows(source,source_fps=fps)
    native=output/'native30';g1=native/'g1'/(name+'.csv');g1.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(g1,rows,delimiter=',',fmt='%.9f')
    result=dict(identity=identity,clips=[],errors=[])
    qfolder=native/'q1';qfolder.mkdir(exist_ok=True);(qfolder/'reports').mkdir(exist_ok=True)
    _,qr=q1_retarget(repo)((str(g1.parent),str(qfolder),name,dict(sha256=sha256(g1))))
    candidates={'g1':(g1,dict(source_format=info))}
    if qr['status']=='passed':candidates['q1']=(qfolder/g1.name,qr)
    else:result['errors'].append(dict(robot='q1',name=name,error=qr.get('error')))
    try:
        bumi=native/'bumi'/g1.name;bumi.parent.mkdir(exist_ok=True)
        fn=_retarget_function(repo);br=fn(g1,bumi)
        from supermimic.bumi_ik import BumiIK,control_samples
        import mujoco
        original=np.loadtxt(bumi,delimiter=',',ndmin=2);ik=BumiIK()
        fixed=ik.ground_motion(ik.repair_interpolation(original))
        samples,_=control_samples(fixed);minimum=0.;contact=0.
        for row in samples:
            ik.forward(np.r_[row[:3],row[6],row[3:6]],row[7:]);minimum=min(minimum,float(ik.ground_heights().min()))
            mujoco.mj_collision(ik.model,ik.data);contact=min(contact,min((c.dist for c in ik.data.contact),default=0.))
        from supermimic.bumi import JOINT_NAMES,motor_of
        limits=np.array([motor_of(n)[1] for n in JOINT_NAMES]);speed=float((np.abs(np.diff(samples[:,7:],axis=0))/.02/limits).max())
        if minimum<-.002 or contact<-.002 or speed>1.0001:raise ValueError(f'Bumi audit ground={minimum} contact={contact} speed_ratio={speed}')
        np.savetxt(bumi,fixed,delimiter=',',fmt='%.9f')
        br.update(adaptation=fn.adaptation,min_ground_m=minimum,min_self_contact_m=contact,max_speed_ratio=speed,
                  max_joint_repair_rad=float(np.abs(fixed[:,7:]-original[:,7:]).max()))
        candidates['bumi']=(bumi,br)
    except Exception as error:result['errors'].append(dict(robot='bumi',name=name,error=f'{type(error).__name__}: {error}'))
    for robot,(path,audit) in candidates.items():
        target=output/robot/(name+'.npz');converted=convert_reference(path,target,robot)
        group='motiondecode/'+rel
        result['clips'].append(dict(name=name,robot=robot,dataset='motiondecode',source_id=group,group_id=group,
            split=partition(group),original_source=str(source),original_sha256=identity['original_sha256'],
            original_fps=fps,output=str(target),output_sha256=sha256(target),human_data=False,
            alignment_eligible=True,phases={},retarget_report=audit,**converted))
    receipt.parent.mkdir(exist_ok=True);receipt.write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retarget-repo',default='/data/kk/supermimic_vibed');p.add_argument('--source-fps',type=int,required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--limit',type=int)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=[args.source/'samples/4. Martial_Arts/WP_Baji_Quan.csv',*sorted((args.source/'samples/5.Dance').rglob('*.csv'))]
    if args.limit:files=files[:args.limit]
    if not files or not all(p.is_file() for p in files):raise ValueError('Selected MotionDecode source is missing')
    report=dict(schema_version=1,human_data=False,alignment_enabled=True,diagnostic_subset=bool(args.limit),
        source_fps=args.source_fps,source_fps_requires_confirmation=True,expected_source_count=465,clips=[],errors=[],complete=False)
    started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[(str(path),str(path.relative_to(args.source)),str(args.output),args.retarget_repo,args.source_fps) for path in files]
        for n,future in enumerate(concurrent.futures.as_completed([pool.submit(prepare_one,t) for t in tasks]),1):
            try:
                result=future.result();report['clips'].extend(result['clips']);report['errors'].extend(result['errors'])
            except Exception as error:report['errors'].append(dict(error=f'{type(error).__name__}: {error}'))
            report['clips'].sort(key=lambda r:(r['robot'],r['name']))
            temporary=args.output/'manifest.json.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(args.output/'manifest.json')
            print('MotionDecode sources',n,'/',len(files),'errors',len(report['errors']),'seconds',round(time.monotonic()-started,1),flush=True)
    report['complete']=not report['errors'] and len(files)==465 and not args.limit
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    if report['errors']:raise SystemExit(1)


if __name__=='__main__':main()
