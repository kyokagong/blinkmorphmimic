"""Replace recovery augmentation with bounded, locally anchored get-ups ending at full neutral."""
import argparse,copy,hashlib,json,os,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation
import numpy as np
from morphmimic.data.neutral_bookends import neutralize_motion,neutralize_human,validate_boundary,source_frame_map,_rot,_quat
from scripts.prepare_unified_neutral import load,store,stored_precision,write_json,pad_native,sha
from preview.reference import source_slice_digest

ROBOTS=('q1','bumi','g1')

def recovery_cut(motion,neutral,category):
    """Never keep post-rise locomotion or force a multi-metre path back to neutral."""
    if category=='half_squat':return 2,'Recorded initial half-squat pose; derived direct neutral rise without source turning/walking.'
    p=motion['body_pos_w'];up=_rot(motion['body_quat_w'][:,0]).as_matrix()[:,2,2]
    height=p[:,0,2]-p[:,[3,6],2].min(-1)
    target_height=neutral['body_pos_w'][11446,0,2]-neutral['body_pos_w'][11446,[3,6],2].min()
    good=(up>.94)&(height>=.9*target_height)
    found=next((i+5 for i in range(5,len(p)-4) if good[i:i+5].all()),None)
    if found is None:raise ValueError('No sustained near-upright pose inside the genuine recovery prefix')
    excursion=np.linalg.norm(p[:found,0,:2]-p[0,0,:2],axis=-1).max()
    if excursion>float(neutral['body_pos_w'][11446,0,2]):raise ValueError(f'Recovery leaves its local footprint ({excursion:.3f}m); not an in-place example')
    return found,'First five-frame sustained upright; later walking/turning discarded. Whole prefix gets one fixed end-anchor transform.'

def clip_arrays(data,a,b,native):
    keys=('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w') if native else ('human_pos_w','human_quat_w')
    result={k:v.copy() for k,v in data.items() if k in keys or k in ('fps','body_names','joint_names','root_body','robot','human_joint_names','human_joint_valid')}
    for key in keys:result[key]=result[key][a:b].copy()
    return result

def rotate_human_basis(human):
    turn=Rotation.from_rotvec([0.,0.,np.pi/2])
    human['human_pos_w']=turn.apply(human['human_pos_w'].reshape(-1,3)).reshape(human['human_pos_w'].shape)
    human['human_quat_w']=_quat(turn*_rot(human['human_quat_w'])).reshape(human['human_quat_w'].shape)
    if 'root_yaw_rotation_rad' in human:human['root_yaw_rotation_rad']=human['root_yaw_rotation_rad']+np.pi/2
    if 'root_translation_xyz' in human:human['root_translation_xyz']=turn.apply(human['root_translation_xyz'])
    return stored_precision(human,False)

def fix_human_basis_job(job):
    row,original=job
    if row['dataset']=='neutral_recovery' or not row.get('human_available'):return row
    human=rotate_human_basis(load(row['human_sidecar'],row['human_sha256']))
    path=Path(row['human_sidecar']);temporary=path.with_suffix('.tmp.npz')
    np.savez(temporary,**human);temporary.replace(path)
    row['human_sha256']=sha(path);c=row['neutral_contract']
    row['human_source_slice_sha256']=source_slice_digest(human,c['source_start_frame'],c['source_stop_frame'])
    row['human_basis_correction']=dict(parent_sidecar=dict(path=original['human_sidecar'],sha256=original['human_sha256']),world_yaw_rotation_rad=float(np.pi/2),reason='Correct SMPL lateral-X versus native forward-X basis; poses/timestamps otherwise preserved.')
    return row

def prepare(old,root):
    if root.exists():raise FileExistsError(root)
    parent=json.loads((old/'manifest.json').read_text());windows=json.loads((old/'windows.json').read_text());review=json.loads((old/'standing-review.json').read_text())
    skip={'manifest.json','releases.json','contact-floor.json','windows.json','flip-preview.json','validation.json','independent-audit.json'}
    shutil.copytree(old,root,copy_function=os.link,ignore=lambda folder,names:[n for n in names if Path(folder)==old and (n in skip or n.startswith('preview-'))])
    def remap(value):
        if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
        if isinstance(value,list):return [remap(v) for v in value]
        return str(root)+value[len(str(old)):] if isinstance(value,str) and value.startswith(str(old)+'/') else value
    review=remap(review)
    for robot in ROBOTS:
        identity=review['human_sources'][robot];path=Path(identity['npz'])
        store(path,rotate_human_basis(load(path,identity['sha256'])));identity['sha256']=sha(path)
        details=review['canonicalization'][robot];details['human_yaw_rad']+=np.pi/2
        details['human_translation']=Rotation.from_rotvec([0.,0.,np.pi/2]).apply(details['human_translation']).tolist()
    review['human_coordinate_convention']='SMPL24 local +X is left/lateral, forward -Y; canonical physical forward matches native +X, lateral +Y.'
    write_json(root/'standing-review.json',review)
    targets={r:load(review['robot_sources'][r]['npz']) for r in ROBOTS};humans={r:load(review['human_sources'][r]['npz']) for r in ROBOTS}
    originals={(w['robot'],w['clip']):w for w in windows['windows']};rows=[];new_windows=[];excluded=[]
    for original in parent['clips']:
        r=remap(copy.deepcopy(original))
        if r['dataset']!='neutral_recovery':rows.append(r);continue
        robot=r['robot'];c=original['neutral_contract'];a=c['source_start_frame'];mapping=source_frame_map(c['original_source_frames'],c['source_ramp_frames'])
        prefix=int(np.searchsorted(mapping,r['derivation']['original_prefix_frames']-1))+1
        motion=clip_arrays(load(original['output'],original['output_sha256']),a,a+prefix,True)
        human=clip_arrays(load(original['human_sidecar'],original['human_sha256']),a,a+prefix,False)
        w=originals[robot,r['name']];category=w['category']
        try:cut,reason=recovery_cut(motion,targets[robot],category)
        except ValueError as error:
            excluded.append(dict(robot=robot,name=r['name'],source_id=r['source_id'],reason=str(error)))
            for key in ('output','human_sidecar'):Path(r[key]).unlink()
            Path(r['output']).with_suffix('.receipt.json').unlink(missing_ok=True)
            continue
        if category=='half_squat':
            for data,keys in [(motion,('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')),(human,('human_pos_w','human_quat_w'))]:
                for key in keys:data[key]=np.repeat(data[key][:1],2,axis=0)
        else:
            motion=clip_arrays(motion,0,cut,True);human=clip_arrays(human,0,cut,False)
        out,nc=neutralize_motion(motion,targets[robot],robot,in_place=True)
        pad_native(out,nc,0,100)
        nc['source_clock_status']='unavailable_in_inherited_derived_sidecar'
        side=neutralize_human(human,humans[robot],nc)
        stored_precision(out);stored_precision(side,False)
        check=validate_boundary(out,targets[robot],robot,nc)
        np.testing.assert_allclose(side['human_pos_w'][0],side['human_pos_w'][-1],atol=1e-6,rtol=0)
        np.testing.assert_allclose(side['human_quat_w'][0],side['human_quat_w'][-1],atol=1e-6,rtol=0)
        aa,bb=nc['source_start_frame'],nc['source_stop_frame'];hold=nc['exit_hold_start_frame']
        reset_count=1 if category=='half_squat' else max(1,min(w['stop_frame']-a,cut-5))
        stop=aa+max(1,int(np.searchsorted(source_frame_map(cut,nc['source_ramp_frames']),reset_count)))
        stop=min(stop,bb-1)
        if hold+150-aa>750:raise ValueError('Corrected recovery exceeds the supported 15s reset-to-hold interval')
        nc['quality'].update(check,bridge_joint_convex_hull_error_rad=0.)
        store(r['output'],out);store(r['human_sidecar'],side)
        r.update(output_sha256=sha(r['output']),human_sha256=sha(r['human_sidecar']),frames=nc['frames'],duration_seconds=(nc['frames']-1)/50,
            neutral_contract=nc,source_slice_sha256=source_slice_digest(out,aa,bb),human_source_slice_sha256=source_slice_digest(side,aa,bb),
            source_body_window_original=[aa,bb],pair_alignment=dict(robot_start_frame=aa,robot_stop_frame=bb),
            phases=dict(entry=aa/50,dance=bb/50,return_to_neutral=hold/50))
        r['derivation']['in_place_correction']=dict(parent_reference=dict(path=original['output'],sha256=original['output_sha256']),parent_prefix_interval=[a,a+prefix],retained_prefix_frames=cut,
            method=reason,terminal='Exact ordinary initialization body positions, orientations, joint positions and zero velocities; root XY=0/yaw=0.',synthetic=True)
        write_json(Path(r['output']).with_suffix('.receipt.json'),dict(neutral_contract=nc,phases=r['phases'],source_id=r['source_id']))
        rows.append(r);new_windows.append(dict(robot=robot,clip=r['name'],category=category,source_id=r['source_id'],start_frame=aa,stop_frame=stop,stand_frame=hold,hold_end_frame=hold+150))
    prior={(r['robot'],r['name']):r for r in parent['clips']}
    with ProcessPoolExecutor(16) as pool:
        rows=list(pool.map(fix_human_basis_job,((row,prior[row['robot'],row['name']]) for row in rows),chunksize=8))
    for row in rows:row['neutral_contract']['human_coordinate_convention']='Native +X forward/+Y left; SMPL24 local +X left/-Y forward; physical headings aligned, same source clock.'
    for robot in ROBOTS:
        if {w['category'] for w in new_windows if w['robot']==robot}!={'lying','seated','half_squat'}:raise ValueError('Missing a recovery category after curation')
    manifest=dict(parent,clips=rows,human_root=str(root/'human50'),neutral_review=dict(path=str(root/'standing-review.json'),sha256=sha(root/'standing-review.json')),
        derivation=dict(parent_manifest=dict(path=str(old/'manifest.json'),sha256=sha(old/'manifest.json')),change='In-place recovery correction: discard post-rise locomotion, fixed standing-anchor coordinates, exact initial/full terminal neutral; half-squat uses recorded initial pose and a derived direct rise.',excluded_recovery_references=excluded,dynamics_qualified=False))
    write_json(root/'manifest.json',manifest);digest=sha(root/'manifest.json');valid={(r['robot'],r['name']):r for r in rows};subsets={}
    for robot in ROBOTS:
        for split in ('train','val'):
            data=json.loads((old/robot/(split+'.yaml')).read_text());data['clips']=[valid[robot,Path(p).stem]['output'] for p in data['clips'] if (robot,Path(p).stem) in valid]
            write_json(root/robot/(split+'.yaml'),data)
            if split=='train':subsets[robot]=sha(root/robot/'train.yaml')
    floor=json.loads((old/'contact-floor.json').read_text());floor['clips']=[dict(f,path=valid[f['robot'],f['name']]['output'],sha256=valid[f['robot'],f['name']]['output_sha256']) for f in floor['clips'] if (f['robot'],f['name']) in valid]
    floor['prepared_manifest_sha256']=digest;write_json(root/'contact-floor.json',floor)
    previous_preview=copy.deepcopy(windows.get('preview_poses',{}))
    windows.update(windows=new_windows,training_manifest_sha256=digest,reference_manifest_sha256=digest,subset_sha256=subsets,max_recovery_seconds=15.,
        evaluation_cases=[dict(w,start_frame=(w['start_frame']+w['stop_frame']-1)//2,stop_frame=(w['start_frame']+w['stop_frame']-1)//2+1,case_id=w['robot']+'/'+w['category']+'/'+w['clip']) for w in new_windows],preview_poses={},
        approved_target=dict(review_manifest=dict(path=str(root/'standing-review.json'),sha256=sha(root/'standing-review.json')),frame=11446,fps=50),derivation='First real rise only; hold at exact common full neutral. No inherited old stand hold frame.')
    for category in ('lying','seated','half_squat'):
        windows['preview_poses'][category]={}
        for robot in ROBOTS:
            choices=[w for w in new_windows if w['robot']==robot and w['category']==category]
            w=min(choices,key=lambda w:valid[robot,w['clip']]['derivation']['in_place_correction']['retained_prefix_frames'])
            start=(w['start_frame']+w['stop_frame']-1)//2
            previous=previous_preview.get(category,{}).get(robot)
            match=next((choice for choice in choices if previous and choice['clip']==previous['clip']),None)
            if match:
                row=valid[robot,match['clip']];c=row['neutral_contract']
                index=previous['start_frame']-row['derivation']['in_place_correction']['parent_prefix_interval'][0]
                if category=='half_squat':w=match;start=c['source_start_frame']
                elif 0<=index<c['original_source_frames']:
                    w=match;start=c['source_start_frame']+int(np.searchsorted(source_frame_map(c['original_source_frames'],c['source_ramp_frames']),index))
            windows['preview_poses'][category][robot]=dict(clip=w['clip'],start_frame=start,hold_frame=w['stand_frame'])
    write_json(root/'windows.json',windows)
    flips=json.loads((old/'flip-preview.json').read_text());flips.update(training_manifest_sha256=digest,reference_manifest_sha256=digest,subset_sha256=subsets);write_json(root/'flip-preview.json',flips)
    write_json(root/'in-place-validation.json',dict(complete=True,total_clips=len(rows),corrected_recoveries=len(new_windows),excluded_recoveries=excluded,manifest_sha256=digest,terminal_matches_initial_full_pose=True,human_terminal_matches_initial=True,dynamics_qualified=False))
    print((root/'in-place-validation.json').read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.parent.resolve(),a.output.absolute())
