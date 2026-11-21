"""Rebuild the complete active corpus with the reviewed arms-down neutral at both ends."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation
from morphmimic.data.neutral_bookends import (neutralize_motion, neutralize_human, validate_boundary,
    source_frame_map, _rot, _quat, _heading)
from morphmimic.data.local_prepare import _robot_model
from morphmimic.data.contact_calibration import FootContactGeometry
from morphmimic.data.seed_convert import finite_diff_velocities
from preview.reference import source_slice_digest

ROBOTS = ('q1', 'bumi', 'g1')
FRAME = 11446

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)

def load(path, expected=None):
    if expected and sha(path) != expected: raise ValueError('Source SHA differs: '+str(path))
    with np.load(path, allow_pickle=False) as f: return {k: f[k] for k in f.files}

def store(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz'); np.savez_compressed(temp, **arrays); temp.replace(path)

def stored_precision(arrays, native=True):
    keys = ('joint_pos', 'body_pos_w', 'body_quat_w') if native else ('human_pos_w', 'human_quat_w')
    for k in keys: arrays[k] = arrays[k].astype(np.float32)
    if native:
        for k, v in zip(('joint_vel','body_lin_vel_w','body_ang_vel_w'), finite_diff_velocities(
                *(arrays[k].astype(float) for k in keys), 50)):
            arrays[k] = v.astype(np.float32)
    return arrays

def canonical_targets(root, review_path):
    review = json.loads(review_path.read_text()); original = root/'provenance/original'
    original.mkdir(parents=True, exist_ok=True)
    shutil.copy2(review_path, original/'review-manifest.json')
    review['origin_review'] = dict(path=str(original/'review-manifest.json'),sha256=sha(review_path))
    review['original_robot_sources'] = {}; review['human_sources'] = {}; review['original_human_sources'] = {}; review['canonicalization'] = {}
    for robot in ROBOTS:
        identity = review['robot_sources'][robot]; source = Path(identity['npz'])
        motion = load(source, identity['sha256'])
        copied = original/robot/'walk2_subject4.npz'; copied.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source,copied)
        review['original_robot_sources'][robot] = dict(npz=str(copied),sha256=sha(copied))
        spec, model, _ = _robot_model(robot)
        floor = float(FootContactGeometry.from_mujoco(model,spec).sole_heights(
            motion['body_pos_w'][FRAME:FRAME+1][:,[3,6]],motion['body_quat_w'][FRAME:FRAME+1][:,[3,6]]).min())
        yaw = -_heading(motion['body_quat_w'][FRAME,0]); turn=Rotation.from_rotvec([0,0,yaw])
        moved=turn.apply(motion['body_pos_w'].reshape(-1,3)).reshape(motion['body_pos_w'].shape)
        translation=np.r_[-moved[FRAME,0,:2],-floor]
        motion['body_pos_w']=moved+translation
        motion['body_quat_w']=_quat(turn*_rot(motion['body_quat_w'])).reshape(motion['body_quat_w'].shape)
        stored_precision(motion)
        dest=root/'provenance/canonical'/robot/'walk2_subject4.npz';store(dest,motion)
        review['robot_sources'][robot]=dict(npz=str(dest),sha256=sha(dest))
        hp=source.parent.parent/'human50'/robot/source.name; human=load(hp)
        hcopy=original/'human50'/robot/source.name;hcopy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(hp,hcopy)
        review['original_human_sources'][robot]=dict(npz=str(hcopy),sha256=sha(hcopy))
        hyaw=-_heading(human['human_quat_w'][FRAME,0]); ht=Rotation.from_rotvec([0,0,hyaw])
        moved=ht.apply(human['human_pos_w'].reshape(-1,3)).reshape(human['human_pos_w'].shape)
        htranslation=np.r_[-moved[FRAME,0,:2],-moved[FRAME,[7,8],2].min()]
        human['human_pos_w']=moved+htranslation
        human['human_quat_w']=_quat(ht*_rot(human['human_quat_w'])).reshape(human['human_quat_w'].shape)
        hd=root/'provenance/canonical/human50'/robot/source.name;store(hd,stored_precision(human,False))
        review['human_sources'][robot]=dict(npz=str(hd),sha256=sha(hd))
        review['canonicalization'][robot]=dict(native_yaw_rad=yaw,native_translation=translation.tolist(),human_yaw_rad=hyaw,human_translation=htranslation.tolist(),native_joint_positions_unchanged=True)
    write_json(root/'standing-review.json',review)

_TARGETS={};_HUMANS={}
def init_worker(root):
    global _TARGETS,_HUMANS
    root=Path(root);review=json.loads((root/'standing-review.json').read_text())
    _TARGETS={r:load(review['robot_sources'][r]['npz'],review['robot_sources'][r]['sha256']) for r in ROBOTS}
    _HUMANS={r:load(review['human_sources'][r]['npz'],review['human_sources'][r]['sha256']) for r in ROBOTS}

def pad_native(out,c,front,back):
    n=c['frames']
    for k,v in list(out.items()):
        if v.ndim and v.shape[0]==n and k not in ('body_names','joint_names'):
            out[k]=np.concatenate([np.repeat(v[:1],front,axis=0),v,np.repeat(v[-1:],back,axis=0)])
    for k in ('entry_hold_end_frame','source_start_frame','source_stop_frame','exit_hold_start_frame'):c[k]+=front
    c['frames']+=front+back
    c['synthetic_intervals']=[[0,c['source_start_frame']],[c['source_stop_frame'],c['frames']]]


def prepare_group(job):
    root,rows,floors=job;root=Path(root);results=[];staged=[]
    for row in rows:
        robot=row['robot'];source=load(row['output'],row['output_sha256']);f=floors.get((robot,row['name']),{})
        offset=f.get('offset_m',0.) if f.get('valid') else 0.
        out,c=neutralize_motion(source,_TARGETS[robot],robot,source_floor_offset=offset)
        staged.append((row,out,c))
    start=max(c['source_start_frame'] for _,_,c in staged)
    length=max(c['frames']+start-c['source_start_frame'] for _,_,c in staged)
    he=max(c['entry_frames'] for _,_,c in staged); hx=max(c['exit_frames'] for _,_,c in staged)
    # Tail holds accommodate the common human bridge even when a native bridge is shorter.
    length=max(length,max(start+c['source_stop_frame']-c['source_start_frame']+hx+50 for _,_,c in staged))
    for row,out,c in staged:
        robot=row['robot'];front=start-c['source_start_frame'];back=length-c['frames']-front
        pad_native(out,c,front,back);c.update(human_entry_frames=he,human_exit_frames=hx)
        stored_precision(out);audit=validate_boundary(out,_TARGETS[robot],robot,c)
        # Convex native bridges may inherit endpoint violations but must introduce no new extrema.
        a,b=c['source_start_frame'],c['source_stop_frame'];q=out['joint_pos'];t=_TARGETS[robot]['joint_pos'][FRAME]
        violation=0.
        for bridge,edge in ((q[:a],q[a]),(q[b:],q[b-1])):
            lo=np.minimum(t,edge);hi=np.maximum(t,edge)
            violation=max(violation,float(np.maximum(lo-bridge,bridge-hi).max(initial=0)))
        if violation>2e-6:raise ValueError('New bridge joint extrema: '+row['name'])
        c['quality'].update(audit,bridge_joint_convex_hull_error_rad=violation)
        dest=root/robot/(row['name']+'.npz');store(dest,out)
        new=dict(row);new['pre_neutral_reference']=dict(path=row['output'],sha256=row['output_sha256'],frames=row['frames'])
        new.update(output=str(dest),output_sha256=sha(dest),frames=c['frames'],duration_seconds=(c['frames']-1)/50,
            neutral_contract=c,source_slice_sha256=source_slice_digest(out,a,b),
            phases=dict(entry=a/50,dance=b/50,return_to_neutral=c['exit_hold_start_frame']/50),
            source_body_window_original=[a,b],pair_alignment=dict(robot_start_frame=a,robot_stop_frame=b,robot_tail_frames=c['frames']-b,human_tail_frames=c['frames']-b))
        if row.get('human_available'):
            human=load(row['human_sidecar'],row['human_sha256']);side=neutralize_human(human,_HUMANS[robot],c)
            stored_precision(side,False);hd=root/'human50'/robot/(row['name']+'.npz');store(hd,side)
            new['pre_neutral_human']=dict(path=row['human_sidecar'],sha256=row['human_sha256'])
            new.update(human_sidecar=str(hd),human_sha256=sha(hd),human_joint_valid=side['human_joint_valid'].tolist(),human_source_slice_sha256=source_slice_digest(side,a,b))
        write_json(dest.with_suffix('.receipt.json'),dict(phases=new['phases'],neutral_contract=c,source_id=row['source_id']))
        results.append(new)
    return results


def remap_index(contract,index):
    mapping=source_frame_map(contract['original_source_frames'],contract['source_ramp_frames'])
    if not 0<=index<=contract['original_source_frames']:raise ValueError('Old source window outside clip')
    return contract['source_start_frame']+int(np.searchsorted(mapping,index,side='left'))

def remap_windows(old,rows):
    out=json.loads(json.dumps(old));by={(r['robot'],r['name']):r for r in rows}
    for key in ('windows','evaluation_cases'):
        for w in out.get(key,[]):
            c=by[w['robot'],w['clip']]['neutral_contract']
            for field in ('start_frame','stop_frame','stand_frame','hold_end_frame'):
                if field in w:w[field]=remap_index(c,w[field])
    for poses in out.get('preview_poses',{}).values():
        for robot,p in poses.items():
            c=by[robot,p['clip']]['neutral_contract']
            for field in ('start_frame','hold_frame'):p[field]=remap_index(c,p[field])
    out['max_recovery_seconds']=max(old.get('max_recovery_seconds',15.),max((w['stand_frame']-w['start_frame'])/50 for w in out['windows'])+1)
    out['derivation']='Original reset labels inverse-mapped through authorized slow edge clock; native arms-down neutral bookends retained.'
    return out

def finish(root,parent,rows):
    old=json.loads((parent/'manifest.json').read_text());review=root/'standing-review.json'
    old_by={(r['robot'],r['name']):r for r in old['clips']}
    for row in rows:
        if not row.get('human_available'):continue
        if not row.get('pair_source'):
            d=row['derivation']
            if row['dataset']=='neutral_recovery':
                row['pair_source']=old_by[row['robot'],d['source_clip']]['pair_source']
            elif row['dataset']=='seed_flips':
                row['pair_source']=dict(d['source_human'],fps=120,frames=d['source_metadata']['move_duration_frames'])
            else:raise ValueError('Missing original human source provenance')
        row['neutral_contract']['source_clock_status']=('unavailable_in_inherited_derived_sidecar'
            if row['dataset']=='neutral_recovery' else 'preserved_from_parent_sidecar')
    m=dict(old,clips=rows,human_root=str(root/'human50'),complete=True,
        neutral_review=dict(path=str(review),sha256=sha(review)),
        derivation=dict(parent_manifest=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json')),
            change='Full rewrite with approved static neutral entry/exit and synchronized edge retiming; middle source samples unchanged modulo fixed world gauge.',
            dynamics_qualified=False))
    pending=root/'.manifest.pending.json'
    write_json(pending,m);manifest_sha=sha(pending)
    subsets={}
    for robot in ROBOTS:
        for split in ('train','val'):
            entries=[r for r in rows if r['robot']==robot and r['split']==split]
            byname={r['name']:r for r in entries}
            template=json.loads((parent/robot/(split+'.yaml')).read_text())
            template['clips']=[byname[Path(value).stem]['output'] for value in template['clips']]
            p=root/robot/(split+'.yaml');write_json(p,template)
            if split=='train':subsets[robot]=sha(p)
    floor=json.loads((parent/'contact-floor.json').read_text());originals={(r['robot'],r['name']):r for r in floor['clips']}
    floor['clips']=[dict(originals[r['robot'],r['name']],path=r['output'],sha256=r['output_sha256'],offset_m=0.,
        original_offset_m=originals[r['robot'],r['name']]['offset_m'],reason='Original calibration confidence retained; offset already subtracted from native core; synthetic neutral holds are not measured contacts.') for r in rows]
    floor['prepared_manifest_sha256']=manifest_sha;write_json(root/'contact-floor.json',floor)
    windows=remap_windows(json.loads((parent/'windows.json').read_text()),rows)
    windows.update(training_manifest_sha256=manifest_sha,reference_manifest_sha256=manifest_sha,subset_sha256=subsets,
        approved_target=dict(review_manifest=dict(path=str(review),sha256=sha(review)),frame=FRAME,fps=50))
    write_json(root/'windows.json',windows)
    flips=json.loads((parent/'flip-preview.json').read_text());by={(r['robot'],r['name']):r for r in rows}
    for f in flips['clips']:
        r=by[f['robot'],f['clip']];c=r['neutral_contract'];f['source_window']=[c['source_start_frame'],c['source_stop_frame']];f['frames']=r['frames']
    flips.update(training_manifest_sha256=manifest_sha,reference_manifest_sha256=manifest_sha,subset_sha256=subsets)
    write_json(root/'flip-preview.json',flips)
    write_json(root/'validation.json',dict(complete=True,clips=len(rows),human_clips=sum(bool(r.get('human_available')) for r in rows),
        manifest_sha256=manifest_sha,maximum_fk_error_m=max(r['neutral_contract']['quality']['maximum_native_fk_error_m'] for r in rows),
        maximum_bridge_convex_hull_error_rad=max(r['neutral_contract']['quality']['bridge_joint_convex_hull_error_rad'] for r in rows),
        dynamics_qualified=False,all_middle_samples_original_speed=True))
    pending.replace(root/'manifest.json')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--limit-groups',type=int)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Refusing to overwrite a completed dataset')
    canonical_targets(root,a.review)
    parent_sha=sha(a.parent/'manifest.json')
    old=json.loads((a.parent/'manifest.json').read_text())
    floor_document=json.loads((a.parent/'contact-floor.json').read_text())
    floors={(r['robot'],r['name']):r for r in floor_document['clips']}
    if (not old.get('complete') or floor_document['prepared_manifest_sha256']!=parent_sha
            or len(floors)!=len(old['clips']) or any(floors.get((r['robot'],r['name']),{}).get('sha256')!=r['output_sha256'] for r in old['clips'])):
        raise ValueError('Parent inventory or floor identities differ')
    archived_review=json.loads((root/'standing-review.json').read_text())
    target_rows={r['robot']:r for r in old['clips'] if r['name']=='walk2_subject4'}
    if (sha(a.review)!=old['derivation']['approved_target']['review_manifest']['sha256']
            or any(archived_review['original_robot_sources'][r]['sha256']!=target_rows[r]['output_sha256']
                or archived_review['original_human_sources'][r]['sha256']!=target_rows[r]['human_sha256'] for r in ROBOTS)):
        raise ValueError('Approved robot/human target identity differs from parent approval')
    groups=defaultdict(list)
    for r in old['clips']:groups[r['source_id']].append(r)
    jobs=[(str(root),rs,{(r['robot'],r['name']):floors.get((r['robot'],r['name']),{}) for r in rs}) for rs in groups.values()]
    if a.limit_groups:jobs=jobs[:a.limit_groups]
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(a.workers,initializer=init_worker,initargs=(str(root),)) as pool:
        for i,result in enumerate(pool.map(prepare_group,jobs,chunksize=1)):
            rows.extend(result)
            if i%100==0:print(json.dumps(dict(groups=i+1,clips=len(rows),total_groups=len(jobs),elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    by={(r['robot'],r['name']):r for r in rows};rows=[by[r['robot'],r['name']] for r in old['clips'] if (r['robot'],r['name']) in by]
    if a.limit_groups:write_json(root/'qualification.json',dict(clips=rows,complete=False));return
    if len(rows)!=len(old['clips']):raise ValueError('Incomplete inventory')
    if sha(a.parent/'manifest.json')!=parent_sha:raise ValueError('Parent manifest changed during preparation')
    finish(root,a.parent,rows);print((root/'validation.json').read_text(),flush=True)

if __name__=='__main__':main()
