"""Pinned native full-clip comparison and actual flip diagnostics; no reference-only success."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import tempfile

SELECTED = ('dance1_subject1', 'motiondecode_baji_WP_Baji_Quan_17851016')
FIVE = (0,7,8,20,21)


def sha(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Require fixed file: '+str(path))
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def json_records(frame):
    records=frame.to_dict('records')
    for row in records:
        for key in ('t_xy_over','t_yaw_over'):
            if key in row and math.isnan(row[key]):row[key]=None
    return records


def select_rows(manifest,robot,suite):
    rows=[r for r in manifest['clips'] if r['robot']==robot and
          (r['name'] in SELECTED if suite=='selected' else r.get('dataset')=='seed_flips')]
    if not rows or (suite=='selected' and {r['name'] for r in rows}!=set(SELECTED)):
        raise ValueError('Missing selected full-clip references')
    return sorted(rows,key=lambda r:r['name'])


def pitch(quat, initial):
    """Projected up-vector angle in the initial pelvis frame, radians."""
    import numpy as np
    a=np.asarray(initial,float).copy();a[1:]*=-1
    b=np.asarray(quat,float)
    w=a[0]*b[0]-np.dot(a[1:],b[1:])
    xyz=a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])
    x,y,z=xyz
    return math.atan2(2*(x*z+w*y),1-2*(x*x+y*y))


def flip_metrics(trace, reference_pitch_rad, standing_height, *, failed, dt=.02):
    """A native full turn, actual flight and a continuous stable landing are all required."""
    import numpy as np
    if not trace: raise ValueError('Missing actual native flip states')
    angles=np.unwrap([r['pitch_rad'] for r in trace]); sign=1 if reference_pitch_rad>=0 else -1
    excursion=np.maximum.accumulate(sign*(angles-angles[0]))
    complete=excursion>=math.radians(330)
    hold=run=flight=longest_flight=0
    for turned,row in zip(complete,trace):
        flight=flight+1 if row.get('robot_floor_contacts',row['foot_floor_contacts'])==0 else 0
        longest_flight=max(longest_flight,flight)
        stable=(turned and row['foot_floor_contacts']>0 and row['root_up']>=.8
            and row['root_height_m']>=.8*standing_height and row['root_speed_m_s']<=.5
            and row['root_angular_speed_rad_s']<=1.5 and row['joint_rmse_rad']<=.75)
        run=run+1 if stable else 0;hold=max(hold,run)
    rotated=bool(complete.any());airborne=longest_flight*dt>=.06
    landed=hold*dt>=.5
    return dict(actual_full_rotation=rotated,actual_rotation_excursion_rad=float(excursion.max()),
        actual_flight_seconds=longest_flight*dt,actual_stable_landing_seconds=hold*dt,
        successful_full_rotation_and_stable_landing=bool(rotated and airborne and landed and not failed),
        failure_reason=('termination' if failed else 'incomplete_actual_rotation' if not rotated else
            'no_actual_flight' if not airborne else 'unstable_actual_landing' if not landed else None),
        mean_joint_rmse_rad=float(np.mean([r['joint_rmse_rad'] for r in trace])),
        max_actual_native_torque_nm=max(max(abs(x) for x in r['actual_native_torques_nm']) for r in trace))


class NativeTrace:
    def __init__(self,env,robot,rows):
        import numpy as np
        import torch
        import warp as wp
        from morphmimic.robots import get_robot
        self.env,self.wp,self.torch=env,wp,torch
        self.command=env.command_manager.get_term('motion');self.lib=self.command.library
        self.rows={r['name']:r for r in rows};self.trace=defaultdict(list)
        self.initial={};self.complete=set();self.owners={}
        model=env.sim.mj_model;spec=get_robot(robot)
        def body_id(name):
            found=[i for i in range(model.nbody) if model.body(i).name.split('/')[-1]==name]
            if len(found)!=1:raise ValueError('Cannot identify native body '+name)
            return found[0]
        self.root_id=body_id(spec.root_body)
        self.native_joint_names=list(spec.joint_names)
        joint_ids=[]
        for name in spec.joint_names:
            found=[i for i in range(model.njnt) if model.joint(i).name.split('/')[-1]==name]
            if len(found)!=1:raise ValueError('Cannot identify native joint '+name)
            joint_ids.append(int(model.jnt_dofadr[found[0]]))
        self.native_dof_ids=torch.tensor(joint_ids,device=env.device,dtype=torch.long)
        feet={body_id(spec.tracked_bodies[i]) for i in (3,6)}
        def is_foot(body):
            while body:
                if body in feet:return True
                body=int(model.body_parentid[body])
            return False
        self.foot_geoms=torch.tensor([is_foot(int(b)) for b in model.geom_bodyid],device=env.device)
        self.floor_geoms=torch.tensor(np.asarray(model.geom_type)==0,device=env.device)
        self.original=env.step
        env.step=self.step

    def step(self,action):
        c=self.command;lib=self.lib;torch=self.torch;wp=self.wp
        frame=c.global_frame.clone();clip=lib.frame_clip[frame]
        result=self.original(action)
        obs,reward,terminated,truncated,extras=result
        data=self.env.sim.wp_data;contact=data.contact
        n=int(wp.to_torch(data.nacon)[0])
        geom=wp.to_torch(contact.geom)[:n].long();world=wp.to_torch(contact.worldid)[:n].long()
        valid=(wp.to_torch(contact.dist)[:n]<=.002) & (geom>=0).all(-1)
        foot=(self.foot_geoms[geom[:,0]] & self.floor_geoms[geom[:,1]]) | (self.foot_geoms[geom[:,1]] & self.floor_geoms[geom[:,0]])
        support=valid & foot
        floor=valid & (self.floor_geoms[geom[:,0]] | self.floor_geoms[geom[:,1]])
        positions=wp.to_torch(contact.pos)[:n]
        com=wp.to_torch(data.subtree_com)[:,self.root_id]
        torque=wp.to_torch(data.qfrc_actuator).index_select(-1,self.native_dof_ids)
        root=c.robot_body_pos_w[:,0]-self.env.scene.env_origins
        quat=c.robot_body_quat_w[:,0]
        qerror=(c.robot_joint_pos-lib.joint_pos[frame].float()).square().mean(-1).sqrt()
        for env_id,clip_id in enumerate(clip.cpu().tolist()):
            name=lib.clip_names[clip_id]
            if name in self.complete:continue
            if name not in self.owners:self.owners[name]=env_id
            if self.owners[name]!=env_id:continue
            local=int(frame[env_id]-lib.clip_start[clip_id])
            if name not in self.initial:
                # Reset is on the exact first reference quaternion; no target turn is treated as actual.
                self.initial[name]=lib.body_quat_w[int(lib.clip_start[clip_id]),0].cpu().tolist()
            q=quat[env_id].cpu().tolist();points=positions[support & (world==env_id)]
            origin=self.env.scene.env_origins[env_id]
            row=dict(reference_frame=local,time_s=local*self.env.step_dt,
                pitch_rad=pitch(q,self.initial[name]),root_height_m=float(root[env_id,2]),
                root_up=1-2*(q[1]*q[1]+q[2]*q[2]),root_speed_m_s=float(c.robot_body_lin_vel_w[env_id,0].norm()),
                root_angular_speed_rad_s=float(c.robot_body_ang_vel_w[env_id,0].norm()),joint_rmse_rad=float(qerror[env_id]),
                foot_floor_contacts=int(points.shape[0]),robot_floor_contacts=int((floor & (world==env_id)).sum()),support_contact_xy_m=(points[:,:2]-origin[:2]).cpu().tolist(),
                com_projection_xy_m=(com[env_id,:2]-origin[:2]).cpu().tolist(),
                actual_native_torques_nm=torque[env_id].cpu().tolist(),
                terminated=bool(terminated[env_id]),truncated=bool(truncated[env_id]))
            self.trace[name].append(row)
            if row['terminated'] or row['truncated']:self.complete.add(name)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ckpt','manifest','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--robot',choices=('q1','bumi','g1'),required=True)
    parser.add_argument('--suite',choices=('selected','flips'),default='selected')
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--num-envs',type=int,default=16)
    args=parser.parse_args()
    if args.num_envs<1:parser.error('--num-envs must be positive')
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads(args.manifest.read_text());manifest_sha=sha(args.manifest)
    if manifest.get('complete') is not True:raise ValueError('Incomplete manifest')
    rows=select_rows(manifest,args.robot,args.suite)
    proof={str(args.manifest.resolve()):manifest_sha,str(args.ckpt.resolve()):sha(args.ckpt),str(Path(__file__).resolve()):sha(__file__)}
    for row in rows:
        proof[row['output']]=row['output_sha256']
        if row.get('human_available'):
            if not row.get('human_sidecar') or not row.get('human_sha256'):raise ValueError('Missing authentic human proof')
            proof[row['human_sidecar']]=row['human_sha256']
    for path,digest in proof.items():
        if sha(path)!=digest:raise ValueError('Reference SHA differs: '+path)
    from morphmimic.scripts.gpu_guard import select_gpu
    select_gpu(args.gpu)
    import torch
    import numpy as np
    from mjlab.utils.torch import configure_torch_backends
    from morphmimic.eval.benchmark import make_eval_env,run_benchmark,require_human_sidecars
    from morphmimic.eval.policy_loader import load_policy
    configure_torch_backends(allow_tf32=False)
    saved=torch.load(args.ckpt,map_location='cpu',weights_only=False);cfg=saved['cfg']
    iteration=saved['iteration']
    if (args.ckpt.name!=f'ckpt_{iteration}.pt' or cfg['robot']!=args.robot or cfg['optimizer']!='muon'):
        raise ValueError('Expected fixed matching Muon checkpoint')
    training_sha=saved.get('optimized_training',{}).get('prepared_manifest_sha256')
    if training_sha!=manifest_sha:
        transition=args.manifest.parent/'transition.json'
        from morphmimic.train.dataset_transition import verify_append_transition
        doc=verify_append_transition(transition,sha(transition),args.manifest,args.robot)
        if iteration!=6000 or training_sha!=doc['parent_manifest']['sha256']:
            raise ValueError('Only exact 6000 append-transition parent may evaluate the expanded manifest')
        proof[str(transition)]=sha(transition)
    standing_height=None
    if args.suite=='flips':
        from morphmimic.standing import load_standing_target
        transition=args.manifest.parent/'transition.json'
        receipt=json.loads(transition.read_text())['approved_target']['review_manifest']
        if sha(receipt['path'])!=receipt['sha256']:raise ValueError('Approved standing target SHA differs')
        target=load_standing_target(receipt['path'],args.robot)
        standing_height=float(target['body_pos'][0,2])
        proof[target['receipt']]=target['receipt_sha256'];proof[target['source']]=target['source_sha256']
    args.output.mkdir(parents=True)
    (args.output/'selection.json').write_text(json.dumps(rows,indent=2)+'\n')
    summaries=[];native_joint_order=None
    for interface in ('motion','human'):
        selected=[r for r in rows if interface=='motion' or (r.get('human_available') and all(r.get('human_joint_valid',[True]*24)[i] for i in FIVE))]
        if not selected:continue
        torch.manual_seed(0)
        policy,meta=load_policy(args.ckpt,'cuda:0',interface=interface,human_points='five',kp_horizon=1,goal_anchor=cfg.get('goal_anchor','reference'))
        with tempfile.TemporaryDirectory(prefix='fullclip-eval-') as temporary:
            subset=Path(temporary)/'subset.json';subset.write_text(json.dumps(dict(root='/',clips=[r['output'] for r in selected])))
            env=make_eval_env(str(subset),min(args.num_envs,len(selected)),'cuda:0',robot=args.robot,seed=0,
                builtin_pd=cfg.get('builtin_pd',False),joint_frame=cfg.get('joint_frame'),imu_body=cfg.get('imu_body'),
                library_dtype=cfg.get('library_dtype','float32'),human_root=manifest.get('human_root'),
                goal_anchor=cfg.get('goal_anchor','reference'),allow_missing_human=interface=='motion',control_graph=False)
            try:
                if interface=='human':require_human_sidecars(env,str(subset),manifest.get('human_root'))
                trace=NativeTrace(env,args.robot,selected) if args.suite=='flips' else None
                if trace:native_joint_order=trace.native_joint_names
                frame=run_benchmark(env,policy,meta.hist_len,verbose=True)
                physics=env.physics_diagnostics.report()
                if physics['overflow_bits']:raise RuntimeError('Native physics overflow: '+str(physics))
                if frame['name'].tolist()!=[r['name'] for r in selected]:raise ValueError('Full-clip membership differs')
                records=json_records(frame)
                for row,record in zip(selected,records):
                    record.update(source_id=row['source_id'],split=row['split'],duration_seconds=(row['frames']-1)*.02,
                        played_seconds=int(record['steps'])*.02,full_clip_completed=bool(not record['failed'] and int(record['steps'])==row['frames']-1))
                    if trace:
                        with np.load(row['output'],allow_pickle=False) as motion:
                            q=motion['body_quat_w'][:,0]
                            angles=np.unwrap([pitch(x,q[0]) for x in q]);angle=float(angles[np.argmax(abs(angles-angles[0]))]-angles[0])
                            height=standing_height
                        record['reference_rotation_excursion_rad']=angle
                        record['approved_neutral_standing_height_m']=height
                        record.update(flip_metrics(trace.trace[row['name']],angle,height,failed=bool(record['failed'])))
                if trace:(args.output/f'{interface}-native-traces.json').write_text(json.dumps(trace.trace,allow_nan=False)+'\n')
            finally:env.close()
        frame.to_csv(args.output/f'{interface}-fullclips.csv',index=False)
        summaries.append(dict(interface=interface,human_points='five' if interface=='human' else None,kp_horizon=1,
            physics=physics,cases=records))
        del env,policy,frame,trace;gc.collect();torch.cuda.empty_cache()
        print(f'Completed full native {args.robot} {iteration} {interface} {args.suite}',flush=True)
    for path,digest in proof.items():
        if sha(path)!=digest:raise ValueError('Input changed during evaluation: '+path)
    report=dict(schema_version=1,metric_scope='full reference clips from frame zero to native termination or final frame',
        suite=args.suite,robot=args.robot,iteration=iteration,checkpoint=dict(path=str(args.ckpt),sha256=proof[str(args.ckpt.resolve())]),
        training_manifest_sha256=training_sha,evaluation_manifest_sha256=manifest_sha,inputs_sha256=proof,
        native_torque_joint_order=native_joint_order,native_torque_definition='Actual qfrc_actuator at native hinge DOFs, Nm' if args.suite=='flips' else None,
        settings=dict(policy='EMA',seed=0,control_graph=False,allow_tf32=False,human_points='five',kp_horizon=1),
        flip_criteria=dict(actual_rotation_degrees_min=330,actual_no_robot_floor_contact_seconds_min=.06,standing_height_reference='approved neutral pelvis height',stable_landing_seconds_min=.5,
            root_height_ratio_min=.8,root_up_min=.8,root_speed_max_m_s=.5,root_angular_speed_max_rad_s=1.5,joint_rmse_max_rad=.75,
            native_contact_distance_max_m=.002) if args.suite=='flips' else None,results=summaries)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output/'summary.json',flush=True)


if __name__=='__main__':main()
