"""Native Warp, reference-conditioned recovery from pinned low-pose windows.

Example: python -m scripts.evaluate_recovery --ckpt runs/run/q1/ckpt_6000.pt
 --parent-ckpt runs/parent/q1/ckpt_5000.pt --robot q1
 --subset-yaml prepared/allopt-20261112/q1/train.yaml
 --windows prepared/recovery-20261114/windows.json --output results/q1_6000.json
Success is a measured continuous standing hold, never reward or clip survival.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
import time

CATEGORIES = ('lying', 'seated', 'half_squat')
CRITERIA = dict(height_ratio_min=.85, pelvis_up_min=.8, root_speed_max_m_s=.35,
                root_angular_speed_max_rad_s=1., hold_seconds=1., timeout_seconds=10.)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_hashes(proof):
    for path, digest in proof.items():
        if Path(path).is_symlink() or file_hash(path) != digest:
            raise ValueError(f'Input SHA changed or is a symlink: {path}')


def validate_windows(document, robot, lengths, *, hold_seconds=1.):
    if (type(document.get('schema_version')) is not int or document['schema_version'] != 1
            or type(document.get('fps')) is not int or document['fps'] != 50):
        raise ValueError('Recovery windows require schema_version=1 and fps=50')
    limit = document.get("max_recovery_seconds", 10)
    if type(limit) not in (int,float) or not 10 <= limit <= 15:
        raise ValueError("Recovery timeout must be between 10 and 15 seconds")
    result = []; seen = set()
    for row in document.get('evaluation_cases', document.get('windows', [])):
        if row.get('robot') != robot: continue
        clip = row.get('clip')
        bounds = [row.get(key) for key in ('start_frame','stand_frame','hold_end_frame','stop_frame')]
        if (clip not in lengths or row.get('category') not in CATEGORIES or not row.get('source_id')
                or any(type(value) is not int for value in bounds)):
            raise ValueError('Invalid recovery category, clip, source or integer frame bounds')
        start, stand, hold_end, stop = bounds
        if not (0 <= start < stop <= stand < hold_end <= lengths[clip]) or hold_end - stand < math.ceil(hold_seconds*50):
            raise ValueError('Recovery bounds exceed clip or lack the required reference standing hold')
        if hold_end - start > limit*50:
            raise ValueError(f'Recovery standing hold must finish within {limit} seconds')
        key = (clip, start)
        if key in seen: raise ValueError('Duplicate recovery case')
        seen.add(key); result.append(dict(row))
    if not result: raise ValueError(f'No recovery cases for {robot}')
    return result


class StandingTracker:
    """Scalar metric state; inputs come only from post-step native robot tensors."""
    def __init__(self, standing_height, dt, *, criteria=None):
        if any(type(value) not in (float,int) or not math.isfinite(value) or value <= 0
               for value in (standing_height,dt)):
            raise ValueError('Standing height and timestep must be finite positive numbers')
        self.criteria = dict(CRITERIA, **(criteria or {}))
        self.height = float(standing_height); self.dt = float(dt)
        self.hold_steps = math.ceil(self.criteria['hold_seconds'] / dt - 1e-9)
        self.steps = self.run = 0; self.reason = None; self.confirmed = None
        self.first = None; self.last = None; self.pose_sum = 0.; self.samples = 0

    @staticmethod
    def valid(value):
        return all(type(v) in (int,float) and math.isfinite(v) for v in value.values())

    def upright(self, value):
        return value['height'] / self.height >= self.criteria['height_ratio_min'] and value['up'] >= self.criteria['pelvis_up_min']

    def initial(self, value):
        self.first = dict(value)
        if not self.valid(value): self.reason = 'nonfinite_state'
        elif self.upright(value): self.reason = 'initial_pose_already_standing'
        return self.reason

    def update(self, value, terminated=False):
        if self.reason is not None: return self.reason
        self.steps += 1
        if not self.valid(value):
            self.reason = 'nonfinite_state'; return self.reason
        self.last = dict(value); self.pose_sum += value['pose_error']; self.samples += 1
        if terminated: self.reason = 'termination'; return self.reason
        stable = (self.upright(value) and value['speed'] <= self.criteria['root_speed_max_m_s']
                  and value['angular_speed'] <= self.criteria['root_angular_speed_max_rad_s'])
        for key in ('upper_arm_deg','shoulder_hand_deg','elbow_bend_deg','wrist_root_height','arm_joint_rmse','body_speed'):
            if key+'_max' in self.criteria:
                stable = stable and value.get(key, float('inf')) <= self.criteria[key+'_max']
        if 'whole_body_height_min' in self.criteria:
            stable = stable and value.get('whole_body_height',0.) >= self.criteria['whole_body_height_min'] and value.get('torso_up',-1.) >= self.criteria['torso_up_min']
        self.run = self.run + 1 if stable else 0
        if self.run >= self.hold_steps and self.steps * self.dt <= self.criteria['timeout_seconds'] + 1e-9:
            self.reason = 'success'; self.confirmed = self.steps * self.dt
        return self.reason

    def result(self):
        def finite_fields(value):
            return {key:v if type(v) in (float,int) and math.isfinite(v) else None for key,v in (value or {}).items()}
        first = finite_fields(self.first)
        last = finite_fields(self.last or self.first)
        result = dict(success=self.reason == 'success', failure_reason=None if self.reason == 'success' else self.reason or 'timeout',
            time_to_stand_s=None if self.confirmed is None else (self.steps-self.hold_steps+1)*self.dt,
            stand_confirmed_s=self.confirmed, steps=self.steps, elapsed_s=self.steps*self.dt,
            reference_standing_height_m=self.height,
            initial_root_height_m=first.get('height'), initial_root_up=first.get('up'),
            initial_root_height_ratio=None if first.get('height') is None else first['height']/self.height,
            final_root_height_m=last.get('height'), final_root_height_ratio=None if last.get('height') is None else last['height']/self.height,
            final_root_up=last.get('up'), final_root_speed_m_s=last.get('speed'),
            final_root_angular_speed_rad_s=last.get('angular_speed'),
            mean_pose_error_rad=self.pose_sum/self.samples if self.samples else None)
        result.update({'final_'+key:last[key] for key in ('upper_arm_deg','shoulder_hand_deg','elbow_bend_deg','wrist_root_height','arm_joint_rmse','body_speed','whole_body_height','torso_up') if key in last})
        return result


def aggregate(rows):
    result = {}
    for category in CATEGORIES:
        group = [row for row in rows if row['category'] == category]
        successes = [row for row in group if row['success']]
        stats = dict(cases=len(group), successes=len(successes), success_rate=len(successes)/len(group) if group else None,
                     failures=dict(Counter(row['failure_reason'] for row in group if not row['success'])))
        for output, key, source in [('mean_time_to_stand_s','time_to_stand_s',successes),
                ('mean_final_root_height_ratio','final_root_height_ratio',group),
                ('mean_final_root_up','final_root_up',group),('mean_pose_error_rad','mean_pose_error_rad',group)]:
            values = [row[key] for row in source if row.get(key) is not None and math.isfinite(row[key])]
            stats[output] = statistics.mean(values) if values else None
        result[category] = stats
    return result


def prepare_cases(windows, subset, robot, interface, *, standing_target=None):
    import numpy as np
    from morphmimic.data.manifest import load_subset
    from morphmimic.robots import get_robot
    subset = Path(subset).resolve(); manifest_path = subset.parent.parent/'manifest.json'
    document = json.loads(Path(windows).read_text()); manifest = json.loads(manifest_path.read_text())
    proof = {str(Path(windows).resolve()):file_hash(windows), str(subset):file_hash(subset), str(manifest_path):file_hash(manifest_path)}
    if (subset.name != 'train.yaml' or not manifest.get('complete')
            or document.get('training_manifest_sha256') != proof[str(manifest_path)]
            or document.get('subset_sha256',{}).get(robot) != proof[str(subset)]):
        raise ValueError('Recovery training manifest/subset SHA identity differs')
    paths = {Path(path).stem:Path(path).resolve() for path in load_subset(str(subset))}
    lookup = {row['name']:row for row in manifest['clips'] if row['robot'] == robot and row['split'] == 'train'}
    lengths = {name:int(lookup[name]['frames']) for name in paths if name in lookup}
    cases = validate_windows(document,robot,lengths,hold_seconds=3. if standing_target else 1.); selected = []
    for case in cases:
        row = lookup[case['clip']]; path = paths[case['clip']]
        if case['source_id'] != row.get('source_id') or path != Path(row['output']).resolve():
            raise ValueError('Recovery source identity differs from prepared training source')
        proof[str(path)] = row['output_sha256']
        with np.load(path,allow_pickle=False) as motion:
            if int(motion['fps']) != 50 or len(motion['joint_pos']) != lengths[case['clip']]:
                raise ValueError('Recovery clip fps/length differs from manifest')
            pelvis = list(motion['body_names']).index(get_robot(robot).tracked_bodies[0])
            height = float(np.median(motion['body_pos_w'][case['stand_frame']:case['hold_end_frame'],pelvis,2]))
        case['reference_standing_height_m'] = float(standing_target['body_pos'][0,2]) if standing_target else height
        case['path'] = str(path)
        if interface == 'human':
            if not row.get('human_available'): continue
            proof[row['human_sidecar']] = row['human_sha256']
        selected.append(case)
    if not selected: raise ValueError(f'No {interface} recovery cases for {robot}')
    if set(case['category'] for case in selected) != set(CATEGORIES):
        raise ValueError(f'{interface} recovery cases must cover every category for {robot}')
    verify_hashes(proof)
    return selected, proof, manifest.get('human_root')


def checkpoint_identity(path, robot, training_sha):
    import torch
    path = Path(path)
    if path.is_symlink() or re.fullmatch(r'ckpt_\d+\.pt',path.name) is None:
        raise ValueError('Recovery requires a fixed ckpt_<iteration>.pt, never latest or a symlink')
    digest = file_hash(path); saved = torch.load(path,map_location='cpu',weights_only=False)
    if (saved['cfg'].get('robot','g1') != robot or saved.get('iteration') != int(path.stem.split('_')[1])
            or saved.get('optimized_training',{}).get('prepared_manifest_sha256') != training_sha):
        raise ValueError('Checkpoint robot, iteration or training manifest SHA differs')
    verify_hashes({str(path):digest})
    return dict(path=str(path.resolve()),sha256=digest,iteration=saved['iteration']), saved



def validate_dataset_transition(receipt, saved, training_sha, proof, *, robot=None, parent_iteration=5000):
    """A different parent dataset is allowed only with the checkpoint's pinned receipt."""
    path = Path(receipt).resolve(); digest = file_hash(path)
    entry = saved.get('dataset_transition', {})
    baseline = saved.get('iteration') == parent_iteration
    if not baseline and (entry.get('sha256') != digest or Path(entry.get('path', '')).resolve() != path):
        raise ValueError('Checkpoint dataset transition receipt differs')
    document = json.loads(path.read_text())
    if document.get('schema_version') != 1 or document.get('manifest',{}).get('sha256') != training_sha:
        raise ValueError('Dataset transition does not identify the evaluated training manifest')
    if baseline and saved.get('optimized_training',{}).get('prepared_manifest_sha256') != document['parent_manifest']['sha256']:
        raise ValueError('Baseline checkpoint does not identify the transition parent manifest')
    if robot is not None:
        from morphmimic.train.dataset_transition import verify_append_transition
        verify_append_transition(path,digest,document['manifest']['path'],robot)
    for key in ('parent_manifest','manifest'):
        row = document[key]; proof[str(Path(row['path']).resolve())] = row['sha256']
    proof[str(path)] = digest; verify_hashes(proof)
    return document['parent_manifest']['sha256']


def validate_checkpoint_ancestry(candidate, saved, parent, parent_saved, *, parent_iteration=5000):
    if any(checkpoint.get('cfg',{}).get('optimizer') != 'muon' for checkpoint in (saved,parent_saved)):
        raise ValueError('Recovery candidate and parent must both use Muon')
    if parent['iteration'] != parent_iteration or candidate['iteration'] < parent_iteration:
        raise ValueError(f'Recovery parent must be the optimized checkpoint {parent_iteration}')
    parent_sha = candidate['sha256'] if candidate['iteration'] == parent_iteration else saved.get('finetuned_from',{}).get('sha256')
    if parent_sha != parent['sha256']:
        raise ValueError('Recovery parent SHA differs from candidate checkpoint ancestry')


def run_recovery(env, policy, hist_len, cases, *, standing_target=None, criteria=None):
    import torch
    from morphmimic.constants import PROPRIO_DIM
    from morphmimic.train.obs_history import ObsHistory
    command = env.command_manager.get_term('motion'); lib = command.library
    if command.cfg.sampling != 'start' or env.cfg.auto_reset:
        raise ValueError('Recovery evaluation requires explicit eval resets')
    if abs(env.step_dt-.02) > 1e-8: raise ValueError('Recovery requires 50 Hz physics control')
    indices = {name:i for i,name in enumerate(lib.clip_names)}
    history = ObsHistory(env.num_envs,hist_len,PROPRIO_DIM,env.device)
    active = {}; trackers = {}; results = [None]*len(cases); cursor = 0; goals = {}

    def measurements(reference_frame=None):
        pos = command.robot_body_pos_w[:,0] - env.scene.env_origins
        quat = command.robot_body_quat_w[:,0]
        up = 1 - 2*(quat[:,1].square()+quat[:,2].square())
        speed = command.robot_body_lin_vel_w[:,0].norm(dim=-1)
        angular = command.robot_body_ang_vel_w[:,0].norm(dim=-1)
        pose_error = (command.robot_joint_pos-lib.joint_pos[command.global_frame if reference_frame is None else reference_frame].float()).square().mean(-1).sqrt()
        values = torch.stack((pos[:,2],up,speed,angular,pose_error),-1).cpu().tolist()
        result = [dict(zip(('height','up','speed','angular_speed','pose_error'),row)) for row in values]
        if standing_target is not None:
            from morphmimic.standing import arm_geometry, native_arm_rmse
            bodies = command.robot_body_pos_w
            extra = arm_geometry(bodies)
            extra['whole_body_height'] = bodies[:,:,2].amax(-1)-bodies[:,[3,6],2].amin(-1)
            torso = command.robot_body_quat_w[:,7]
            extra['torso_up'] = 1-2*(torso[:,1].square()+torso[:,2].square())
            extra['body_speed'] = command.robot_body_lin_vel_w.norm(dim=-1).amax(-1)
            extra['arm_joint_rmse'] = native_arm_rmse(command.robot.data.joint_pos,command.robot.joint_names,standing_target)
            for key, metric in extra.items():
                for row, value in zip(result, metric.cpu().tolist()): row[key] = value
        return result

    def start(ids):
        nonlocal cursor, goals
        assigned = []
        for env_id in ids.tolist():
            index = min(cursor,len(cases)-1); cursor += 1; case = cases[index]
            active[env_id] = index if cursor <= len(cases) else None
            assigned.append(case)
        command.assign_clips(ids,torch.tensor([indices[c['clip']] for c in assigned],device=env.device,dtype=torch.long),
                             start_frames=torch.tensor([c['start_frame'] for c in assigned],device=env.device,dtype=torch.long))
        obs,_ = env.reset(env_ids=ids)
        if env.physics_diagnostics.report()['overflow_bits']: raise RuntimeError('Native physics overflow during recovery reset')
        history.reset(ids,obs['policy'][ids])
        if not goals: goals = {key:obs[key].clone() for key in ('goal','goal_kp')}
        else:
            for key in ('goal','goal_kp'): goals[key][ids] = obs[key][ids]
        values = measurements()
        for env_id in ids.tolist():
            if active[env_id] is None: continue
            tracker = StandingTracker(cases[active[env_id]]['reference_standing_height_m'],env.step_dt,criteria=criteria)
            tracker.initial(values[env_id]); trackers[env_id] = tracker

    with torch.no_grad():
        start(torch.arange(env.num_envs,device=env.device))
        completed = 0
        while completed < len(cases):
            if lib.has_human: goals['human_joint_valid'] = lib.human_valid_at(command.global_frame)
            action = policy(history.flat(),goals)
            if not bool(torch.isfinite(action).all()): raise ValueError('Nonfinite recovery policy action')
            frame = command.global_frame.clone()
            obs,_,terminated,truncated,_ = env.step(action)
            if env.physics_diagnostics.report()['overflow_bits']: raise RuntimeError('Native physics overflow during recovery step')
            history.push(obs['policy']); goals = {key:obs[key].clone() for key in ('goal','goal_kp')}
            values = measurements(frame); done = []
            term = terminated.cpu().tolist(); trunc = truncated.cpu().tolist(); frames = frame.cpu().tolist()
            clip_starts = lib.clip_start.cpu().tolist()
            for env_id in range(env.num_envs):
                index = active[env_id]
                if index is None:
                    if term[env_id] or trunc[env_id]: done.append(env_id)
                    continue
                case = cases[index]; tracker = trackers[env_id]
                reason = tracker.update(values[env_id],terminated=term[env_id])
                local = frames[env_id]-clip_starts[indices[case['clip']]]
                if reason is None and trunc[env_id]: reason = 'reference_exhausted'
                if reason is None and tracker.steps >= math.ceil(tracker.criteria['timeout_seconds']/env.step_dt): reason = 'timeout'
                if reason is not None:
                    tracker.reason = reason
                    results[index] = {key:value for key,value in case.items() if key != 'path'}
                    results[index].update(tracker.result(),final_reference_frame=local,native_overflow_bits=0)
                    completed += 1; done.append(env_id)
            if done and completed < len(cases): start(torch.tensor(done,device=env.device,dtype=torch.long))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ckpt','subset-yaml','windows','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--parent-ckpt',type=Path)
    parser.add_argument('--parent-iteration',type=int,default=5000,help='Explicit fixed ancestor for this continuation')
    parser.add_argument('--standing-target',type=Path,help='Hash-pinned approved native standing review receipt')
    parser.add_argument('--dataset-transition',type=Path,help='Explicit original-to-mixed-data training receipt')
    parser.add_argument('--robot',choices=('q1','bumi','g1'),required=True)
    parser.add_argument('--interface',choices=('motion','human'),default='motion')
    parser.add_argument('--num-envs',type=int,default=64); parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 256: parser.error('--num-envs must be in 1..256')
    if args.parent_iteration < 0: parser.error('--parent-iteration must be nonnegative')
    if args.output.exists() or args.output.with_suffix('.csv').exists(): raise FileExistsError(args.output)
    started = time.monotonic()
    target = None; criteria = dict(CRITERIA)
    if args.standing_target:
        from morphmimic.standing import load_standing_target, arm_geometry
        from morphmimic.robots import get_robot
        target = load_standing_target(args.standing_target,args.robot)
        criteria.update(hold_seconds=3.,timeout_seconds=15.,height_ratio_min=.95,pelvis_up_min=.9,
            torso_up_min=.9,whole_body_height_min=.95*float(target['body_pos'][:,2].max()-target['body_pos'][[3,6],2].min()),
            body_speed_max=.5*get_robot(args.robot).height_scale,upper_arm_deg_max=20.,shoulder_hand_deg_max=25.,
            elbow_bend_deg_max=35.,arm_joint_rmse_max=.25,
            wrist_root_height_max=float(arm_geometry(target['body_pos'])['wrist_root_height'])+.02*get_robot(args.robot).height_scale)
    cases,proof,human_root = prepare_cases(args.windows,args.subset_yaml,args.robot,args.interface,standing_target=target)
    proof[str(Path(__file__).resolve())] = file_hash(__file__)
    if target:
        import morphmimic.standing as standing_source
        proof[str(Path(standing_source.__file__).resolve())] = file_hash(standing_source.__file__)
        approved = json.loads(Path(args.windows).read_text()).get('approved_target',{}).get('review_manifest',{})
        if approved.get('sha256') != target['receipt_sha256']:
            raise ValueError('Recovery windows use another approved standing target receipt')
        proof.update({target['receipt']:target['receipt_sha256'], target['source']:target['source_sha256']})
    manifest = args.subset_yaml.resolve().parent.parent/'manifest.json'; training_sha = proof[str(manifest)]
    candidate_training_sha = training_sha
    if args.dataset_transition and args.ckpt.name == f'ckpt_{args.parent_iteration}.pt':
        candidate_training_sha = json.loads(args.dataset_transition.read_text())['parent_manifest']['sha256']
    identity,saved = checkpoint_identity(args.ckpt,args.robot,candidate_training_sha)
    parent_path = args.parent_ckpt or (args.ckpt if identity['iteration'] == args.parent_iteration else args.ckpt.parent/f'ckpt_{args.parent_iteration}.pt')
    parent_training_sha = validate_dataset_transition(args.dataset_transition,saved,training_sha,proof,robot=args.robot,parent_iteration=args.parent_iteration) if args.dataset_transition else training_sha
    parent,parent_saved = checkpoint_identity(parent_path,args.robot,parent_training_sha)
    validate_checkpoint_ancestry(identity,saved,parent,parent_saved,parent_iteration=args.parent_iteration)
    proof[identity['path']] = identity['sha256']; proof[parent['path']] = parent['sha256']
    from morphmimic.scripts.gpu_guard import select_gpu
    select_gpu(args.gpu)
    from mjlab.utils.torch import configure_torch_backends
    from morphmimic.eval.benchmark import make_eval_env, require_human_sidecars
    from morphmimic.eval.policy_loader import load_policy
    configure_torch_backends(allow_tf32=False)
    goal_anchor = saved['cfg'].get('goal_anchor','reference')
    policy,meta = load_policy(args.ckpt,'cuda:0',interface=args.interface,human_points='five',kp_horizon=1,goal_anchor=goal_anchor)
    if meta.robot != args.robot or meta.iteration != identity['iteration']: raise ValueError('Loaded policy identity differs')
    with tempfile.TemporaryDirectory(prefix='recovery-eval-') as temporary:
        subset = Path(temporary)/'subset.yaml'
        subset.write_text(json.dumps(dict(root='/',clips=list(dict.fromkeys(case['path'] for case in cases)))))
        env = make_eval_env(str(subset),min(args.num_envs,len(cases)),'cuda:0',
            library_dtype=saved['cfg'].get('library_dtype','float32'),seed=args.seed,robot=args.robot,
            human_root=human_root,goal_anchor=goal_anchor,allow_missing_human=args.interface=='motion')
        try:
            if args.interface == 'human': require_human_sidecars(env,str(subset),human_root)
            rows = run_recovery(env,policy,meta.hist_len,cases,standing_target=target,criteria=criteria)
            physics = env.physics_diagnostics.report()
            if physics['overflow_bits']: raise RuntimeError(f'Native recovery physics overflow: {physics}')
        finally: env.close()
    verify_hashes(proof)
    document = dict(schema_version=1,metric='native_neutral_standing_recovery_v2' if target else 'native_reference_conditioned_recovery_v1',
        scope='Bounded prepared reference poses, deterministic nominal physics; not arbitrary autonomous get-up.',
        window_definition='start_frame:stop_frame is the reset interval; evaluation starts at start_frame and runs to the configured timeout or clip exhaustion',
        created_utc=datetime.now(timezone.utc).isoformat(),robot=args.robot,interface=args.interface,
        human_points='five' if args.interface=='human' else None,kp_horizon=1,seed=args.seed,
        checkpoint=identity,parent_checkpoint=parent,criteria=criteria,floor_height_m=0.,height_definition='actual tracked pelvis z minus environment origin z, flat floor at zero',training_manifest_sha256=training_sha,
        input_sha256=proof,physics=physics,training_physics=saved.get('physics'),
        actual_num_envs=min(args.num_envs,len(cases)),elapsed_seconds=time.monotonic()-started,
        categories=aggregate(rows),cases=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document,indent=2,allow_nan=False)); temporary.replace(args.output)
    with args.output.with_suffix('.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(output=str(args.output),categories=document['categories']),indent=2))


if __name__ == '__main__': main()
