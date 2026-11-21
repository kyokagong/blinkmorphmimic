"""CPU delivery audit and observed curves for the fixed 6000→7000 continuation.

Missing checkpoints remain missing. Human coverage counts successful human training
starts, not human-policy simulation success. No curve interpolates missing observations.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from scripts.audit_optimized import _fixed_checkpoint, finite_tree, require_adam, require_muon
from scripts.plot_optimized import normalize_scalar_events, series
from scripts.prepare_optimized import ROBOTS, sha
from morphmimic.train.importance_sampling import create_state, source_balanced_clip_weights, validate_state

ITERATIONS = (6250, 6500, 6750, 7000)
STAGES = ('collected', 'world_model', 'policy', 'human')


def optimizer_history(saved, previous, name, phase_steps):
    """Inspect the existing three Muon+Adam optimizers and two Adam critics."""
    finite_tree(saved, name)
    mixed = name in ('wm', 'policy', 'kp')
    if mixed:
        if set(saved) != {'muon', 'adam'} or set(previous) != {'muon', 'adam'}:
            raise ValueError(f'{name}: expected inherited Muon+Adam history')
        named = [(str(i), saved['muon']['state'][i]['momentum_buffer'])
                 for g in saved['muon']['param_groups'] for i in g['params']]
        proof = require_muon(saved, named, phase_steps, name)
        for group, old in zip(saved['muon']['param_groups'], previous['muon']['param_groups'], strict=True):
            if group['params'] != old['params']:
                raise ValueError(f'{name}: Muon parameter inventory changed')
        for key, old in previous['muon']['state'].items():
            if saved['muon']['state'][key]['momentum_buffer'].shape != old['momentum_buffer'].shape:
                raise ValueError(f'{name}: Muon momentum shape changed')
        saved, previous = saved['adam'], previous['adam']
    else:
        proof = {}
    ids = [i for g in saved['param_groups'] for i in g['params']]
    if ids != [i for g in previous['param_groups'] for i in g['params']]:
        raise ValueError(f'{name}: Adam parameter inventory changed')
    states, old_states = saved['state'], previous['state']
    if set(states) != set(old_states) or not states:
        raise ValueError(f'{name}: Adam history inventory changed')
    values = [int(v['step']) for v in states.values()]
    named = [(str(i), states[i]['exp_avg']) for i in ids]
    proof['adam'] = require_adam(saved, named, values[0], name)
    increases = []
    for key, old in old_states.items():
        if states[key]['exp_avg'].shape != old['exp_avg'].shape or int(states[key]['step']) < int(old['step']):
            raise ValueError(f'{name}: Adam history reset')
        increases.append(int(states[key]['step']) > int(old['step']))
    if not all(increases):
        raise ValueError(f'{name}: Adam update counter did not increase')
    return proof


def exact_update_deltas(checkpoint, parent, proofs):
    from morphmimic.train.imagination import ImaginationCfg
    elapsed = checkpoint['iteration'] - parent['iteration']
    batches = checkpoint['cfg']['wm_batches']
    if type(batches) is not int or batches < 1:
        raise ValueError('Invalid world-model updates per iteration')
    expected = dict(world_model=elapsed*batches, motion=elapsed, human=elapsed)
    current, old = (value['optimization_state']['optimizer_steps'] for value in (checkpoint, parent))
    phase_delta = {key:current[key]-old[key] for key in expected}
    ema_delta = checkpoint['ema_updates']-parent['ema_updates']
    if phase_delta != expected or ema_delta != elapsed:
        raise ValueError('Exact optimizer/EMA iteration deltas differ from the 6000 parent')
    critic = ImaginationCfg()
    updates = critic.critic_epochs*critic.critic_minibatches
    expected_raw = dict(wm=expected['world_model'], policy=elapsed, kp=elapsed,
                        critic=elapsed*updates, critic_kp=elapsed*updates)
    raw_delta = {}
    for name, wanted in expected_raw.items():
        saved = parent['optim'][name]
        adam = saved['adam'] if name in ('wm','policy','kp') else saved
        steps = {int(value['step']) for value in adam['state'].values()}
        if len(steps) != 1:
            raise ValueError(f'{name}: parent Adam steps differ')
        raw_delta[name] = proofs[name]['adam']['state_steps']-next(iter(steps))
        if raw_delta[name] != wanted:
            raise ValueError(f'{name}: exact Adam update delta differs')
    physics = checkpoint.get('physics', {})
    if (type(physics.get('overflow_bits')) is not int or physics['overflow_bits'] != 0
            or type(physics.get('checked_physics_steps')) is not int or physics['checked_physics_steps'] < 1
            or physics.get('flagged_step_world_observations', 0) != 0
            or physics.get('flagged_forward_world_observations', 0) != 0):
        raise ValueError('Checkpoint physics overflow or missing sampled diagnostics')
    return dict(phase_step_deltas=phase_delta, optimizer_step_deltas=raw_delta,
                ema_update_delta=ema_delta, physics=physics)


def ledger(checkpoint):
    coverage, human = checkpoint['coverage'], checkpoint['human_coverage']
    names = coverage['clip_names']; n = len(names)
    if human['clip_names'] != names or human['since_iteration'] != 5000:
        raise ValueError('Human ledger must preserve inventory and epoch 5000')
    total = {key: torch.as_tensor(coverage['counts'][key]) for key in STAGES[:-1]}
    total['human'] = human['counts']
    source = {key: torch.as_tensor(coverage['source']['counts'][key]) for key in STAGES[:-1]}
    source['human'] = human['source_counts']
    for stage in STAGES:
        for value in (total[stage], source[stage]):
            if value.dtype != torch.long or value.shape != (n,) or bool((value < 0).any()):
                raise ValueError(f'Invalid {stage} coverage counters')
        if bool((source[stage] > total[stage]).any()):
            raise ValueError(f'{stage}: source coverage exceeds total')
    return names, total, source


def clip_coverage_summary(records):
    """Keep individual gaps visible even when the dataset aggregate is positive."""
    def gaps(rows, key):
        return {stage:[row['clip'] for row in rows if row[key][stage] == 0
                       and (stage != 'human' or row['human_available'])] for stage in STAGES}
    flips = [row for row in records if row['dataset'] == 'seed_flips']
    used = {stage:bool(flips) and all(row['delta'][stage] > 0 for row in flips) for stage in STAGES}
    source_used = {stage:bool(flips) and all(row['source_delta'][stage] > 0 for row in flips) for stage in STAGES}
    complete = all(used.values()) and all(source_used.values())
    return dict(zero_coverage=gaps(records, 'delta'), zero_source_coverage=gaps(records, 'source_delta'),
        human_not_applicable=[row['clip'] for row in records if not row['human_available']],
        flip_coverage=dict(train_clip_count=len(flips), clips=flips,
            all_clips_used_by_stage=used, all_clips_source_used_by_stage=source_used,
            zero_by_stage=gaps(flips, 'delta'), zero_source_by_stage=gaps(flips, 'source_delta'),
            all_stages_all_clips_used=complete,
            conclusion=(f'All {len(flips)} train flips have positive source-body usage in all four stages.' if complete
                        else 'Some train flips still lack actual source-body usage in one or more stages.')))


def audit_snapshot(checkpoint, parent, previous, manifest, robot, iteration, parent_path, parent_sha):
    cfg = checkpoint['cfg']
    if (checkpoint['iteration'] != iteration or cfg.get('robot') != robot or cfg.get('num_envs') != 2048
            or cfg.get('iterations') != 7000 or cfg.get('optimizer') != 'muon'
            or cfg.get('importance_sampling') is not True or not cfg.get('kp_interface')
            or cfg.get('policy_goal') != 'motion' or cfg.get('recovery_reset_share') != 0
            or cfg.get('failure_reset') != 0 or cfg.get('probe_focus_dir') is not None):
        raise ValueError('Expected 2048-env joint Muon continuation without fixed reset focus')
    lineage = checkpoint.get('finetuned_from', {})
    if (lineage.get('iteration') != 6000 or lineage.get('path') != str(Path(parent_path).resolve())
            or lineage.get('sha256') != parent_sha or lineage.get('ancestor') != parent.get('finetuned_from')
            or lineage.get('ancestor', {}).get('iteration') != 5000
            or lineage.get('optimizer_state_restored') is not True or lineage.get('ema_state_restored') is not True):
        raise ValueError('Immediate 6000 parent SHA or 5000 ancestor differs')
    previous = parent if previous is None else previous
    steps = checkpoint['optimization_state']['optimizer_steps']
    if (steps != checkpoint['optimized_training']['optimizer_steps']
            or set(steps) != {'world_model', 'motion', 'human'}
            or any(type(v) is not int or v <= previous['optimization_state']['optimizer_steps'][k] for k, v in steps.items())
            or type(checkpoint['ema_updates']) is not int or checkpoint['ema_updates'] != steps['motion']
            or checkpoint['ema_updates'] <= previous['ema_updates']):
        raise ValueError('Optimizer or EMA counters did not increase')
    finite_tree(checkpoint['ema_policy'], 'EMA')
    if not checkpoint['ema_policy'] or checkpoint['ema_policy'].keys() != parent['ema_policy'].keys():
        raise ValueError('EMA inventory changed')
    proofs = {}
    for name, phase in (('wm','world_model'), ('policy','motion'), ('kp','human'), ('critic','motion'), ('critic_kp','human')):
        proofs[name] = optimizer_history(checkpoint['optim'][name], previous['optim'][name], name, steps[phase])
    names, totals, sources = ledger(checkpoint)
    old_names, old_totals, old_sources = ledger(parent)
    before_names, before_totals, before_sources = ledger(previous)
    if (names[:len(old_names)] != old_names or names[:len(before_names)] != before_names
            or (previous['iteration'] > 6000 and names != before_names) or len(set(names)) != len(names)):
        raise ValueError('Coverage inventory is not append-only')
    for key in ('since_iteration', 'start', 'stop', 'has_phases'):
        current, old = checkpoint['coverage']['source'][key], parent['coverage']['source'][key]
        if (current != old if key == 'since_iteration' else current[:len(old_names)] != old):
            raise ValueError('Original source windows or coverage epoch changed')
    for stage in STAGES:
        if (bool((totals[stage][:len(before_names)] < before_totals[stage]).any())
                or bool((sources[stage][:len(before_names)] < before_sources[stage]).any())):
            raise ValueError(f'{stage}: coverage rolled back')
    rows_by_name = {row['name']: row for row in manifest['clips'] if row['robot'] == robot and row['split'] == 'train'}
    rows = [rows_by_name[name] for name in names]
    if len(names) <= len(old_names) or any(row['dataset'] != 'seed_flips' for row in rows[len(old_names):]):
        raise ValueError('Expected actual appended seed_flips clips')
    bins, starts, offset = [], [], 0
    for index, row in enumerate(rows):
        full, tail = divmod(row['frames'], 50)
        count = full + int(tail >= 25 or (tail > 0 and full == 0))
        bins.extend([index] * count); starts.extend(offset + 50*j for j in range(count)); offset += row['frames']
    weights, group_names, clip_groups = source_balanced_clip_weights(names, manifest['clips'], robot)
    initial = create_state(SimpleNamespace(num_clips=len(names), bin_clip=torch.tensor(bins), bin_start=torch.tensor(starts),
        importance_clip_weights=weights, importance_group_names=group_names, importance_clip_groups=clip_groups))
    importance = checkpoint['importance_sampling']
    if importance['since_iteration'] != 6000:
        raise ValueError('Importance sampler must preserve epoch 6000')
    validate_state(initial, importance['state'])
    state = importance['state']; p = state['probabilities']; clip = state['bin_clip']
    if not bool(((p >= .3*initial['base']*(1-1e-5)) & (p <= 8*initial['base']*(1+1e-5))).all()):
        raise ValueError('Importance probability violates relative exploration floor/cap')
    previous_state = previous.get('importance_sampling', {}).get('state', initial)
    for key in ('visits', 'episodes', 'failures'):
        if bool((state[key] < previous_state[key]).any()):
            raise ValueError(f'Importance sampler {key} rolled back')
    if not bool((state['visits'] > 0).any()) or not bool((state['episodes'] > 0).any()):
        raise ValueError('Sampler has no real visits or rollout outcomes')
    clip_mass = torch.zeros(len(names)).index_add_(0, clip, p)
    base_mass = torch.zeros(len(names)).index_add_(0, clip, initial['base'])
    previous_mass = torch.zeros(len(names)).index_add_(0, clip, previous_state['probabilities'])
    clip_visits = torch.zeros(len(names), dtype=torch.long).index_add_(0, clip, state['visits'])
    records = []
    for index, row in enumerate(rows):
        old = index < len(old_names)
        records.append(dict(clip=names[index], dataset=row['dataset'], source=row['source_id'],
            inventory='original' if old else 'appended_flip', human_available=row.get('human_available', False),
            base_mass=float(base_mass[index]), probability_mass=float(clip_mass[index]),
            previous_probability_mass=float(previous_mass[index]),
            probability_mass_change=float(clip_mass[index]-previous_mass[index]),
            probability_mass_change_from_base=float(clip_mass[index]-base_mass[index]), visits=int(clip_visits[index]),
            delta={stage:int(totals[stage][index] - (old_totals[stage][index] if old else 0)) for stage in STAGES},
            source_delta={stage:int(sources[stage][index] - (old_sources[stage][index] if old else 0)) for stage in STAGES}))
    def aggregate(keys):
        grouped = {}
        for row in records:
            key = tuple(row[k] for k in keys)
            if key not in grouped:
                grouped[key] = dict(zip(keys, key), base_mass=0., probability_mass=0., previous_probability_mass=0.,
                    probability_mass_change=0., probability_mass_change_from_base=0., visits=0, clips=0,
                    delta={s:0 for s in STAGES}, source_delta={s:0 for s in STAGES}, human_available=False)
            value = grouped[key]; value['base_mass'] += row['base_mass']; value['probability_mass'] += row['probability_mass']; value['visits'] += row['visits']; value['clips'] += 1
            for metric in ('previous_probability_mass','probability_mass_change','probability_mass_change_from_base'):
                value[metric] += row[metric]
            value['human_available'] |= row['human_available']
            for stage in STAGES:
                value['delta'][stage] += row['delta'][stage]; value['source_delta'][stage] += row['source_delta'][stage]
        return list(grouped.values())
    datasets = aggregate(('dataset',))
    for row in datasets:
        required = STAGES if row['human_available'] else STAGES[:-1]
        if any(row['delta'][stage] <= 0 or row['source_delta'][stage] <= 0 for stage in required):
            raise ValueError(f"{row['dataset']}: missing actual dataset/source training usage")
    return dict(robot=robot, iteration=iteration, sampler_semantics_version=2, source_groups=list(group_names),
        source_balance='Equal dataset base mass; reported actual probability mass is adaptive, not a fixed dataset quota.', optimizer_steps=steps, optimizers=proofs,
        ema_updates=checkpoint['ema_updates'], human_since_iteration=5000, sampler_since_iteration=6000,
        probability_l1_from_base=float((p-initial['base']).abs().sum()),
        probability_l1_from_previous=float((p-previous_state['probabilities']).abs().sum()),
        entropy=float(-(p*p.clamp_min(1e-12).log()).sum()), max_base_ratio=float((p/initial['base']).max()),
        unseen_bins=torch.where(state['visits']==0)[0].tolist(),
        unseen_clips=[name for name, visits in zip(names, clip_visits) if visits == 0],
        clips=records, datasets=datasets, sources=aggregate(('dataset','source')),
        **clip_coverage_summary(records), **exact_update_deltas(checkpoint, parent, proofs))


def audit_run(run_root, parent_root):
    run_root, parent_root = Path(run_root), Path(parent_root)
    report = dict(schema_version=1, status='incomplete', expected_iterations=list(ITERATIONS), num_envs=2048,
        coverage_definition='Cumulative starts used since parent; source counts exclude synthetic blend/hold tails; human counts are successful training starts, not policy evaluation success.',
        missing=[], errors=[], checkpoints=[], parent_restorations=[])
    for robot in ROBOTS:
        parent_path = parent_root/robot/'ckpt_6000.pt'
        expected = [run_root/robot/f'ckpt_{i}.pt' for i in ITERATIONS]
        restoration_path = run_root/robot/'parent-restoration.json'
        report['missing'].extend(str(p) for p in [parent_path, restoration_path, *expected] if not p.exists())
        if not parent_path.exists():
            continue
        try:
            if restoration_path.exists():
                restoration = json.loads(restoration_path.read_text())
                flags = ('pass_check','optimizer_states_exact','models_exact','ema_exact',
                         'human_coverage_prefix_exact','forced_recovery_disabled')
                if (restoration_path.is_symlink() or restoration.get('iteration') != 6000
                        or restoration.get('importance_epoch') != 6000
                        or any(restoration.get(key) is not True for key in flags)):
                    raise ValueError('Pre-update parent restoration receipt failed')
                report['parent_restorations'].append(dict(robot=robot,path=str(restoration_path.resolve()),
                    sha256=sha(restoration_path), **restoration))
            parent, parent_sha = _fixed_checkpoint(parent_path)
            if parent['iteration'] != 6000 or parent['cfg']['robot'] != robot:
                raise ValueError('Parent identity differs')
            previous = None
            for iteration, path in zip(ITERATIONS, expected):
                if not path.exists():
                    continue
                checkpoint, digest = _fixed_checkpoint(path)
                manifest_path = Path(checkpoint['cfg']['prepared_manifest'])
                if sha(manifest_path) != checkpoint['optimized_training']['prepared_manifest_sha256']:
                    raise ValueError('Training manifest SHA differs')
                manifest = json.loads(manifest_path.read_text())
                if previous is not None and any(checkpoint['cfg'].get(key) != previous['cfg'].get(key)
                        for key in ('prepared_manifest','dataset_transition_sha256','freeze_receipt')):
                    raise ValueError('Frozen dataset/run identity changed between checkpoints')
                row = audit_snapshot(checkpoint, parent, previous, manifest, robot, iteration, parent_path, parent_sha)
                row.update(path=str(path.resolve()), sha256=digest, manifest_sha256=sha(manifest_path))
                report['checkpoints'].append(row); previous = checkpoint
                if iteration == 7000 and not row['flip_coverage']['all_stages_all_clips_used']:
                    report['errors'].append(dict(robot=robot, error='Final checkpoint has train flips with zero stage/source-body coverage',
                        zero_source_by_stage=row['flip_coverage']['zero_source_by_stage']))
            selected = [r for r in report['checkpoints'] if r['robot'] == robot]
            if selected and not any(r['probability_l1_from_base'] > 1e-7 for r in selected):
                raise ValueError('Sampler probabilities never differed from the initial distribution')
            if len(selected) > 1 and not any(r['probability_l1_from_previous'] > 1e-7 for r in selected[1:]):
                raise ValueError('Sampler probabilities did not change between saved checkpoints')
        except (ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
            report['errors'].append(dict(robot=robot, error=str(error)))
    report['status'] = 'error' if report['errors'] else ('incomplete' if report['missing'] else 'ok')
    return report


def read_training_csv(path):
    rows = []
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if not {'robot','tag','iteration','value'} <= row.keys():
                raise ValueError('Training CSV requires robot,tag,iteration,value columns')
            value = float(row['value']) if row['value'] else math.nan
            rows.append(dict(robot=row['robot'], tag=row['tag'], iteration=int(row['iteration']),
                value=value if math.isfinite(value) else None, status='ok' if math.isfinite(value) else 'nonfinite'))
    return rows


def training_rows(run_root, csv_paths):
    if csv_paths:
        return [row for path in csv_paths for row in read_training_csv(path)], [str(p) for p in csv_paths]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows, inputs = [], []
    for robot in ROBOTS:
        directory = Path(run_root)/robot
        events = list(directory.glob('events.out.tfevents.*'))
        if not events:
            continue
        accumulator = EventAccumulator(str(directory), size_guidance={'scalars':0}); accumulator.Reload()
        inputs.extend(str(p) for p in events)
        for tag in accumulator.Tags()['scalars']:
            rows.extend(dict(robot=robot, tag=tag, **point) for point in normalize_scalar_events(accumulator.Scalars(tag)))
    return rows, inputs


def invalid_training_observations(rows):
    return [row for row in rows if 6000 <= row['iteration'] <= 7000 and
            (row['status'] != 'ok' or ((row['tag'].endswith('/nonfinite_steps') or
                                      row['tag'] == 'physics/overflow_bits') and row['value'] != 0))]


def write_outputs(report, out, training):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    def dump(name, rows):
        (out/name).write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    dump('audit.json', report)
    with (out/'training-scalars.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('robot','tag','iteration','value','status')); writer.writeheader(); writer.writerows(training)
    flat = [dict(robot=r['robot'], iteration=r['iteration'], **d) for r in report['checkpoints'] for d in r['datasets']]
    dump('sampling-coverage.json', flat)
    if not training and not flat:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def plot(groups, name, ylabel):
        fig, ax = plt.subplots(figsize=(10,5))
        for label, points in groups.items():
            points = sorted(points, key=lambda p:p['iteration'])
            ax.plot(*series(points, 'value'), '.-', label=label)
        ax.set_xlabel('Completed iteration'); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out/name, dpi=160); plt.close(fig)
    tags = sorted({row['tag'] for row in training})
    dump('training-tags.json', dict(enumerate(tags)))
    for index, tag in enumerate(tags):
        groups = defaultdict(list)
        for row in training:
            if row['tag'] == tag and 6000 <= row['iteration'] <= 7000:
                groups[row['robot']].append(dict(row, value=row['value'] if row['value'] is not None else float('nan')))
        if groups:plot(groups, f'train_{index:03}.png', tag)
    for metric in ('entropy', 'probability_l1_from_base', 'probability_l1_from_previous', 'unseen_bins', 'unseen_clips'):
        groups = defaultdict(list)
        for row in report['checkpoints']:
            value = len(row[metric]) if metric.startswith('unseen_') else row[metric]
            groups[row['robot']].append(dict(iteration=row['iteration'], status='ok', value=value))
        if groups:plot(groups, f'sampling_{metric}.png', metric)
    for metric in ('probability_mass', *STAGES):
        groups = defaultdict(list)
        for row in flat:
            groups[row['robot']+'/'+row['dataset']].append(dict(iteration=row['iteration'], status='ok',
                value=row['probability_mass'] if metric=='probability_mass' else row['source_delta'][metric]))
        if groups:plot(groups, f'{metric}.png', metric+' (source-body delta)' if metric!='probability_mass' else metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True); parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--training-csv', type=Path, nargs='+', help='Observed long-form robot,tag,iteration,value CSV; otherwise read real TensorBoard events')
    args = parser.parse_args()
    report = audit_run(args.run_root, args.parent_root)
    rows, inputs = training_rows(args.run_root, args.training_csv)
    report['training_inputs'] = inputs
    report['training_by_robot'] = {robot:dict(points=sum(r['robot']==robot and 6000 <= r['iteration'] <= 7000 for r in rows),
        nonfinite=sum(r['robot']==robot and r['status']!='ok' and 6000 <= r['iteration'] <= 7000 for r in rows)) for robot in ROBOTS}
    report['training_status'] = 'ok' if all(v['points'] for v in report['training_by_robot'].values()) else 'missing'
    if report['training_status'] != 'ok' and report['status']=='ok':report['status']='incomplete'
    report['training_invalid_observations'] = invalid_training_observations(rows)
    if report['training_invalid_observations']:
        report['errors'].append(dict(error='Observed nonfinite training values/updates or physics overflow',
            observations=len(report['training_invalid_observations'])))
        report['status'] = 'error'
    write_outputs(report, args.out, rows)
    print(json.dumps(dict(status=report['status'], checkpoints=len(report['checkpoints']), missing=report['missing'], errors=report['errors']), indent=2))
    if report['status'] != 'ok':raise SystemExit(1)


if __name__ == '__main__':
    main()
