"""Fixed-checkpoint MorphMimic policy + physics preview (one shared browser session).

Run --help without importing torch, MuJoCo, or mjlab. HTTP stays responsive while
the control worker loads; comparison rendering owns its separate thread.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import queue
import re
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

ROBOTS = ("q1", "bumi", "g1")
STATIC = Path(__file__).parent / "static"
STANDUP_POSES = {
    'lying': ('fallAndGetUp2_subject2', 650),
    'seated': ('fallAndGetUp2_subject2', 700),
    'half_squat': ('fallAndGetUp1_subject5', 8298),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedCheckpoint:
    path: Path
    iteration: int
    sha256: str

    @classmethod
    def pin(cls, path, robot):
        raw = Path(path).absolute()
        resolved = raw.resolve(strict=True)
        match = re.fullmatch(r"ckpt_(\d+)\.pt", resolved.name)
        if (raw != resolved or not resolved.is_file() or not match
                or resolved.parent.name != robot):
            raise ValueError(f"{robot}: require a regular, nonsymlink {robot}/ckpt_<iteration>.pt")
        return cls(resolved, int(match[1]), sha256(resolved))

    def verify(self):
        if self.path.is_symlink() or self.path.resolve() != self.path or sha256(self.path) != self.sha256:
            raise ValueError("Pinned checkpoint changed; restart with a reviewed fixed checkpoint")

    def public(self):
        return dict(path=str(self.path), iteration=self.iteration, sha256=self.sha256)


class Catalog:
    def __init__(self, root, checkpoints, *, interface='motion', reference_manifest=None):
        if interface not in ('motion','human'):
            raise ValueError('Unknown release interface')
        self.interface = interface
        self.recovery_windows = None
        self.flip_sources = None
        self.flip_motions = {}
        self.training_manifest = None
        self.training_manifest_sha256 = None
        self.reference_manifest = None
        self.reference_manifest_sha256 = None
        self.neutral_review = None
        self.root = Path(root).resolve(strict=True)
        if reference_manifest is not None:
            if not isinstance(reference_manifest, dict) or set(reference_manifest) != {'path', 'sha256'}:
                raise ValueError('Invalid reference manifest')
            manifest = Path(reference_manifest['path']).absolute()
            if (manifest != self.root / 'manifest.json' or manifest != manifest.resolve(strict=True)
                    or sha256(manifest) != reference_manifest['sha256']):
                raise ValueError('Reference manifest SHA or fixed root differs')
            self.reference_manifest = manifest
            self.reference_manifest_sha256 = reference_manifest['sha256']
        self.checkpoints = checkpoints
        self.motions = {}
        for robot in ROBOTS:
            folder = self.root / robot
            self.motions[robot] = {
                str(path.relative_to(folder).with_suffix("")): path.resolve()
                for path in sorted(folder.rglob("*.npz"))
                if path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink()
            }
        manifest = self.root / "manifest.json"
        self.manifest_sha256 = sha256(manifest) if manifest.is_file() else None
        document = json.loads(manifest.read_text()) if manifest.is_file() else {}
        records = document.get("clips", [])
        self.records = {(row.get('robot'), row.get('name')): row for row in records}
        if self.reference_manifest is not None:
            from preview.reference import neutral_contract, reference_path, review_document
            self.neutral_review = document.get("neutral_review")
            review_document(self.root, self.neutral_review)
            human_root = self.root / document.get('human_root', 'human50')
            if (not human_root.is_dir() or human_root != human_root.resolve(strict=True)
                    or not human_root.is_relative_to(self.root)):
                raise ValueError('Migrated human root must stay inside its fixed reference root')
            if len(self.records) != len(records) or not records:
                raise ValueError('Empty or duplicate reference inventory')
            for (robot, name), record in self.records.items():
                if robot not in ROBOTS or self.motions[robot].get(name) != reference_path(self.root, record['output']):
                    raise ValueError('Motion outside its fixed reference root or inventory')
                neutral_contract(record)
                if record.get('human_available'):
                    if reference_path(self.root, record['human_sidecar']) != human_root / robot / (name + '.npz'):
                        raise ValueError('Migrated human sidecar differs from the loader human root')
            for robot in ROBOTS:
                self.motions[robot] = {name: path for name, path in self.motions[robot].items()
                    if (robot, name) in self.records
                    and (interface != 'human' or self.records[robot, name].get('human_available'))}
        self.motion_hashes={(row.get('robot'),row.get('name')):row.get('output_sha256') for row in records}
        sources = {(row.get("robot"), row.get("name")): row.get("source_id") for row in records}
        common = set.intersection(*(set(self.motions[r]) for r in ROBOTS))
        self.common = {
            name for name in common
            if all(r in checkpoints for r in ROBOTS)
            and sources.get((ROBOTS[0], name))
            and len({sources.get((r, name)) for r in ROBOTS}) == 1
        }

    @property
    def reference_path(self):
        return self.reference_manifest or self.training_manifest

    @property
    def human_root(self):
        manifest = self.reference_path
        value = json.loads(manifest.read_text()).get('human_root', str(self.root / 'human50')) if manifest else str(self.root / 'human50')
        return str(self.root / value) if self.reference_manifest is not None else value

    def verify_reference_manifest(self):
        if self.reference_manifest is not None and (self.reference_manifest != self.reference_manifest.resolve(strict=True)
                or sha256(self.reference_manifest) != self.reference_manifest_sha256):
            raise ValueError('Reference manifest SHA differs')

    def select(self, robot, motion):
        self.verify_reference_manifest()
        if self.training_manifest is not None and (self.training_manifest.is_symlink()
                or sha256(self.training_manifest) != self.training_manifest_sha256):
            raise ValueError('Joint training manifest SHA differs')
        if robot == "all":
            if not all(r in self.checkpoints for r in ROBOTS):
                raise ValueError("All three fixed checkpoints are required for comparison")
            if not isinstance(motion, str) or motion not in self.common:
                raise ValueError("Select a common source motion from the comparison catalog")
            return {r: self.select(r, motion) for r in ROBOTS}
        if robot not in ROBOTS:
            raise ValueError("Unknown robot")
        if robot not in self.checkpoints:
            raise ValueError(f"No fixed trained checkpoint configured for {robot}; playback refused")
        if not isinstance(motion, str) or motion not in self.motions[robot]:
            raise ValueError("Select a motion from this robot's catalog")
        path = self.motions[robot][motion]
        if not path.is_file() or (motion not in self.flip_motions and not path.resolve().is_relative_to(self.root / robot)):
            raise ValueError("Motion file is missing or outside its data root")
        expected=self.motion_hashes.get((robot,motion))
        if expected and sha256(path)!=expected:
            raise ValueError('Motion reference SHA differs from its pinned data manifest')
        if self.reference_manifest is not None:
            from preview.reference import verify_neutral_reference
            verify_neutral_reference(self.records[robot, motion], path, self.root, review=self.neutral_review, human=self.interface == 'human')
        if motion in self.flip_motions:
            entry = self.flip_sources
            if entry['path'].is_symlink() or sha256(entry['path']) != entry['sha256']:
                raise ValueError('Flip preview SHA differs')
            row = self.flip_motions[motion][robot]
            for proof in ('source', 'human_source') if self.interface == 'human' else ('source',):
                item = row[proof]; source = Path(item['path'])
                if source.is_symlink() or sha256(source) != item['sha256']:
                    raise ValueError('Flip original source SHA differs')
            if self.interface == 'human':
                item = row['human_sidecar']
                if Path(item['path']).is_symlink() or sha256(item['path']) != item['sha256']:
                    raise ValueError('Flip authentic human sidecar SHA differs')
        elif self.interface=='human' and self.reference_manifest is None:
            from morphmimic.data.human_prepare import verify_prepared_pairs
            verify_prepared_pairs(self.root,robot=robot,names=[motion+'.npz'],expected_sha256=self.manifest_sha256)
        return path

    def set_flip_sources(self, entry):
        if not isinstance(entry, dict) or set(entry) != {'path','sha256'} or self.training_manifest is None:
            raise ValueError('Flip preview requires pinned sources and training manifest')
        path = Path(entry['path']).absolute()
        if path != path.resolve(strict=True) or sha256(path) != entry['sha256']:
            raise ValueError('Flip preview SHA differs')
        document = json.loads(path.read_text())
        if (document.get('schema_version') != 1 or document.get('fps') != 50
                or document.get('training_manifest_sha256') != self.training_manifest_sha256):
            raise ValueError('Flip preview training identity differs')
        self.verify_reference_manifest()
        if self.reference_manifest is not None and document.get('reference_manifest_sha256') != self.reference_manifest_sha256:
            raise ValueError('Flip preview reference manifest SHA differs')
        manifest = json.loads(self.reference_path.read_text())
        records = {(r['robot'],r['name']):r for r in manifest['clips']}
        grouped = {}
        for row in document.get('clips', []):
            robot, name = row['robot'], row['clip']
            record = records.get((robot,name))
            if (robot not in ROBOTS or not record or record['source_id'] != row['source_id']
                    or record['split'] != row['split'] or record['frames'] != row['frames']
                    or (self.reference_manifest is None and name in self.motions[robot]) or robot in grouped.get(name, {})):
                raise ValueError('Flip preview source differs from training manifest')
            output = self.root / record['output'] if self.reference_manifest is not None else Path(record['output']).absolute()
            if output != output.resolve(strict=True) or sha256(output) != record['output_sha256']:
                raise ValueError('Flip reference SHA differs')
            row = dict(row)
            if self.interface == 'human':
                if not record.get('human_available') or not row.get('human_source'):
                    raise ValueError('Flip preview requires authentic human sources')
                row['human_sidecar'] = dict(path=str(self.root / record['human_sidecar']) if self.reference_manifest is not None else record['human_sidecar'],sha256=record['human_sha256'])
            grouped.setdefault(name, {})[robot] = row
        if not grouped or any(set(rows) != set(ROBOTS) or len({r['source_id'] for r in rows.values()}) != 1 for rows in grouped.values()):
            raise ValueError('Flip preview requires all three robots with common source identities')
        self.flip_sources = dict(path=path,sha256=entry['sha256'])
        self.flip_motions = grouped
        for name, rows in grouped.items():
            for robot in rows:
                record = records[robot,name]
                self.motions[robot][name] = (self.root / record['output']).resolve() if self.reference_manifest is not None else Path(record['output']).resolve()
                self.motion_hashes[robot,name] = record['output_sha256']
            self.common.add(name)

    def set_recovery_windows(self, entry):
        if not isinstance(entry,dict) or set(entry) != {'path','sha256'}:
            raise ValueError('Invalid recovery windows proof')
        path = Path(entry['path']).absolute()
        if path != path.resolve(strict=True) or sha256(path) != entry['sha256']:
            raise ValueError('Recovery windows SHA differs')
        if self.training_manifest is None:
            raise ValueError('Recovery windows require a pinned training manifest')
        document = json.loads(path.read_text())
        if document.get('training_manifest_sha256') != self.training_manifest_sha256:
            raise ValueError('Recovery windows training manifest SHA differs')
        self.verify_reference_manifest()
        if self.reference_manifest is not None and document.get('reference_manifest_sha256') != self.reference_manifest_sha256:
            raise ValueError('Recovery windows reference manifest SHA differs')
        self.recovery_windows = dict(path=path,sha256=entry['sha256'])

    def select_recovery(self, robot, pose):
        entry = self.recovery_windows
        if entry is None: raise ValueError('No reviewed neutral recovery references configured')
        if (entry['path'].is_symlink() or sha256(entry['path']) != entry['sha256']
                or self.training_manifest.is_symlink() or sha256(self.training_manifest) != self.training_manifest_sha256):
            raise ValueError('Recovery windows or training manifest SHA differs')
        self.verify_reference_manifest()
        document = json.loads(entry['path'].read_text())
        rows = {(row['robot'],row['name']):row for row in json.loads(self.reference_path.read_text())['clips']}
        result = {}
        for name in ROBOTS if robot == 'all' else (robot,):
            option = document.get('preview_poses',{}).get(pose,{}).get(name)
            if not option or (name,option['clip']) not in rows:
                raise ValueError('No reviewed neutral recovery pose for this robot')
            row = rows[name,option['clip']]; path = (self.root / row['output']) if self.reference_manifest is not None else Path(row['output']).absolute()
            if (path != path.resolve(strict=True) or sha256(path) != row['output_sha256']):
                raise ValueError('Neutral recovery reference SHA differs')
            start, hold = option['start_frame'],option['hold_frame']
            if type(start) is not int or type(hold) is not int or not 0 <= start < hold:
                raise ValueError('Invalid neutral recovery reference frame')
            if self.reference_manifest is not None:
                from preview.reference import neutral_contract, verify_neutral_reference
                contract = neutral_contract(row)
                if contract.get('recovery_in_place') and hold != contract['exit_hold_start_frame']:
                    raise ValueError('In-place recovery must hold the terminal neutral frame')
                if not (contract['source_start_frame'] <= start < contract['source_stop_frame']
                        and start < hold < row['frames'] - 2):
                    raise ValueError('Recovery start must remain inside its source action window')
                verify_neutral_reference(row, path, self.root, review=self.neutral_review, human=self.interface == 'human')
            if self.interface == 'human':
                human = (self.root / row.get('human_sidecar','')) if self.reference_manifest is not None else Path(row.get('human_sidecar',''))
                if not row.get('human_available') or human.is_symlink() or not human.is_file() or sha256(human) != row.get('human_sha256'):
                    raise ValueError('Neutral recovery human sidecar SHA differs')
            result[name] = dict(path=path,start_frame=start,hold_frame=hold)
        return result

    def search(self, robot, query):
        if robot not in (*ROBOTS, "all") or not isinstance(query, str) or len(query) > 256:
            raise ValueError("Invalid robot or search query")
        names = sorted(self.common) if robot == "all" else self.motions[robot]
        found = [name for name in names if query.casefold() in name.casefold()]
        return dict(robot=robot, total=len(found), motions=found[:200])


def load_release_catalogs(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows,list) or not rows:
        raise ValueError('Expected a nonempty fixed release list')
    releases = {}
    for row in rows:
        fields = {'id','label','data_root','checkpoints','interface','parent_iteration','human_updates'}
        if not fields <= set(row) <= fields | {'training_manifest','reference_manifest','recovery_windows','flip_sources'}:
            raise ValueError('Invalid fixed release fields')
        training = row.get('training_manifest')
        if row.get('reference_manifest') is not None and training is None:
            raise ValueError('Migrated references require an immutable joint training manifest')
        training_path = training_hash = None
        if training is not None:
            if not isinstance(training, dict) or set(training) != {'path', 'sha256'}:
                raise ValueError('Invalid joint training manifest')
            training_path = Path(training['path']).absolute()
            training_hash = training['sha256']
            if (not isinstance(training_hash, str) or not re.fullmatch('[0-9a-f]{64}', training_hash)
                    or training_path != training_path.resolve(strict=True) or not training_path.is_file()
                    or sha256(training_path) != training_hash):
                raise ValueError('Joint training manifest SHA differs')
        name = row['id']
        if not isinstance(name,str) or not re.fullmatch('[a-z0-9][a-z0-9-]{0,63}',name) or name in releases:
            raise ValueError('Invalid or duplicate release id')
        if set(row['checkpoints']) != set(ROBOTS):
            raise ValueError('A release requires all three robots')
        checks = {}
        for robot,entry in row['checkpoints'].items():
            check = FixedCheckpoint.pin(entry['path'],robot)
            if check.sha256 != entry['sha256']:
                raise ValueError('Release checkpoint SHA differs')
            checks[robot] = check
        expected = row['human_updates'] if row['interface']=='human' else row['parent_iteration']
        if any(c.iteration!=expected for c in checks.values()):
            raise ValueError('Release iteration differs from its fixed checkpoints')
        if row['interface']=='human':
            if training is None and row['parent_iteration']!=30000:
                raise ValueError('Human release parent iteration must match its fixed 30000 teacher')
            manifest=json.loads((Path(row['data_root'])/'manifest.json').read_text())
            if manifest.get('complete') is not True or (row.get('reference_manifest') is None and manifest.get('human_data') is not True):
                raise ValueError('Human release requires a complete original-human paired manifest')
        if row['interface']=='human' or training is not None:
            import torch
            from preview.release import verify_human_identity
            data_hash=sha256(Path(row['data_root'])/'manifest.json') if row['interface']=='human' else None
            for robot,check in checks.items():
                ckpt=torch.load(check.path,map_location='cpu',weights_only=False)
                verify_human_identity(ckpt,robot=robot,update=check.iteration,data_sha256=data_hash,training_sha256=training_hash)
                check.verify()
        catalog = Catalog(row['data_root'],checks,interface=row['interface'],reference_manifest=row.get('reference_manifest'))
        catalog.training_manifest, catalog.training_manifest_sha256 = training_path, training_hash
        if row.get("recovery_windows") is not None: catalog.set_recovery_windows(row["recovery_windows"])
        if row.get("flip_sources") is not None: catalog.set_flip_sources(row["flip_sources"])
        releases[name] = dict(catalog=catalog,
            label=row['label'],parent_iteration=row['parent_iteration'],human_updates=row['human_updates'])
    return releases


class PolicySession:
    """Uses the training env's exact obs/goals, action mapping, PD, and timestep."""
    def __init__(self, robot, path, checkpoint, device, *, render=True, source_body=False, shared_policy=None,
                 interface='motion', human_root=None, human_points='five',paired_manifest_sha256=None,training_manifest_sha256=None,start_frame=0,hold_frame=None,control_graph=False,backend='warp'):
        if backend not in ('warp','cpu'):raise ValueError('Unknown preview backend')
        if backend=='cpu' and control_graph:raise ValueError('CPU preview does not support control graph')
        if backend=='cpu' and str(device)!='cpu':raise ValueError('CPU preview currently requires a CPU actor')
        self.backend=backend
        import torch
        import yaml
        from mjlab.utils.torch import configure_torch_backends
        from morphmimic.constants import PROPRIO_DIM
        from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
        from morphmimic.env.morphmimic_env import MorphMimicEnv
        from morphmimic.eval.policy_loader import load_policy
        from morphmimic.train.obs_history import ObsHistory

        self.env = self.renderer = self.physics_graph = self.actor_graph = self.graph_stream = None
        self.tmp = tempfile.TemporaryDirectory(prefix="blinkmorph-preview-")
        self.source_path = str(path)
        self.start_frame = start_frame
        self.hold_frame = hold_frame
        self.human_root = human_root
        self.source_metadata, self.source_window = {}, None
        try:
            if source_body:
                path = self._prepare_source_body(robot, Path(path))
            configure_torch_backends(allow_tf32=False)
            checkpoint.verify()
            if shared_policy is None:
                saved = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
                if interface=='human' or training_manifest_sha256 is not None:
                    from preview.release import verify_human_identity
                    verify_human_identity(saved,robot=robot,update=checkpoint.iteration,data_sha256=paired_manifest_sha256,training_sha256=training_manifest_sha256)
                self.training_physics = saved.get("physics")
                saved_cfg = saved["cfg"]
                goal_anchor = saved_cfg.get("goal_anchor", "reference")
                self.policy, self.meta = load_policy(checkpoint.path, device, interface=interface,
                    human_points=human_points,kp_horizon=1,goal_anchor=goal_anchor)
            else:
                entry = shared_policy.entries[robot]
                saved_cfg = entry['cfg']
                self.training_physics = entry['training_physics']
                self.meta, self.policy = entry['meta'], None
                goal_anchor = saved_cfg.get('goal_anchor', 'reference')
            checkpoint.verify()
            if self.meta.robot != robot or self.meta.iteration != checkpoint.iteration:
                raise ValueError("Checkpoint robot/iteration differs from its configured fixed path")
            subset = Path(self.tmp.name) / "subset.yaml"
            subset.write_text(yaml.safe_dump(dict(root=str(path.parent), clips=[path.name])))
            cfg = morphmimic_env_cfg(robot, subset_yaml=str(subset), num_envs=1, play=True, eval=True,
                goal_anchor=goal_anchor,human_root=self.human_root)
            cfg.commands[COMMAND_NAME].library_dtype = saved_cfg.get("library_dtype", "float32")
            if control_graph:
                self.graph_stream = torch.cuda.Stream(device=device)
            if backend=='cpu':
                from preview.cpu_backend import CpuPreviewEnv
                self.env=CpuPreviewEnv(cfg,device='cpu')
            else:
                self.env = MorphMimicEnv(cfg, device=device, **({'graph_stream': self.graph_stream} if control_graph else {}))
            if abs(self.env.step_dt - 0.02) > 1e-8:
                raise ValueError("Preview requires training's 50 Hz control timestep")
            self.command = self.env.command_manager.get_term(COMMAND_NAME)
            if hold_frame is not None and (type(hold_frame) is not int or not start_frame < hold_frame < int(self.command.library.clip_len[0])-2):
                raise ValueError('Standing hold frame must be inside the selected recovery clip')
            if self.meta.interface=='human' and not self.command.library.has_human:
                raise ValueError('This motion has no original human sidecar')
            self.history = ObsHistory(1, self.meta.hist_len, PROPRIO_DIM, self.env.device)
            if render:
                if backend=='cpu':
                    from preview.cpu_backend import CpuRenderer
                    self.renderer=CpuRenderer(self.env)
                else:
                    from mjlab.viewer import OffscreenRenderer
                    cfg.viewer.width, cfg.viewer.height = 1280, 720
                    cfg.viewer.max_extra_envs = 0
                    cfg.viewer.distance = 3.0
                    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
                    cfg.viewer.entity_name = "robot"
                    cfg.viewer.elevation = -20.0
                    cfg.viewer.azimuth = 135.0
                    self.command.cfg.debug_vis = True
                    self.command.cfg.viz.mode = "ghost"
                    self.renderer = OffscreenRenderer(self.env.sim.mj_model, cfg.viewer, self.env.scene)
                    self.renderer.initialize()
            self.torch = torch
            if control_graph:
                from preview.control_graph import ActorGraph, PhysicsGraph
                self.physics_graph = PhysicsGraph(self.env)
                if self.policy is not None:
                    policy = self.policy
                    keys = ('goal', 'goal_kp', 'human_joint_valid') if self.command.library.has_human else ('goal', 'goal_kp')
                    self.actor_graph = (ActorGraph(lambda history, *values: policy(history, dict(zip(keys, values, strict=True))), device), keys)
            self.reset()
            if self.physics_graph is not None:
                from morphmimic.constants import NUM_JOINTS
                self.physics_graph.warmup(torch.zeros((1, NUM_JOINTS), device=self.env.device))
                self.reset()
        except Exception:
            self.close()
            raise

    def _prepare_source_body(self, robot, path):
        import numpy as np
        from morphmimic.constants import FPS

        manifest = path.resolve().parent.parent / "manifest.json"
        rows = json.loads(manifest.read_text()).get("clips", []) if manifest.is_file() else []
        self.source_metadata = next((row for row in rows if row.get("robot") == robot
                                     and row.get("name") == path.stem), {})
        frame_keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                      "body_lin_vel_w", "body_ang_vel_w", "root_pos_w", "root_quat_w",
                      "root_lin_vel_w", "root_ang_vel_w", "human_pos_w", "human_quat_w",
                      "time_s", "source_time_s", "source_frame_index")
        with np.load(path, allow_pickle=False) as source:
            frames = len(source["joint_pos"])
            if int(source["fps"]) != FPS:
                raise ValueError("Comparison requires prepared 50 Hz source frames")
            if self.source_metadata.get('neutral_contract') is not None:
                from preview.reference import neutral_contract
                neutral_contract(self.source_metadata)
                start, stop, from_manifest = 0, frames, True
            else:
                from morphmimic.data.motion_library import _source_frame_windows
                start, stop, from_manifest = _source_frame_windows([path], [frames], [robot])[0]
            if stop - start < 2:
                raise ValueError("Comparison source body must contain at least two frames")
            arrays = {}
            for key in source.files:
                value = source[key]
                if key in frame_keys:
                    if value.ndim == 0 or len(value) != frames:
                        raise ValueError(f"{path}: {key} frame count differs from joint_pos")
                    value = value[start:stop]
                arrays[key] = value
        self.source_window = dict(start_frame=start, stop_frame=stop, source_frames=frames,
                                  body_frames=stop - start, offset_seconds=start / FPS,
                                  duration_seconds=(stop - start - 1) / FPS,
                                  from_manifest=from_manifest)
        cropped = Path(self.tmp.name) / path.name
        np.savez(cropped, **arrays)
        if self.human_root is not None:
            from morphmimic.data.human_convert import sidecar_path
            source_human = sidecar_path(path,self.human_root)
            if not source_human.is_file():
                raise ValueError('Human comparison requires the matching original sidecar')
            self.human_root = str(Path(self.tmp.name)/'human50')
            target = sidecar_path(cropped,self.human_root)
            target.parent.mkdir(parents=True)
            with np.load(source_human,allow_pickle=False) as source:
                human = {k:source[k] for k in source.files}
            for key in ('human_pos_w','human_quat_w','source_time_s','time_s','source_frame_index'):
                if key in human:
                    if len(human[key])!=frames:raise ValueError('Human/robot frame count differs')
                    human[key]=human[key][start:stop]
            np.savez(target,**human)
        return cropped

    def reset(self):
        if getattr(self, 'physics_graph', None) is not None:
            self.physics_graph.reset()
        self._reset()

    def _reset(self):
        env_ids = self.torch.arange(1, device=self.env.device)
        self.command.assign_clips(env_ids, self.torch.zeros_like(env_ids),
                                  **({'start_frames': self.torch.full_like(env_ids, self.start_frame)}
                                     if self.start_frame else {}))
        obs, _ = self.env.reset(env_ids=env_ids)
        physics = self.env.physics_diagnostics.report()
        if physics['overflow_bits']:
            raise RuntimeError(f"Native physics overflow during reset; playback stopped: {physics}")
        self.history.reset(env_ids, obs["policy"])
        self.goals = {group: obs[group].clone() for group in ("goal", "goal_kp")}
        if self.command.library.has_human:
            self.goals['human_joint_valid'] = self.command.library.human_valid_at(self.command.global_frame)
        self.steps, self.total_reward = 0, 0.0
        self.failed = self.completed = False

    def step(self, action=None):
        with self.torch.no_grad():
            if getattr(self, 'hold_frame', None) is not None:
                self.command.global_frame.clamp_(max=self.hold_frame)
            if action is None:
                if self.policy is None:
                    raise ValueError('Shared policy world requires a batched action')
                if getattr(self, 'actor_graph', None) is None:
                    action = self.policy(self.history.flat(), self.goals)
                else:
                    actor, keys = self.actor_graph
                    action = actor(self.history.flat(), *(self.goals[key] for key in keys))
            if not bool(self.torch.isfinite(action).all()):
                raise ValueError("Policy produced a nonfinite action; simulation stopped")
            obs, reward, terminated, truncated, _ = (self.env.step(action) if getattr(self, 'physics_graph', None) is None
                                                       else self.physics_graph.step(action))
            physics = self.env.physics_diagnostics.report()
            if physics["overflow_bits"]:
                raise RuntimeError(f"Native physics overflow; playback stopped: {physics}")
            self.history.push(obs["policy"])
            self.goals = {group: obs[group].clone() for group in ("goal", "goal_kp")}
            if self.command.library.has_human:
                self.goals['human_joint_valid'] = self.command.library.human_valid_at(self.command.global_frame)
            self.total_reward += float(reward[0])
            self.steps += 1
            self.failed = bool(terminated[0])
            self.completed = bool(truncated[0]) and not self.failed

    def status(self):
        active = self.env.termination_manager.active_terms
        terms = [name for name in active if bool(self.env.termination_manager.get_term(name)[0])]
        return dict(steps=self.steps, simulated_seconds=self.steps * self.env.step_dt,
                    physics_backend=self.backend, control_graph=getattr(self, 'physics_graph', None) is not None,
                    clip_frames=int(self.command.library.clip_len[0]), reward=self.total_reward,
                    failed=self.failed, completed=self.completed, termination_terms=terms,
                    interface=self.meta.interface, input_mode=self.meta.mode, checkpoint_iteration=self.meta.iteration,
                    training_data=self.meta.train_cfg.get("subset_yaml"),
                    training_share=self.meta.train_cfg.get("share"),
                    training_physics=self.training_physics,
                    source_path=self.source_path, source_metadata=self.source_metadata,
                    source_window=self.source_window,
                    initial_frame=self.start_frame, standing_hold_frame=self.hold_frame, reference_frame=int(self.command.global_frame[0]),
                    source_start_frame=self.source_window["start_frame"] if self.source_window else 0,
                    source_stop_frame=self.source_window["stop_frame"] if self.source_window else int(self.command.library.clip_len[0]),
                    source_frames=self.source_window["source_frames"] if self.source_window else int(self.command.library.clip_len[0]),
                    physics=self.env.physics_diagnostics.report())

    def frame_rgb(self):
        if self.renderer is None:
            raise ValueError("This policy session has no individual renderer")
        self.renderer.update(self.env.sim.data, debug_vis_callback=self.command._debug_vis_impl)
        return self.renderer.render().copy()

    def frame(self):
        from PIL import Image
        output = io.BytesIO()
        Image.fromarray(self.frame_rgb()).save(output, format="JPEG", quality=80)
        return output.getvalue()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        if self.env is not None:
            self.env.close()
        self.tmp.cleanup()


class Preview:
    def __init__(self, catalog, device="cuda:0", factory=PolicySession,
                 comparison_factory=None, compare_devices=None, releases=None, control_graph=False, backend='warp'):
        if backend not in ('warp','cpu'):raise ValueError('Unknown preview backend')
        if backend=='cpu' and control_graph:raise ValueError('CPU preview does not support control graph')
        if backend=='cpu' and (str(device)!='cpu' or any(str(d)!='cpu' for d in (compare_devices or (device,)))):
            raise ValueError('CPU preview currently requires a CPU actor')
        self.backend=backend
        self.catalog, self.device, self.factory = catalog, device, factory
        self.releases = releases or {'motion-30000':dict(catalog=catalog,label='Motion 30000')}
        self.release_id = next(iter(self.releases))
        self.catalog = self.releases[self.release_id]['catalog']
        self.recordings = None
        self.comparison_factory = comparison_factory
        self.control_graph = control_graph
        self.compare_devices = tuple(compare_devices or (device,) * 3)
        self.lock = threading.Lock()
        self.requests = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.jpeg = None
        self.rgb = None
        self.state = dict(physics_backend=backend, mode="single", session_generation=0, robot=None, motion=None, loading=False, playing=False,
                          failed=False, completed=False, error=None, steps=0, frame_id=0, frame_available=False,
                          participants=[], simulation_hz=0., render_fps=0., sim_speed=0.,
                          qualified=False, session_scope="shared", control_hz=50,
                          rendering="MuJoCo scene; solid policy/physics, translucent reference")

    def snapshot(self):
        with self.lock:
            return {**self.state, "robots": list(ROBOTS),
                    'release_id':self.release_id,
                    'releases':[dict(id=name,**{k:v for k,v in entry.items() if k!='catalog'},
                        interface=entry['catalog'].interface,
                        joint_training=entry['catalog'].training_manifest is not None,
                        neutral_standing=entry['catalog'].recovery_windows is not None,
                        flips=entry['catalog'].flip_sources is not None) for name,entry in self.releases.items()],
                    "checkpoints": {r: c.public() for r, c in self.catalog.checkpoints.items()},
                    "data_root": str(self.catalog.root),
                    "data_manifest_sha256": self.catalog.manifest_sha256}

    def catalog_for(self, release=None):
        name = self.release_id if release is None else release
        if not isinstance(name,str) or name not in self.releases:
            raise ValueError('Unknown fixed release')
        return self.releases[name]['catalog']

    def search(self,robot,query,release=None):
        return self.catalog_for(release).search(robot,query)

    def latest_frame(self):
        with self.lock:
            return self.state["frame_id"], self.rgb

    def diagnostic_jpeg(self):
        with self.lock:
            rgb, jpeg = self.rgb, self.jpeg
        if rgb is None:
            return jpeg
        from PIL import Image
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format="JPEG", quality=80)
        return output.getvalue()

    def recording_catalog(self, release, identity):
        if self.recordings is None:
            raise ValueError('PICO 上传尚未启用')
        return self.recordings.catalog(self.catalog_for(release), identity)

    def control(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected a control object")
        action = data.get("action")
        if action == "standup":
            if not {"action", "robot"}.issubset(data) or set(data) - {"action", "robot", "release", "pose"}:
                raise ValueError("Standup requires robot and optional fixed release and pose")
            pose = data.get('pose', 'lying')
            if not isinstance(pose, str) or pose not in STANDUP_POSES:
                raise ValueError("Unknown standup pose")
            release = data.get('release')
            if 'release' not in data:
                release = next((name for name, entry in self.releases.items()
                                if entry['catalog'].interface == 'motion'
                                and entry['catalog'].recovery_windows is not None), None)
                if release is None:
                    release = ('joint-motion-6000' if 'joint-motion-6000' in self.releases
                               else 'joint-motion-5000')
            catalog = self.catalog_for(release)
            migrated = getattr(catalog, 'reference_manifest', None) is not None
            legacy = not migrated and release in ('joint-motion-6000', 'joint-motion-5000')
            reviewed = catalog.recovery_windows is not None and (migrated or release in ('joint-human-6000','joint-motion-7000','joint-human-7000'))
            if not (legacy or reviewed):
                raise ValueError('Standup requires a fixed motion release or a reviewed neutral standing release')
            motion, _ = STANDUP_POSES[pose]
            if catalog.recovery_windows is not None:
                catalog.select_recovery(data['robot'],pose)
                motion = 'neutral_'+pose
            else:
                catalog.select(data['robot'], motion)
            data = dict(action='standup', robot=data['robot'], motion=motion, release=release, pose=pose)
        elif action == "select":
            if not {"action","robot","motion"}.issubset(data) or set(data)-{"action","robot","motion","release","human_points","recording"}:
                raise ValueError("Select requires robot, motion and optional fixed release")
            catalog = (self.recording_catalog(data.get('release'), data['recording'])
                       if 'recording' in data else self.catalog_for(data.get('release')))
            if 'recording' in data and (data['motion'] != 'pico' or data.get('human_points', 'five') != 'five'):
                raise ValueError('PICO 录制仅支持五点输入')
            if 'recording' in data:
                data = dict(data, human_points='five')
            if 'human_points' in data and (catalog.interface!='human' or data['human_points'] not in ('five','full')):
                raise ValueError('Human points must be five or full on a human release')
            catalog.select(data["robot"], data["motion"])
        elif action in ("start", "pause", "reset"):
            if set(data) != {"action"}:
                raise ValueError("This action accepts no parameters")
            with self.lock:
                if self.state["loading"]:
                    raise ValueError("Wait for checkpoint/environment loading")
                if self.state["robot"] is None or self.state["error"]:
                    raise ValueError("Select a motion with a fixed checkpoint first")
                if action == "start" and (self.state["failed"] or self.state["completed"]):
                    raise ValueError("Reset the finished simulation before starting")
        else:
            raise ValueError("Unknown action")
        try:
            self.requests.put_nowait(data)
        except queue.Full:
            raise ValueError("Control queue is busy") from None

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def run(self):
        session = None
        deadline = time.monotonic()
        next_render = last_status = 0.
        rendered_serial = 0
        rate_started, rate_steps, rate_frames = deadline, 0, 0
        try:
            while not self.stop.is_set():
                try:
                    request = self.requests.get(timeout=0 if self.state["playing"] else 0.05)
                except queue.Empty:
                    request = None
                try:
                    if request:
                        action = request["action"]
                        if action in ("select", "standup"):
                            self.update(loading=True, frame_available=False, playing=False, error=None, failed=False,
                                        input_mode=None,interface=None,
                                        completed=False, steps=0, simulated_seconds=0, source=None, checkpoint=None,
                                        clip_frames=0, motion=request["motion"], robot=request["robot"],
                                        recording_id=request.get("recording"),
                                        mode="compare" if request["robot"] == "all" else "single",
                                        session_generation=self.state["session_generation"] + 1,
                                        participants=[], simulation_hz=0., render_fps=0., sim_speed=0.)
                            if session:
                                session.close()
                                session = None
                            rendered_serial = 0
                            release_id = request.get('release',self.release_id)
                            catalog = (self.recording_catalog(release_id, request['recording'])
                                       if 'recording' in request else self.catalog_for(release_id))
                            with self.lock:
                                self.release_id,self.catalog = release_id,catalog
                            session_factory = self.factory
                            if self.backend=='cpu':
                                from functools import partial
                                session_factory=partial(session_factory,backend='cpu')
                            policy_factory = None
                            if self.control_graph:
                                from functools import partial
                                from preview.shared_policy import SharedPolicy
                                session_factory = partial(self.factory,control_graph=True)
                                policy_factory = partial(SharedPolicy,control_graph=True)
                            if self.catalog.training_manifest_sha256 is not None:
                                from functools import partial
                                from preview.shared_policy import SharedPolicy
                                session_factory = partial(session_factory,training_manifest_sha256=self.catalog.training_manifest_sha256)
                                policy_factory = partial(policy_factory or SharedPolicy,training_manifest_sha256=self.catalog.training_manifest_sha256)
                            if self.catalog.interface=='human':
                                from functools import partial
                                from preview.shared_policy import SharedPolicy
                                points = request.get('human_points','five')
                                session_factory = partial(session_factory,interface='human',human_points=points,human_root=self.catalog.human_root if self.catalog.reference_manifest else str(self.catalog.root/'human50'),paired_manifest_sha256=self.catalog.manifest_sha256,training_manifest_sha256=self.catalog.training_manifest_sha256)
                                policy_factory = partial(policy_factory or SharedPolicy,interface='human',human_points=points,paired_manifest_sha256=self.catalog.manifest_sha256,training_manifest_sha256=self.catalog.training_manifest_sha256)
                            robot, motion = request["robot"], request["motion"]
                            if self.catalog.interface == 'human' and motion in self.catalog.flip_motions:
                                from functools import partial
                                session_factory = partial(session_factory,human_root=self.catalog.human_root)
                            if action == 'standup' and self.catalog.recovery_windows is not None:
                                selected = self.catalog.select_recovery(robot,request['pose'])
                                if self.catalog.interface == 'human':
                                    from functools import partial
                                    human_root = self.catalog.human_root
                                    session_factory = partial(session_factory,human_root=human_root,human_points='five')
                                base_factory = session_factory
                                def session_factory(name, *args, **kwargs):
                                    option = selected[name]
                                    return base_factory(name,*args,start_frame=option['start_frame'],hold_frame=option['hold_frame'],**kwargs)
                                path = {name:option['path'] for name,option in selected.items()} if robot == 'all' else selected[robot]['path']
                            else:
                                if action == 'standup':
                                    from functools import partial
                                    session_factory = partial(session_factory, start_frame=STANDUP_POSES[request['pose']][1])
                                path = self.catalog.select(robot, motion)
                            if robot == "all":
                                factory = self.comparison_factory
                                if factory is None:
                                    from comparison import ComparisonSession
                                    factory = ComparisonSession
                                options = dict(session_factory=session_factory)
                                if policy_factory is not None:options['policy_factory']=policy_factory
                                session = factory(path, self.catalog.checkpoints, self.compare_devices,**options)
                                self.update(source={r: str(p) for r, p in path.items()}, **session.status())
                            else:
                                session = session_factory(robot, path, self.catalog.checkpoints[robot], self.device)
                                self.update(source=str(path), checkpoint=self.catalog.checkpoints[robot].public(),
                                            **session.status())
                        elif action == "pause":
                            self.update(playing=False, **(session.status() if session else {}))
                        elif action == "reset" and session:
                            self.update(playing=False, frame_available=False)
                            session.reset()
                            self.update(error=None, **session.status())
                        elif action == "start" and session and not (session.failed or session.completed):
                            self.update(playing=True)
                            deadline = time.monotonic()
                            rate_started, rate_steps, rate_frames = deadline, 0, 0
                    stepped = False
                    if self.state["playing"] and session:
                        session.step()
                        stepped = True
                        rate_steps += 1
                        self.update(steps=session.steps, simulated_seconds=session.steps * .02,
                                    failed=session.failed, completed=session.completed)
                        if time.monotonic() - last_status >= .25 or session.failed or session.completed:
                            self.update(**session.status())
                            last_status = time.monotonic()
                        if session.failed or session.completed:
                            self.update(playing=False)
                    if session and (request or stepped and (time.monotonic() >= next_render or session.failed or session.completed)):
                        if hasattr(session, "submit_frame"):
                            session.submit_frame(reset=bool(request and request["action"] in ("select", "standup", "reset")))
                        else:
                            if hasattr(session, "frame_rgb"):
                                rgb, jpeg = session.frame_rgb(), None
                            else:
                                rgb, jpeg = None, session.frame()
                            with self.lock:
                                self.rgb, self.jpeg = rgb, jpeg
                                self.state["frame_id"] += 1
                                self.state.update(loading=False, frame_available=True)
                            rate_frames += 1
                        next_render = max(next_render + 1 / 30, time.monotonic())
                    if session and hasattr(session, "latest_render"):
                        rendered = session.latest_render()
                        if rendered and rendered[0] != rendered_serial:
                            rendered_serial, rgb = rendered
                            with self.lock:
                                self.rgb, self.jpeg = rgb, None
                                self.state["frame_id"] += 1
                                self.state.update(loading=False, frame_available=True)
                            rate_frames += 1
                    now = time.monotonic()
                    if now - rate_started >= 1.:
                        elapsed = now - rate_started
                        self.update(simulation_hz=rate_steps / elapsed, render_fps=rate_frames / elapsed,
                                    sim_speed=rate_steps * .02 / elapsed)
                        rate_started, rate_steps, rate_frames = now, 0, 0
                    if self.state["playing"]:
                        deadline += 0.02
                        self.stop.wait(max(0.0, deadline - time.monotonic()))
                        # Physics never skips steps when the worker falls behind.
                        deadline = max(deadline, time.monotonic() - 0.02)
                except Exception as error:
                    physics = {}
                    if session is not None and getattr(session, "env", None) is not None:
                        physics["physics"] = session.env.physics_diagnostics.report()
                    self.update(loading=False, playing=False, error=f"{type(error).__name__}: {error}", **physics)
        finally:
            if session:
                session.close()


def handler_for(preview):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, content_type="application/json"):
            if content_type == "application/json":
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path)
            try:
                if route.path == "/api/status":
                    self.respond(200, preview.snapshot())
                elif route.path == "/api/motions":
                    query = parse_qs(route.query)
                    self.respond(200, preview.search(query.get("robot", ["q1"])[0], query.get("q", [""])[0],query.get('release',[None])[0]))
                elif route.path == "/api/frame":
                    frame = preview.diagnostic_jpeg()
                    if frame is None:
                        self.respond(409, dict(error="No policy simulation frame; select a configured robot/motion"))
                    else:
                        self.respond(200, frame, "image/jpeg")
                elif route.path in ("/", "/app.js", "/style.css", "/preview-font.woff2"):
                    filename = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/preview-font.woff2": "preview-font.woff2"}[route.path]
                    mime = {"/": "text/html; charset=utf-8", "/app.js": "text/javascript", "/style.css": "text/css", "/preview-font.woff2": "font/woff2"}[route.path]
                    self.respond(200, (STATIC / filename).read_bytes(), mime)
                else:
                    self.respond(404, dict(error="Not found"))
            except (ValueError, TypeError) as error:
                self.respond(400, dict(error=str(error)))

        def do_POST(self):
            try:
                if urlsplit(self.path).path != "/api/control":
                    self.respond(404, dict(error="Not found"))
                    return
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    raise ValueError("Cross-origin controls are refused")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Control JSON must be between 1 and 4096 bytes")
                preview.control(json.loads(self.rfile.read(size)))
                self.respond(202, dict(accepted=True))
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.respond(400, dict(error=str(error)))

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pico-data-root", help="Persistent directory for uploaded PICO recordings")
    parser.add_argument("--data-root", required=True, help="Prepared NPZ root with q1/bumi/g1 subdirectories")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="ROBOT=PATH", help="Fixed robot/ckpt_<iteration>.pt (repeat per robot); omission disables that robot")
    parser.add_argument('--releases',help='Reviewed JSON list of fixed checkpoint releases')
    parser.add_argument("--backend", choices=("warp","cpu"), default="warp", help="Physics backend (default: warp)")
    parser.add_argument("--device", default=None, help="Actor device (default: cuda:0 for warp, cpu for CPU)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31769)
    parser.add_argument("--public-host", help="Public IPv4 for the configured UDP31870–31889 media relay")
    parser.add_argument("--compare-devices", help="One or three comma-separated devices for synchronized comparison")
    parser.add_argument("--control-graph", action="store_true", help="Capture actor and fixed-shape no-reset physics in separate CUDA graphs")
    parser.add_argument("--nvenc", action="store_true", help="Require NVENC H264 video; startup fails if unavailable")
    parser.add_argument("--encoder-gpu", type=int, default=0, help="Visible GPU index for NVENC; use a GPU outside the physics barrier")
    args = parser.parse_args()
    args.device = args.device or ('cpu' if args.backend=='cpu' else 'cuda:0')
    if args.backend=='cpu' and args.control_graph:parser.error('CPU preview does not support control graph')
    if args.backend=='cpu' and args.device!='cpu':parser.error('CPU preview currently requires a CPU actor')
    checkpoints = {}
    for item in args.checkpoint:
        robot, sep, path = item.partition("=")
        if not sep or robot not in ROBOTS or robot in checkpoints:
            parser.error("--checkpoint requires a distinct q1|bumi|g1=PATH")
        checkpoints[robot] = FixedCheckpoint.pin(path, robot)
    os.environ.setdefault("MUJOCO_GL", "egl")
    devices = tuple(args.compare_devices.split(",")) if args.compare_devices else (args.device,) * 3
    if len(devices) == 1:
        devices *= 3
    if len(devices) != 3 or any(not d.strip() for d in devices):
        parser.error("--compare-devices requires one or three nonempty comma-separated devices")
    if args.backend=="cpu" and any(d!="cpu" for d in devices):parser.error("CPU preview requires CPU comparison devices")
    releases = load_release_catalogs(args.releases) if args.releases else None
    catalog = next(iter(releases.values()))['catalog'] if releases else Catalog(args.data_root,checkpoints)
    preview = Preview(catalog, args.device, compare_devices=devices,releases=releases,control_graph=args.control_graph,backend=args.backend)
    if args.pico_data_root:
        from preview.pico import PicoRecordings
        base = next((entry['catalog'] for entry in preview.releases.values()
                     if entry['catalog'].interface == 'human'), None)
        if base is None:
            parser.error('--pico-data-root requires a Human model release')
        templates = {}
        for robot in ROBOTS:
            template = base.select(robot, 'dance1_subject1')
            templates[robot] = dict(motion=template,
                                   human=Path(base.human_root) / robot / template.name)
        preview.recordings = PicoRecordings(args.pico_data_root, templates)
    worker = threading.Thread(target=preview.run, name="policy-physics", daemon=True)
    import asyncio
    from aiohttp import web
    from webrtc import create_app
    app = create_app(preview, public_host=args.public_host, nvenc=args.nvenc, encoder_gpu=args.encoder_gpu)
    async def start_worker(_app):
        worker.start()
    async def stop_worker(_app):
        preview.stop.set()
        await asyncio.to_thread(worker.join, 10)
    app.on_startup.append(start_worker)
    app.on_cleanup.append(stop_worker)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
