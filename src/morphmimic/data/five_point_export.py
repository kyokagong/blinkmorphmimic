"""Atomic, provenance-bound five-point exports from authentic prepared human poses.

Human poses are never inferred from robot FK. Native bytes are hash-checked for
derived neutral contracts. Human sources, clocks and source maps remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from morphmimic.constants import HUMAN_FIVE_IDX, HUMAN_JOINTS
from morphmimic.data.human_prepare import five_points
from morphmimic.data.local_prepare import sha256

_SCHEMA = 'authentic-human-five-point-v1'


def _json(value):
  return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _component(value):
  if not isinstance(value,str) or not value or Path(value).name!=value or value in ('.','..') or '\\' in value:
    raise ValueError('Unsafe export path component')
  return value


def _plain(path):
  if path.is_symlink() or path.absolute()!=path.resolve():
    raise ValueError(f'Refuse symbolic path: {path}')


def _publish(path,content,*,resume):
  """Durable temporary write then atomic no-clobber link, including crash recovery."""
  _plain(path)
  path.parent.mkdir(parents=True,exist_ok=True)
  _plain(path.parent)
  if path.exists():
    if not resume: raise FileExistsError(path)
    if path.read_bytes()!=content: raise ValueError(f'Existing output/receipt differs: {path}')
    return
  descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
  try:
    with os.fdopen(descriptor,'wb') as stream:
      stream.write(content);stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path)
    parent_fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(parent_fd)
    finally: os.close(parent_fd)
  finally:
    Path(temporary).unlink(missing_ok=True)


def _source(path,expected):
  path=Path(path).absolute();_plain(path)
  if not isinstance(expected,str) or len(expected)!=64 or sha256(path)!=expected:
    raise ValueError(f'Source SHA256 mismatch: {path}')
  return path


def _neutral_clock(row,human,frames,timestamp):
  """Derived clocks distinguish output time, original samples and synthetic spans."""
  from morphmimic.data.neutral_bookends import source_frame_map
  contract=row['neutral_contract']
  fields=('frames','fps','source_start_frame','source_stop_frame','entry_hold_end_frame',
          'exit_hold_start_frame','entry_frames','exit_frames','original_source_frames','source_ramp_frames')
  if not isinstance(contract,dict) or any(type(contract.get(k)) is not int for k in fields):
    raise ValueError('Invalid neutral contract integer fields')
  a,b=contract['source_start_frame'],contract['source_stop_frame']
  hold,end_hold=contract['entry_hold_end_frame'],contract['exit_hold_start_frame']
  if (contract['frames']!=frames or contract['fps']!=50 or contract.get('robot')!=row['robot']
      or not 50<=hold<a<b<=end_hold<=frames-50
      or a-hold!=contract['entry_frames'] or end_hold-b!=contract['exit_frames']
      or min(contract['entry_frames'],contract['exit_frames'])<50):
    raise ValueError('Neutral contract frame intervals differ from exported clip')
  mapping=source_frame_map(contract['original_source_frames'],contract['source_ramp_frames'])
  clock=np.asarray(human.get('time_s'));index=np.asarray(human.get('source_frame_index'))
  if clock.shape!=(frames,) or not np.isfinite(clock).all() or not np.allclose(clock,np.arange(frames)/50,atol=1e-8,rtol=0):
    raise ValueError('Neutral output time_s must be the complete contiguous 50 Hz clock')
  if (b-a!=len(mapping) or index.shape!=(frames,) or not np.isfinite(index).all()
      or not np.allclose(index[a:b],mapping,atol=1e-8,rtol=0)
      or not np.all(index[:a]==-1) or not np.all(index[b:]==-1)):
    raise ValueError('Neutral source_frame_index differs from the authorized source map')
  if timestamp.shape!=(frames,) or not np.isnan(timestamp[:a]).all() or not np.isnan(timestamp[b:]).all():
    raise ValueError('Synthetic neutral spans require NaN source timestamps')
  if contract.get('source_clock_status')=='unavailable_in_inherited_derived_sidecar':
    if row.get('derived') is not True or not np.isnan(timestamp[a:b]).all():
      raise ValueError('Unavailable source clock requires an explicitly inherited derived sidecar')
  elif (not np.isfinite(timestamp[a:b]).all() or not (np.diff(timestamp[a:b])>0).all()
        or not np.allclose(timestamp[a:b],timestamp[a]+mapping/50,atol=1e-8,rtol=0)):
    raise ValueError('Neutral original source timestamps differ from the authorized slow map')
  crop=row.get('pair_alignment')
  if crop is not None and (crop.get('robot_start_frame')!=a or crop.get('robot_stop_frame')!=b):
    raise ValueError('Neutral pair crop differs from the retimed core contract')
  return clock,index


def export_five_points(manifest,output_root,*,workers=8,resume=False):
  """Export only explicitly human_available rows, checking both source identities.

  A root marker pins the input manifest. Resume rechecks raw source identities,
  sidecar bytes, each receipt and output hash. An interrupted output lacking its
  receipt is accepted only after deterministic byte-for-byte source reconstruction.
  Existing unowned files and changed source data are never overwritten.
  """
  if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=16:
    raise ValueError('workers must be an integer in [1,16]')
  manifest=Path(manifest).absolute();raw=manifest.read_bytes()
  manifest_sha=hashlib.sha256(raw).hexdigest();document=json.loads(raw)
  if document.get('schema_version')!=1 or document.get('complete') is not True:
    raise ValueError('A complete schema1 composite manifest is required')
  rows=document['clips'];root=Path(output_root).absolute();_plain(root)
  marker=root/'.five-point-export.json'
  if root.exists() and not marker.exists() and any(root.iterdir()):
    raise FileExistsError(f'Export root contains unrelated files: {root}')
  owner=dict(schema=_SCHEMA,input_manifest_sha256=manifest_sha,human_joint_indices=list(HUMAN_FIVE_IDX),fps=50)
  _publish(marker,_json(owner),resume=resume)
  eligible=[];skipped=[];identities=set();sources={}
  for row in rows:
    identity=(_component(row['robot']),_component(row['name']))
    if identity in identities: raise ValueError(f'Duplicate export identity: {identity}')
    identities.add(identity)
    if row.get('human_available') is not True:
      skipped.append(dict(robot=identity[0],name=identity[1],dataset=row['dataset'],
                          source_id=row.get('source_id'),reason='no authentic human sidecar; motion-only source'))
      continue
    if row.get('fps')!=50 or row.get('frames',0)<2:
      raise ValueError(f'Invalid prepared fps/frame count: {identity}')
    source=row.get('pair_source')
    if not isinstance(source,dict) or not source.get('path') or not source.get('sha256'):
      raise ValueError(f'Authentic original-human provenance is required: {identity}')
    if Path(source['path']).suffix.lower() not in ('.bvh','.fbx','.npz'):
      raise ValueError('Original-human source must not be a robot CSV trajectory')
    previous=sources.setdefault(source['path'],source['sha256'])
    if previous!=source['sha256']: raise ValueError('Conflicting original-human source SHA')
    eligible.append(row)

  def verify_original(item): _source(*item)
  # Each original source is read once per run, independently of its three robot crops.
  with ThreadPoolExecutor(max_workers=workers) as pool:
    for _ in pool.map(verify_original,sources.items()): pass

  def convert(row):
    side=_source(row['human_sidecar'],row['human_sha256'])
    stem=Path(row['robot'])/row['name'];relative=stem.with_name(stem.name+'.npz')
    receipt_relative=stem.with_name(stem.name+'.receipt.json')
    output=root/relative;receipt_path=root/receipt_relative
    provenance=dict(schema=_SCHEMA,input_manifest_sha256=manifest_sha,robot=row['robot'],
      name=row['name'],dataset=row['dataset'],source_id=row.get('source_id'),split=row.get('split'),
      frames=row['frames'],fps=50,human_sidecar=str(side),human_sha256=row['human_sha256'],
      pair_source=row['pair_source'],pair_alignment=row.get('pair_alignment'),
      source_body_window_original=row.get('source_body_window_original'),
      human_joint_indices=list(HUMAN_FIVE_IDX),output=str(relative),receipt=str(receipt_relative))
    if 'neutral_contract' in row:
      native=_source(row['output'],row['output_sha256'])
      provenance.update(neutral_contract=row['neutral_contract'],synthetic_derived=True,
        native_reference=dict(path=str(native),sha256=row['output_sha256']),
        timing_semantics='time_s is output time; source_frame_index follows authorized original edge retiming; -1/NaN marks synthetic bookends; inherited unavailable clocks remain explicit.')
    _plain(output);_plain(receipt_path)
    if output.exists() and receipt_path.exists():
      if not resume: raise FileExistsError(output)
      saved=json.loads(receipt_path.read_bytes())
      expected=dict(provenance,output_sha256=sha256(output))
      if saved!=expected: raise ValueError(f'Output SHA/receipt mismatch: {output}')
      return expected,True
    # Bind decoding to the checked bytes even if the path changes during a read.
    source_bytes=side.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest()!=row['human_sha256']:
      raise ValueError(f'Source SHA256 changed during read: {side}')
    with np.load(io.BytesIO(source_bytes),allow_pickle=False) as saved:
      human={key:saved[key] for key in saved.files}
    positions,quaternions=five_points(human)
    frames=len(positions);valid=np.asarray(human['human_joint_valid'])
    if frames!=row['frames'] or np.asarray(human['fps']).shape!=() or float(human['fps'])!=50:
      raise ValueError('Human sidecar frame count/fps differs from paired manifest')
    if valid.shape!=(24,) or not np.isin(valid,(False,True)).all():
      raise ValueError('Invalid authentic-human joint validity mask')
    if 'human_joint_valid' in row and not np.array_equal(valid,row['human_joint_valid']):
      raise ValueError('Human validity mask differs from manifest')
    timestamp=np.asarray(human['source_time_s'])
    clock=np.arange(frames,dtype=np.float64)/50;source_index=None
    if 'neutral_contract' in row:
      clock,source_index=_neutral_clock(row,human,frames,timestamp)
    else:
      if timestamp.shape!=(frames,) or not np.isfinite(timestamp).all() or not np.allclose(np.diff(timestamp),1/50,atol=1e-8,rtol=0):
        raise ValueError('Human source timestamps must be finite contiguous 50 Hz samples')
      crop=row.get('pair_alignment')
      if crop is not None and crop['robot_stop_frame']-crop['robot_start_frame']!=frames:
        raise ValueError('Per-robot crop does not match exported frame count')
    arrays=dict(human_pos_w=positions,human_quat_w=quaternions,
      human_joint_indices=np.array(HUMAN_FIVE_IDX,dtype=np.int64),
      human_joint_names=np.array([HUMAN_JOINTS[i] for i in HUMAN_FIVE_IDX]),
      human_joint_valid=valid[list(HUMAN_FIVE_IDX)].astype(bool),fps=np.array(50,dtype=np.int64),
      time_s=clock,source_time_s=timestamp)
    if source_index is not None: arrays['source_frame_index']=source_index
    buffer=io.BytesIO();np.savez(buffer,**arrays);content=buffer.getvalue()
    receipt=dict(provenance,output_sha256=hashlib.sha256(content).hexdigest())
    was_present=output.exists()
    _publish(output,content,resume=resume)
    _publish(receipt_path,_json(receipt),resume=resume)
    return receipt,was_present

  with ThreadPoolExecutor(max_workers=workers) as pool:
    converted=list(pool.map(convert,eligible))
  clips=[row for row,_ in converted]
  missing=dict(sorted(Counter(row['dataset'] for row in skipped).items()))
  summary=dict(**owner,complete=True,clips=clips,skipped=skipped,
               counts=dict(exported=len(clips),skipped=len(skipped)),missing_human_by_dataset=missing)
  _publish(root/'manifest.json',_json(summary),resume=resume)
  resumed=sum(was_present for _,was_present in converted)
  return dict(summary,counts=dict(exported=len(clips)-resumed,resumed=resumed,skipped=len(skipped)))


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest',required=True)
  parser.add_argument('--output-root',required=True)
  parser.add_argument('--workers',type=int,default=8)
  parser.add_argument('--resume',action='store_true')
  args=parser.parse_args()
  report=export_five_points(args.manifest,args.output_root,workers=args.workers,resume=args.resume)
  print(json.dumps(dict(output_root=str(Path(args.output_root).absolute()),counts=report['counts'],
                        missing_human_by_dataset=report['missing_human_by_dataset']),sort_keys=True))


if __name__=='__main__': main()
