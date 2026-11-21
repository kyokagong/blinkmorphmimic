"""Write a source-bound per-clip floor-calibration receipt without changing motions."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from morphmimic.data.contact_calibration import FootContactGeometry,calibrate_reference_file
from morphmimic.data.five_point_export import _json,_plain,_publish
from morphmimic.robots import get_robot


def calibrate_manifest(manifest,output,*,workers=8):
  started=time.monotonic()
  if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=16:
    raise ValueError('workers must be an integer in [1,16]')
  output=Path(output).absolute();_plain(output)
  if output.exists():raise FileExistsError(output)
  raw=Path(manifest).read_bytes();source=json.loads(raw)
  if source.get('schema_version')!=1 or source.get('complete') is not True:
    raise ValueError('Complete schema1 prepared manifest required')
  rows=source['clips'];seen=set();names=sorted({row['robot'] for row in rows})
  for row in rows:
    identity=(row['robot'],row['name'])
    if identity in seen:raise ValueError('Duplicate robot/clip identity')
    seen.add(identity)
    if row['split'] not in ('train','val') or not isinstance(row.get('output_sha256'),str) or len(row['output_sha256'])!=64:
      raise ValueError('Every training/validation reference requires a source SHA256')
  specs={name:get_robot(name) for name in names}
  geometries={name:FootContactGeometry.from_spec(specs[name]) for name in names}
  metadata={}
  for name,geometry in geometries.items():
    serial=[[{key:value if isinstance(value,str) else value.tolist() for key,value in surface.items()} for surface in foot] for foot in geometry.feet]
    hashes=sorted({row['asset_sha256'] for row in rows if row['robot']==name and row.get('asset_sha256')})
    metadata[name]=dict(collision_geometry_sha256=hashlib.sha256(_json(serial)).hexdigest(),
      asset_sha256=hashes[0] if len(hashes)==1 else None,manifest_asset_sha256_values=hashes,
      manifest_asset_sha256_missing_rows=sum(row['robot']==name and not row.get('asset_sha256') for row in rows))
  def calibrate(row):
    name=row['robot']
    receipt=calibrate_reference_file(row['output'],geometries[name],specs[name],expected_sha256=row['output_sha256'])
    return dict(name=row['name'],dataset=row['dataset'],split=row['split'],**receipt)
  pool=ThreadPoolExecutor(max_workers=workers)
  try:clips=list(pool.map(calibrate,rows))
  finally:pool.shutdown(wait=True,cancel_futures=True)
  counts=Counter((row['robot'],row['valid']) for row in clips)
  report=dict(schema_version=1,prepared_manifest_sha256=hashlib.sha256(raw).hexdigest(),
    method='stationary-collision-sole-kinematic-v1',clips=clips,geometry=metadata,
    summary={name:dict(valid=counts[name,True],unknown=counts[name,False]) for name in names},
    calibration=dict(fps='per-clip',declared_floor=0.,max_offset=.12,stationary_speed=.08,
      stationary_angular_speed=.35,minimum_support_seconds=.2,support_band=.015,max_spread=.02),
    workers=workers,elapsed_seconds=time.monotonic()-started)
  _publish(output,_json(report),resume=False)
  return report


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest',required=True)
  parser.add_argument('--output',required=True)
  parser.add_argument('--workers',type=int,default=8)
  args=parser.parse_args()
  report=calibrate_manifest(args.manifest,args.output,workers=args.workers)
  print(json.dumps({key:report[key] for key in ('prepared_manifest_sha256','summary','workers','elapsed_seconds')},sort_keys=True))


if __name__=='__main__':main()
