"""Bind conservative reference-ground calibration to immutable prepared clips."""
import hashlib
import json
import math
from pathlib import Path

import torch


def load_reference_floors(receipt_path, manifest_path, robot, clip_paths, device):
    raw = Path(receipt_path).read_bytes()
    receipt = json.loads(raw)
    manifest_raw = Path(manifest_path).read_bytes()
    if receipt.get('schema_version') != 1 or receipt.get('prepared_manifest_sha256') != hashlib.sha256(manifest_raw).hexdigest():
        raise ValueError('Reference-floor manifest identity differs')
    sources = {}
    for row in json.loads(manifest_raw)['clips']:
        key = (row['robot'], row['name'])
        if key in sources:
            raise ValueError('Duplicate reference-floor source identity')
        sources[key] = row
    calibrations = {}
    for row in receipt['clips']:
        key = (row['robot'], row['name'])
        source = sources.get(key)
        if key in calibrations or source is None or row['sha256'] != source['output_sha256'] or Path(row['path']).resolve() != Path(source['output']).resolve():
            raise ValueError('Reference-floor source identity differs')
        if type(row['valid']) is not bool or type(row['offset_m']) not in (int, float) or not math.isfinite(row['offset_m']) or abs(row['offset_m']) > .12:
            raise ValueError('Invalid reference-floor estimate')
        if not row['valid'] and row['offset_m'] != 0:
            raise ValueError('Unknown reference floor must not contain an offset')
        calibrations[key] = row
    selected = []
    for path in clip_paths:
        key = (robot, Path(path).stem)
        row = calibrations.get(key)
        if row is None or Path(row['path']).resolve() != Path(path).resolve():
            raise ValueError('Missing reference-floor source for training clip')
        selected.append(row)
    floor = torch.tensor([row['offset_m'] for row in selected], dtype=torch.float32, device=device)
    valid = torch.tensor([row['valid'] for row in selected], dtype=torch.bool, device=device)
    return floor, valid, hashlib.sha256(raw).hexdigest()
