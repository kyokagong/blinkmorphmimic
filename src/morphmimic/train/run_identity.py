"""Bind a training resume to its prepared data and optional freeze receipt."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
import re

_SHA=re.compile(r'[0-9a-f]{64}\Z')
_ROOTS=('src','scripts','preview','deploy','tests','docs')
_EXTENSIONS={'.py','.sh','.js','.css','.html','.woff2','.md'}
_ROOT_FILES=('pyproject.toml','Dockerfile','Dockerfile.preview','Dockerfile.preview-deps','.dockerignore','README.md','LICENSE','MORPHMIMIC_LICENSE','THIRD_PARTY_NOTICES.md')


def _canonical(value):
    try:return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    except (ValueError,TypeError) as e:raise ValueError('Configuration must be finite JSON') from e


def _sha(data):return hashlib.sha256(data).hexdigest()


def _valid_hash(value):return isinstance(value,str) and _SHA.fullmatch(value) is not None


def _allowed(path):
    p=PurePosixPath(path)
    return (not p.is_absolute() and '..' not in p.parts and str(p)==path and
            (path in _ROOT_FILES or (len(p.parts)>1 and p.parts[0] in _ROOTS and
             (p.suffix in _EXTENSIONS or path=='preview/static/OFL-NotoSansSC.txt') and '__pycache__' not in p.parts)))


def source_files(root):
    """Hash explicit project source/document types; never traverse data or secrets."""
    root=Path(root).resolve(strict=True);files=[]
    for name in _ROOT_FILES:
        path=root/name
        if path.is_symlink():raise ValueError(f'Symbolic source file: {path}')
        if path.is_file():files.append(path)
    for name in _ROOTS:
        folder=root/name
        if folder.is_symlink():raise ValueError(f'Symbolic source root: {folder}')
        if folder.is_dir():
            for path in folder.rglob('*'):
                relative=path.relative_to(root).as_posix()
                if path.is_symlink():raise ValueError(f'Symbolic source path: {path}')
                if _allowed(relative):
                    if path.is_file():files.append(path)
    if not files:raise ValueError('No source files found')
    result={}
    for path in sorted(files):
        with path.open('rb') as stream:result[path.relative_to(root).as_posix()]=hashlib.file_digest(stream,'sha256').hexdigest()
    return result


def create_receipt(source_root,image_id,config,manifest_sha256):
    if not isinstance(image_id,str) or not image_id.startswith('sha256:') or not _valid_hash(image_id[7:]):
        raise ValueError('Runtime image ID must be sha256:<digest>')
    if not _valid_hash(manifest_sha256):raise ValueError('Invalid prepared manifest SHA')
    if not isinstance(config,dict):raise ValueError('Configuration must be an object')
    sources=source_files(source_root)
    return dict(schema_version=1,source_tree_sha256=_sha(_canonical(sources)),source_files=sources,
                runtime_image_id=image_id,config_sha256=_sha(_canonical(config)),data_manifest_sha256=manifest_sha256)


def _validate_receipt(value,expected_data_sha):
    keys={'schema_version','source_tree_sha256','source_files','runtime_image_id','config_sha256','data_manifest_sha256'}
    if not isinstance(value,dict) or set(value)!=keys or type(value['schema_version']) is not int or value['schema_version']!=1:
        raise ValueError('Malformed freeze receipt schema')
    if not all(_valid_hash(value[k]) for k in ('source_tree_sha256','config_sha256','data_manifest_sha256')):
        raise ValueError('Malformed freeze receipt SHA')
    image=value['runtime_image_id']
    if not isinstance(image,str) or not image.startswith('sha256:') or not _valid_hash(image[7:]):
        raise ValueError('Malformed runtime image ID')
    sources=value['source_files']
    if not isinstance(sources,dict) or not sources or any(not isinstance(k,str) or not _allowed(k) or not _valid_hash(v) for k,v in sources.items()):
        raise ValueError('Malformed freeze receipt source files')
    if _sha(_canonical(sources))!=value['source_tree_sha256']:raise ValueError('Freeze receipt source digest differs')
    if not _valid_hash(expected_data_sha) or value['data_manifest_sha256']!=expected_data_sha:
        raise ValueError('Freeze receipt data SHA differs')
    return value


def write_receipt(path,receipt):
    path=Path(path)
    if not isinstance(receipt,dict):raise ValueError('Malformed freeze receipt')
    _validate_receipt(receipt,receipt.get('data_manifest_sha256'))
    name=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.freeze-',delete=False) as stream:
            name=stream.name
            json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False)
        os.link(name,path)
    finally:
        if name is not None:os.unlink(name)


def read_receipt(path,expected_data_sha,*,source_root=None,image_id=None,config=None):
    path=Path(path)
    if path.is_symlink():raise ValueError('Freeze receipt must not be a symlink')
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate freeze receipt key')
            result[k]=v
        return result
    try:value=json.loads(path.read_text(),object_pairs_hook=unique)
    except (json.JSONDecodeError,UnicodeDecodeError) as e:raise ValueError('Malformed freeze receipt JSON') from e
    _validate_receipt(value,expected_data_sha)
    if source_root is not None and source_files(source_root)!=value['source_files']:raise ValueError('Current source tree differs from freeze receipt')
    if image_id is not None and image_id!=value['runtime_image_id']:raise ValueError('Runtime image differs from freeze receipt')
    if config is not None and _sha(_canonical(config))!=value['config_sha256']:raise ValueError('Configuration differs from freeze receipt')
    return value


class ManifestIdentity:
    """Hash the prepared manifest once and check its identity on each resume."""
    def __init__(self,path):
        path=Path(path)
        if path.is_symlink():raise ValueError('Prepared manifest must not be a symlink')
        self.path=path.resolve(strict=True)
        with self.path.open('rb') as stream:self.sha256=hashlib.file_digest(stream,'sha256').hexdigest()

    def verify_checkpoint(self,checkpoint,receipt_path=None):
        if not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('optimized_training'),dict) or checkpoint['optimized_training'].get('prepared_manifest_sha256')!=self.sha256:
            raise ValueError('Checkpoint prepared manifest SHA differs')
        if checkpoint['optimized_training'].get('freeze_receipt') is not None and receipt_path is None:
            raise ValueError('Frozen checkpoint requires its freeze receipt')
        if receipt_path is not None:
            receipt=read_receipt(receipt_path,self.sha256)
            if checkpoint['optimized_training'].get('freeze_receipt')!=receipt:
                raise ValueError('Checkpoint freeze receipt differs')
