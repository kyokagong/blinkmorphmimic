"""Create a reference-only preview release; historical policy/data identities stay fixed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from preview.server import ROBOTS, sha256


def migrate(new_root, old_catalog):
    root = Path(new_root).resolve(strict=True)
    manifest = root / 'manifest.json'
    data = json.loads(manifest.read_text())
    if manifest.is_symlink() or data.get('complete') is not True or not data.get('clips'):
        raise ValueError('A complete nonempty unified reference manifest is required')
    reference = dict(path=str(manifest), sha256=sha256(manifest))
    old = json.loads(Path(old_catalog).read_text())
    wanted = [f'joint-{interface}-{iteration}' for iteration in (7000, 6000, 5000) for interface in ('motion', 'human')]
    selected = [row for row in old if row.get('id') in wanted]
    by_id = {row['id']: row for row in selected}
    if len(selected) != len(by_id) or set(by_id) != set(wanted):
        raise ValueError('Require exactly one Motion/Human joint release at 5000, 6000 and 7000')
    templates = {name: json.loads((root / name).read_text()) for name in ('windows.json', 'flip-preview.json')}
    planned, releases = {}, []
    for name in wanted:
        row = copy.deepcopy(by_id[name])
        training = row['training_manifest']
        training_path = Path(training['path'])
        if training_path != training_path.resolve(strict=True) or sha256(training_path) != training['sha256']:
            raise ValueError('Historical training manifest SHA differs')
        subsets = {robot: sha256(training_path.parent / robot / 'train.yaml') for robot in ROBOTS}
        row.update(data_root=str(root), reference_manifest=reference)
        for template, field, suffix in (('windows.json', 'recovery_windows', 'windows'),
                                         ('flip-preview.json', 'flip_sources', 'flips')):
            proof = copy.deepcopy(templates[template])
            proof.update(training_manifest_sha256=training['sha256'],
                         reference_manifest_sha256=reference['sha256'], subset_sha256=subsets)
            path = root / f"preview-{row['parent_iteration']}-{suffix}.json"
            content = json.dumps(proof, indent=2, ensure_ascii=False) + '\n'
            if path in planned and planned[path] != content:
                raise ValueError('Motion/Human releases disagree on their historical training identity')
            planned[path] = content
            # Hash the exact bytes that will be published, without modifying a historical proof.
            row[field] = dict(path=str(path), sha256=hashlib.sha256(content.encode('utf-8')).hexdigest())
        releases.append(row)
    output = root / 'releases.json'
    planned[output] = json.dumps(releases, indent=2, ensure_ascii=False) + '\n'
    for path, content in planned.items():
        if path.is_symlink() or (path.exists() and path.read_text() != content):
            raise FileExistsError(f'Refusing to replace reviewed preview metadata: {path}')
    for path, content in planned.items():
        if not path.exists():
            with path.open('x') as stream:
                stream.write(content)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--old-catalog', type=Path, required=True)
    args = parser.parse_args()
    print(migrate(args.new_root, args.old_catalog))


if __name__ == '__main__':
    main()
