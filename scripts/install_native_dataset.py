"""Install a registered dataset archive without server-local dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from package_native_dataset import registered_files
from sage.assets import safe_relative
from sage.provenance import sha256_file


def install(archive_path, destination, suite, split):
    expected = registered_files(suite, split)
    destination = Path(destination).resolve()
    with tarfile.open(archive_path, 'r:') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or set(names) != set(expected) | {'DATASET_MANIFEST.json'}:
            raise ValueError('Archive members do not match the registered dataset')
        for member in members:
            safe_relative(member.name)
            if not member.isfile():
                raise ValueError(f'Only regular files are supported: {member.name}')
        metadata_member = archive.getmember('DATASET_MANIFEST.json')
        if metadata_member.size > 16 * 1024 * 1024:
            raise ValueError('Oversized dataset manifest')
        with archive.extractfile(metadata_member) as handle:
            metadata = json.load(handle)
        if metadata != dict(suite=suite, split=split, files=expected):
            raise ValueError('Dataset manifest does not match the release registry')
        for name, record in expected.items():
            target = destination / safe_relative(name)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f'Destination escapes data root: {name}')
            member = archive.getmember(name)
            if member.size != record['bytes']:
                raise ValueError(f'Unexpected member size: {name}')
            if target.exists():
                if not target.is_file() or sha256_file(target) != record['sha256']:
                    raise ValueError(f'Existing data has the wrong checksum: {name}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=target.name + '.', suffix='.partial',
                delete=False,
            ) as output:
                pending = Path(output.name)
                try:
                    with archive.extractfile(member) as source:
                        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                            output.write(chunk)
                            digest.update(chunk)
                    if digest.hexdigest() != record['sha256']:
                        raise ValueError(f'Archive checksum mismatch: {name}')
                    output.close()
                    pending.replace(target)
                finally:
                    output.close()
                    pending.unlink(missing_ok=True)
    return dict(suite=suite, split=split, files=len(expected), verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--suite', choices=['libero_scene2', 'libero_caddy', 'robotwin_a2b'], required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'evaluation'], required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.archive, args.out, args.suite, args.split), indent=2))


if __name__ == '__main__':
    main()
