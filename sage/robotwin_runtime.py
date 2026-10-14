"""Install the content-locked RoboTwin source and simulator assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from sage.assets import install as download_assets, safe_relative
from sage.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def extract_verified(archive_path, destination, expected):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            relative = safe_relative(member.name)
            if member.name in seen or member.name not in expected or not member.isfile():
                raise ValueError(f'Unexpected archive member: {member.name}')
            record = expected[member.name]
            if member.size != record['bytes']:
                raise ValueError(f'Unexpected member size: {member.name}')
            target = destination / relative
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f'Member escapes runtime root: {member.name}')
            if target.exists():
                if not target.is_file() or sha256_file(target) != record['sha256']:
                    raise ValueError(f'Existing file has wrong checksum: {member.name}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.partial', delete=False) as output:
                    pending = Path(output.name)
                    try:
                        with archive.extractfile(member) as source:
                            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                                digest.update(block)
                                output.write(block)
                        if digest.hexdigest() != record['sha256']:
                            raise ValueError(f'Archive checksum mismatch: {member.name}')
                        output.close()
                        pending.replace(target)
                    finally:
                        output.close()
                        pending.unlink(missing_ok=True)
            seen.add(member.name)
            if len(seen) % 500 == 0:
                print(f'verified {len(seen)}/{len(expected)} runtime files', flush=True)
    if seen != set(expected):
        raise ValueError(f'Archive is missing {len(set(expected) - seen)} registered files')
    return len(seen)


def verify_tree(destination, expected):
    destination = Path(destination).resolve()
    for name, record in expected.items():
        target = destination / safe_relative(name)
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f'File escapes runtime root: {name}')
        if not target.is_file() or target.stat().st_size != record['bytes'] or sha256_file(target) != record['sha256']:
            raise ValueError(f'Runtime checksum mismatch: {name}')
    return len(expected)


def download_archive(registry, downloads, source_dir=None):
    downloads = Path(downloads)
    for name, record in registry.items():
        if not record.get('parts'):
            download_assets({name: record}, downloads, source_dir=source_dir, project_root=ROOT)
            continue
        archive = downloads / safe_relative(record['filename'])
        if archive.is_file():
            if archive.stat().st_size != record['bytes'] or sha256_file(archive) != record['sha256']:
                raise ValueError(f'Existing archive checksum mismatch: {name}')
            continue
        parts = {part['filename']: part for part in record['parts']}
        if len(parts) != len(record['parts']):
            raise ValueError('Duplicate archive parts')
        download_assets(parts, downloads, source_dir=source_dir, project_root=ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=archive.parent, suffix='.partial', delete=False) as output:
            pending = Path(output.name)
            try:
                for part in record['parts']:
                    with (downloads / safe_relative(part['filename'])).open('rb') as source:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                output.close()
                if pending.stat().st_size != record['bytes'] or sha256_file(pending) != record['sha256']:
                    raise ValueError(f'Assembled archive checksum mismatch: {name}')
                pending.replace(archive)
            finally:
                output.close()
                pending.unlink(missing_ok=True)


def install_registry(folder, kind, destination, downloads, source_dir=None, verify_only=False):
    inventory_name = 'source_files.json' if kind == 'source' else 'assets.json'
    registry_name = 'source_archive.json' if kind == 'source' else 'asset_archive.json'
    expected = json.loads((folder / inventory_name).read_text())
    bootstrap = json.loads((folder / 'bootstrap_assets.json').read_text()) if kind == 'assets' else {}
    if verify_only:
        return verify_tree(destination, dict(expected, **bootstrap))
    registry = json.loads((folder / registry_name).read_text())
    download_archive(registry, downloads, source_dir=source_dir)
    count = 0
    for record in registry.values():
        archive_path = downloads / safe_relative(record['filename'])
        count += extract_verified(archive_path, destination, expected)
    if bootstrap:
        download_assets(bootstrap, destination, project_root=ROOT)
        count += verify_tree(destination, bootstrap)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--downloads', type=Path, default=Path('downloads'))
    parser.add_argument('--source-dir', type=Path, help='Optional mirror containing both archives')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'runtime/robotwin'
    counts = {}
    for kind in ('source', 'assets'):
        counts[kind] = install_registry(folder, kind, args.out, args.downloads,
                                       args.source_dir, args.verify_only)
    print(json.dumps({'files_verified': counts, 'runtime_root': str(args.out.resolve())}, indent=2))


if __name__ == '__main__':
    main()
