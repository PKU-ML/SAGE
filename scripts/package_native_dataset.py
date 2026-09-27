"""Create a self-contained, checksum-verified archive of a paper dataset split."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sage.native_data import resolve_path
from sage.provenance import sha256_file


def registered_files(suite, split):
    if split == 'evaluation':
        path = ROOT / 'data/manifests' / suite / 'assets.json'
        catalog = json.loads(path.read_text())
        return {row.get('filename', name): {'bytes': row['bytes'], 'sha256': row['sha256']}
                for name, row in catalog.items()}
    directory = ROOT / 'data/training' / suite
    if suite == 'robotwin_a2b':
        rows = json.loads((directory / f'{split}.json').read_text())['episodes']
        field = 'source'
    else:
        rows = json.loads((directory / f'{split}_shards.json').read_text())['shards']
        field = 'path'
    return {row[field]: {'bytes': row['bytes'], 'sha256': row['sha256']} for row in rows}


def package(files, data_root, output, suite, split):
    output = Path(output).resolve()
    if output.suffix != '.tar':
        raise ValueError('Use a .tar output path (input HDF5/JPEG is already compressed)')
    if output.exists():
        raise FileExistsError(output)
    if not files:
        raise ValueError('An empty dataset cannot be packaged')
    sources = {}
    for name, record in files.items():
        path = resolve_path(data_root, name)
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Missing or size-mismatched source: {name}')
        sources[name] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + '.partial')
    with tarfile.open(pending, 'x') as archive:
        for index, (name, source) in enumerate(sorted(sources.items())):
            if sha256_file(source) != files[name]['sha256']:
                raise ValueError(f'Source checksum mismatch: {name}')
            # Explicit file members prevent server-local hardlinks from escaping
            # into an archive that would require the original filesystem.
            info = tarfile.TarInfo(name)
            info.size = source.stat().st_size
            info.mode = 0o644
            with source.open('rb') as handle:
                archive.addfile(info, handle)
            if (index + 1) % 100 == 0:
                print(json.dumps(dict(archived=index + 1, total=len(files))), flush=True)
        metadata = json.dumps(dict(suite=suite, split=split, files=files), indent=2).encode()
        info = tarfile.TarInfo('DATASET_MANIFEST.json')
        info.size = len(metadata)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(metadata))
    with tarfile.open(pending, 'r') as archive:
        for name, record in files.items():
            member = archive.getmember(name)
            if not member.isfile() or member.size != record['bytes']:
                raise ValueError(f'Non-portable or invalid archived member: {name}')
            digest = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256']:
                raise ValueError(f'Archived checksum mismatch: {name}')
    pending.replace(output)
    digest = sha256_file(output)
    output.with_suffix('.tar.sha256').write_text(f'{digest}  {output.name}\n')
    result = dict(suite=suite, split=split, files=len(files), bytes=output.stat().st_size,
                  sha256=digest, member_checksums_verified=True)
    output.with_suffix('.tar.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['libero_scene2', 'libero_caddy', 'robotwin_a2b'], required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'evaluation'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(registered_files(args.suite, args.split), args.data_root,
                             args.out, args.suite, args.split), indent=2))


if __name__ == '__main__':
    main()
