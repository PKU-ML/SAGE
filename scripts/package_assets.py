"""Pack only registered, SHA-verified inference assets from a local mirror."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sage.assets import safe_relative
from sage.provenance import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suites', nargs='+', required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    suites = json.loads((ROOT/'configs/suites.json').read_text())
    registry = json.loads((ROOT/'configs/assets.json').read_text())
    names = set()
    for suite_name in args.suites:
        suite = suites[suite_name]
        names.update(suite[k] for k in ('world_model', 'generator', 'prior', 'far_prior') if k in suite)
        names.update(suite.get('companion_assets', []))
    selected = {name: registry[name] for name in sorted(names)}
    output = args.out.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        p.error('Choose a new archive outside the source tree')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name+'.partial')
    with tarfile.open(pending, 'x') as archive:
        manifest = {}
        for name, entry in selected.items():
            filename = safe_relative(entry.get('filename', name)).as_posix()
            source = (ROOT/safe_relative(entry['bundled_file'])) if 'bundled_file' in entry else args.source_dir/filename
            if sha256_file(source) != entry['sha256']:
                raise ValueError(f'Asset checksum mismatch: {filename}')
            info = tarfile.TarInfo(filename)
            info.size = source.stat().st_size
            info.mode = 0o644
            with source.open('rb') as handle:
                archive.addfile(info, handle)
            manifest[filename] = {'sha256': entry['sha256'], 'bytes': info.size}
        content = (json.dumps({'suites': args.suites, 'assets': manifest}, indent=2)+'\n').encode()
        info = tarfile.TarInfo('ASSET_MANIFEST.json')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    with tarfile.open(pending) as archive:
        for name, entry in manifest.items():
            digest = hashlib.sha256()
            with archive.extractfile(name) as handle:
                for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Archived asset checksum mismatch: {name}')
    pending.replace(output)
    digest = sha256_file(output)
    output.with_suffix(output.suffix+'.sha256').write_text(f'{digest}  {output.name}\n')
    print(json.dumps({'archive': str(output), 'assets': len(manifest), 'sha256': digest,
                      'bytes': output.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
