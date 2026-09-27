"""Build a source-only candidate archive after the release audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('configs', 'data', 'runtime', 'sage', 'scripts', 'stable_worldmodel', 'tests')
FILES = ('.gitattributes', '.gitignore', 'DATA.md', 'environment.yml', 'LICENSE',
         'NATIVE.md', 'pyproject.toml', 'README.md', 'REPRODUCING.md', 'THIRD_PARTY.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    output = args.out.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        p.error('Choose a new archive path outside the source tree')
    subprocess.run([sys.executable, str(ROOT/'scripts/audit_release.py')], cwd=ROOT, check=True)
    files = [ROOT/name for name in FILES]
    for folder in DIRECTORIES:
        files.extend(path for path in (ROOT/folder).rglob('*') if path.is_file()
                     and not any(part in ('__pycache__', '.pytest_cache') for part in path.parts)
                     and path.suffix not in ('.pyc', '.pyo'))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.partial')
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        checksums = {}
        for path in sorted(files):
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError(f'External or symlinked source: {path}')
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            checksums[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr('SAGE/'+relative, content)
        archive.writestr('SAGE/SOURCE_MANIFEST.json', json.dumps({
            'kind': 'source_release_candidate', 'weights_and_datasets_included': False,
            'files': checksums}, indent=2)+'\n')
    with zipfile.ZipFile(temporary) as archive:
        for name, expected in checksums.items():
            if hashlib.sha256(archive.read('SAGE/'+name)).hexdigest() != expected:
                raise RuntimeError(f'Archive verification failed: {name}')
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(f'{digest}  {output.name}\n')
    print(json.dumps({'archive': str(output), 'files': len(checksums), 'sha256': digest,
                      'weights_and_datasets_included': False}, indent=2))


if __name__ == '__main__':
    main()
