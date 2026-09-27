"""Merge verified anchor shards without dropping or replacing locked queries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.native_data import resolve_path
from sage.provenance import sha256_file


def merge(queries, indices, output):
    manifest = json.loads(queries.read_text())
    digest = sha256_file(queries)
    expected = {
        hashlib.sha256(json.dumps({'suite': manifest['suite'], 'query': row},
                                 sort_keys=True).encode()).hexdigest()
        for row in manifest['queries']
    }
    anchors = {}
    for index_path in indices:
        index = json.loads(index_path.read_text())
        if index.get('schema') != 1 or index.get('query_manifest_sha256') != digest:
            raise ValueError(f'Anchor manifest mismatch: {index_path}')
        for key, entry in index['anchors'].items():
            if key not in expected or not entry.get('source_prefix_proprio_exact'):
                raise ValueError(f'Unexpected or uncertified anchor: {key}')
            path = resolve_path(index_path.parent, entry['path']).resolve()
            if sha256_file(path) != entry['sha256']:
                raise ValueError(f'Anchor checksum mismatch: {path}')
            if key in anchors:
                raise ValueError(f'Duplicate anchor: {key}')
            # Keep the entire merged tree relocatable; references may not escape it.
            relative = path.relative_to(output.parent.resolve())
            anchors[key] = dict(entry, path=relative.as_posix())
    if set(anchors) != expected:
        raise ValueError(f'Incomplete anchor set: {len(anchors)}/{len(expected)}')
    result = {'schema': 1, 'query_manifest_sha256': digest, 'anchors': anchors}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    return len(anchors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', required=True, type=Path)
    parser.add_argument('--indices', required=True, nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({'anchors': merge(args.queries, args.indices, args.out),
                      'index': str(args.out)}))


if __name__ == '__main__':
    main()
