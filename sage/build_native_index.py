"""Build a relocatable LIBERO training index from an ordered shard manifest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from sage.provenance import sha256_file


def build_index(manifest: Path, data_root: Path, output: Path):
    payload = json.loads(manifest.read_text())
    records = payload['shards']
    if not records:
        raise ValueError('The shard manifest is empty')
    root = data_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    sources, lengths, schema, total = [], [], None, 0
    for record in records:
        name = Path(record['path'])
        path = (root / name).resolve()
        if name.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f'Shard escapes data root: {name}')
        if path in [s[0] for s in sources]:
            raise ValueError(f'Duplicate shard: {name}')
        if sha256_file(path) != record['sha256']:
            raise ValueError(f'Shard SHA mismatch: {name}')
        with h5py.File(path, 'r') as handle:
            count = len(handle['pixels'])
            ep_len = np.asarray(handle['ep_len'], dtype=np.int64)
            offsets = np.asarray(handle['ep_offset'], dtype=np.int64)
            expected = np.r_[0, np.cumsum(ep_len[:-1])]
            if np.any(ep_len <= 0) or int(ep_len.sum()) != count or not np.array_equal(offsets, expected):
                raise ValueError(f'Invalid episode boundaries: {name}')
            current = {key: (ds.shape[1:], ds.dtype.str) for key, ds in handle.items()
                       if isinstance(ds, h5py.Dataset) and key not in ('ep_len', 'ep_offset')}
            if any(len(handle[key]) != count for key in current):
                raise ValueError(f'Frame columns have unequal lengths: {name}')
            if schema is not None and current != schema:
                raise ValueError(f'Shard schema mismatch: {name}')
            schema = current
            lengths.append(ep_len)
            sources.append((path, total, count))
            total += count
    output.parent.mkdir(parents=True, exist_ok=True)
    # v110 bounds keep the VDS readable by the pinned HDF5 runtimes.
    with h5py.File(output, 'x', libver=('v110', 'v110')) as target:
        for key, (tail, dtype) in schema.items():
            layout = h5py.VirtualLayout(shape=(total, *tail), dtype=dtype)
            for path, start, count in sources:
                relative = Path(os.path.relpath(path, output.parent)).as_posix()
                source = h5py.VirtualSource(relative, key, shape=(count, *tail))
                layout[start:start + count] = source
            target.create_virtual_dataset(key, layout)
        ep_len = np.concatenate(lengths)
        target.create_dataset('ep_len', data=ep_len.astype(np.int32))
        target.create_dataset('ep_offset', data=np.r_[0, np.cumsum(ep_len[:-1])])
        target.attrs.update(complete=True, virtual=True, episodes=len(ep_len),
                            transitions=total, source_shards=len(sources),
                            manifest_sha256=sha256_file(manifest))
    # HDF5 can silently return fill values for missing VDS sources.
    with h5py.File(output, 'r') as index:
        for path, start, count in sources:
            with h5py.File(path, 'r') as shard:
                for key in schema:
                    np.testing.assert_array_equal(index[key][start], shard[key][0])
                    np.testing.assert_array_equal(index[key][start + count - 1], shard[key][-1])
    return {'episodes': len(ep_len), 'frames': total, 'shards': len(sources)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_index(args.manifest, args.data_root, args.output)))


if __name__ == '__main__':
    main()
