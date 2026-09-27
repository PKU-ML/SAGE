"""Export prefix anchors on the collection-compatible LIBERO runtime.

This does not alter the query bank. Online evaluation must independently verify
freshly rendered handoff images after loading these anchors.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.native_data import resolve_path
from sage.provenance import sha256_file
from sage.runtime.libero import configure
from sage.runtime.libero_anchor import save_anchor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--libero-root', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--query-start', type=int, default=0)
    p.add_argument('--num-queries', type=int, default=1)
    p.add_argument('--physics-only', action='store_true')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    contract = json.loads((Path(__file__).resolve().parents[1] /
                           'runtime/native_contracts.json').read_text())['libero']['packages']
    for package in ('numpy', 'mujoco', 'robosuite'):
        actual = importlib.metadata.version(package)
        if actual != contract[package]:
            raise RuntimeError(f'Collection dependency mismatch: {package}={actual}; '
                               f'expected {contract[package]}. Use the pinned export environment.')
    if args.query_start < 0 or args.num_queries <= 0:
        p.error('Select a nonempty, nonnegative query slice')
    if not 0 <= args.shard_index < args.num_shards:
        p.error('Require 0 <= shard-index < num-shards')
    manifest = json.loads(args.queries.read_text())
    suite = manifest['suite']
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/native/libero_environment.json').read_text())
    spec = cfg['suites'][suite]
    rows = manifest['queries'][args.query_start:args.query_start + args.num_queries]
    if len(rows) != args.num_queries:
        p.error('Insufficient locked queries')
    if args.physics_only:
        # No GL context is created; avoid eager EGL driver enumeration on import.
        os.environ['MUJOCO_GL'] = 'glx'
        os.environ['PYOPENGL_PLATFORM'] = 'glx'
    configure(args.libero_root, args.out_dir / 'libero_config', args.data_root)
    from libero.libero import get_libero_path
    from libero.libero.envs import TASK_MAPPING
    index_path = args.out_dir / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        'schema': 1, 'query_manifest_sha256': sha256_file(args.queries), 'anchors': {}}
    if index['schema'] != 1 or index['query_manifest_sha256'] != sha256_file(args.queries):
        raise ValueError('Anchor directory belongs to another manifest')
    random.seed(42); np.random.seed(42)
    groups = {}
    for query in rows:
        context = {'suite': suite, 'query': query}
        key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        if key in index['anchors']:
            entry = index['anchors'][key]
            if sha256_file(resolve_path(args.out_dir, entry['path'])) != entry['sha256']:
                raise ValueError('Existing anchor checksum mismatch')
        groups.setdefault((query['source_shard'], query['source_episode']), []).append((query, context, key))
    for group_index, group in enumerate(groups.values()):
        if group_index % args.num_shards != args.shard_index:
            continue
        group = [item for item in group if item[2] not in index['anchors']]
        if not group:
            continue
        if len({(q['task'], q['seed'], q.get('settle_steps', 0)) for q, _, _ in group}) != 1:
            raise ValueError('Conflicting environment metadata for one source episode')
        query = group[0][0]
        by_step = {}
        for item in group:
            by_step.setdefault(int(item[0]['start']), []).append(item)
        kwargs = dict(cfg['env_kwargs'])
        if args.physics_only:
            kwargs.update(has_offscreen_renderer=False, use_camera_obs=False)
        kwargs['bddl_file_name'] = str(next(Path(get_libero_path('bddl_files')).rglob(spec['tasks'][query['task']])))
        env = TASK_MAPPING[spec['problem_name']](**kwargs)
        try:
            with h5py.File(resolve_path(args.data_root, query['source_shard']), 'r') as data:
                episode = int(query['source_episode'])
                offset = int(data['ep_offset'][episode])
                for previous in range(episode):
                    env.seed(int(data['episode_seed'][int(data['ep_offset'][previous])]))
                    env.reset()
                env.seed(int(query['seed']))
                obs = env.reset()
                for _ in range(int(query.get('settle_steps', 0))):
                    obs, _, _, _ = env.step(np.zeros(7, np.float32))
                last_step = max(by_step)
                for step in range(last_step + 1):
                    state = np.concatenate([obs['robot0_joint_pos'], obs['robot0_gripper_qpos']]).astype(np.float32)
                    if not np.array_equal(state, data['proprio'][offset + step]):
                        error = float(np.max(np.abs(state - data['proprio'][offset + step])))
                        raise RuntimeError(f'Collection replay differs: query={query.get("query_index")}, '
                                           f'step={step}, max_proprio_error={error}; '
                                           'do not export from this runtime')
                    if not args.physics_only:
                        for field, sensor in (('pixels', 'agentview_image'), ('wrist_pixels', 'robot0_eye_in_hand_image')):
                            if not np.array_equal(obs[sensor], data[field][offset + step]):
                                raise RuntimeError(f'Collection rendering differs at step {step}: {field}')
                    for query_at_step, context, key in by_step.get(step, []):
                        path = args.out_dir / f'{key}.npz'
                        save_anchor(env, path, context)
                        index['anchors'][key] = {'path': path.name, 'sha256': sha256_file(path),
                            'source_prefix_proprio_exact': True, 'source_prefix_images_checked': not args.physics_only}
                        temporary = index_path.with_suffix('.partial')
                        temporary.write_text(json.dumps(index, indent=2) + '\n')
                        temporary.replace(index_path)
                        print(json.dumps({'anchors': len(index['anchors']), 'query_index': query_at_step.get('query_index')}), flush=True)
                    if step < last_step:
                        obs, _, _, _ = env.step(np.asarray(data['action'][offset + step], np.float32))
        finally:
            env.close()


if __name__ == '__main__':
    main()
