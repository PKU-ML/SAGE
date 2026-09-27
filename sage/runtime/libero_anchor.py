"""Correct cross-CPU prefix drift at a certified LIBERO handoff.

An anchor is not a standalone simulator snapshot. Replay to the requested time
first so task bookkeeping and sensor schedules are initialized identically.
No pickle, flattened-state fallback, or cached source images are used.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

DERIVED_FIELDS = (
    'xpos', 'xquat', 'xmat', 'xipos', 'ximat', 'geom_xpos', 'geom_xmat',
    'site_xpos', 'site_xmat', 'cam_xpos', 'cam_xmat', 'light_xpos', 'light_xdir',
)
CLOCK_FIELDS = ('_time_since_last_sample', '_current_delay', '_sampled')


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name in sorted(dir(model)):
        if name.startswith('_'):
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
            digest.update(name.encode())
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def observe_without_advancing_clock(env):
    clocks = {name: {key: getattr(obs, key) for key in CLOCK_FIELDS}
              for name, obs in env._observables.items()}
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    try:
        # robosuite's force_update also advances each observable's clock.
        return env._get_observations(force_update=True)
    finally:
        for name, values in clocks.items():
            for key, value in values.items():
                setattr(env._observables[name], key, value)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def save_anchor(env, path, context):
    import mujoco

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(env.sim.model._model, spec), np.float64)
    mujoco.mj_getState(env.sim.model._model, env.sim.data._data, state, spec)
    arrays = {'integration': state}
    for name in DERIVED_FIELDS:
        arrays['derived__' + name] = getattr(env.sim.data, name).copy()
    scalars = {}
    for prefix, obj in (('controller', env.robots[0].controller), ('gripper', env.robots[0].gripper)):
        for name, value in vars(obj).items():
            key = prefix + '__' + name
            if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
                arrays[key] = value.copy()
            elif isinstance(value, (int, float, bool)):
                scalars[key] = value
    metadata = {'schema': 1, 'context': context, 'mujoco': mujoco.__version__,
                'model_sha256': model_fingerprint(env.sim.model), 'spec': spec,
                'timestep': int(env.timestep), 'cur_time': float(env.cur_time),
                'scalars': scalars}
    arrays['metadata'] = np.asarray(json.dumps(metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def restore_anchor(env, path, context, expected_sha256):
    import mujoco
    from sage.provenance import sha256_file

    if sha256_file(path) != expected_sha256:
        raise ValueError('Handoff anchor SHA256 mismatch')
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        if meta['schema'] != 1 or meta['context'] != context:
            raise ValueError('Handoff anchor belongs to a different query')
        if meta['mujoco'] != mujoco.__version__ or meta['model_sha256'] != model_fingerprint(env.sim.model):
            raise ValueError('Handoff anchor simulator/model mismatch')
        if int(env.timestep) != meta['timestep'] or abs(env.cur_time - meta['cur_time']) > 1e-9:
            raise ValueError('Replay the complete prefix before applying a handoff anchor')
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        if meta['spec'] != spec or archive['integration'].shape != (mujoco.mj_stateSize(env.sim.model._model, spec),):
            raise ValueError('Integration-state contract mismatch')
        objects = {'controller': env.robots[0].controller, 'gripper': env.robots[0].gripper}
        for key in archive.files:
            if key in ('metadata', 'integration'):
                continue
            prefix, name = key.split('__', 1)
            obj = env.sim.data if prefix == 'derived' and name in DERIVED_FIELDS else objects[prefix]
            current = getattr(obj, name)
            if current.shape != archive[key].shape or current.dtype != archive[key].dtype:
                raise ValueError(f'Anchor array layout mismatch: {key}')
        mujoco.mj_setState(env.sim.model._model, env.sim.data._data, archive['integration'], spec)
        for key in archive.files:
            if key in ('metadata', 'integration'):
                continue
            prefix, name = key.split('__', 1)
            if prefix == 'derived':
                getattr(env.sim.data, name)[...] = archive[key]
            else:
                setattr(objects[prefix], name, archive[key].copy())
        for key, value in meta['scalars'].items():
            prefix, name = key.split('__', 1)
            if type(getattr(objects[prefix], name)) is not type(value):
                raise ValueError(f'Anchor scalar type mismatch: {key}')
            setattr(objects[prefix], name, value)
    # Preserve pre-integration render transforms. An extra sim.forward() would
    # change the image's temporal alignment with the collection observations.
    return observe_without_advancing_clock(env)


def apply_indexed_anchor(env, index_path, suite, query):
    from sage.native_data import resolve_path

    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    if index['schema'] != 1:
        raise ValueError('Unsupported anchor index')
    context = {'suite': suite, 'query': query}
    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    entry = index['anchors'][key]
    path = resolve_path(index_path.parent, entry['path'])
    return restore_anchor(env, path, context, entry['sha256'])
