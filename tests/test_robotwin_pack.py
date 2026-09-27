import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest

from sage.train.robotwin_pack import pack, raw_actions, source_path, ACTION_PARTS


def test_stride_one_cache_and_train_statistics():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        episodes = []
        arrays = []
        for i, length in enumerate((7, 4)):
            action = np.arange(length * 14, dtype=np.float32).reshape(length, 14) + i
            arrays.append(action)
            with h5py.File(root / f"{i}.h5", "w") as f:
                f["action/joint_states"] = action
            episodes.append(dict(source=f"{i}.h5", frames=length))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(dict(episodes=episodes)))
        flat = np.arange(11 * 3, dtype=np.float32).reshape(11, 3)
        np.save(root / "flat.npy", flat)
        stats = root / "stats.npz"
        result = pack(manifest, root / "flat.npy", root, root / "train", stats, True)
        all_actions = np.concatenate(arrays).astype(np.float64)
        mean = all_actions.mean(0).astype(np.float32)
        std = np.sqrt(np.maximum((all_actions ** 2).mean(0) - all_actions.mean(0) ** 2, 1e-6)).astype(np.float32)
        with np.load(stats) as saved:
            np.testing.assert_array_equal(saved["mean"], mean)
            np.testing.assert_array_equal(saved["std"], std)
        packed = np.load(root / "train/actions.npy")
        np.testing.assert_array_equal(packed[0], ((arrays[0] - mean) / std).astype(np.float16))
        np.testing.assert_array_equal(packed[1, 4:], 0)
        np.testing.assert_array_equal(np.load(root / "train/latents.npy")[1, :4], flat[7:].astype(np.float16))
        before = stats.read_bytes()
        pack(manifest, root / "flat.npy", root, root / "val", stats)
        assert stats.read_bytes() == before
        assert result["frames"] == 11 and result["cache_stride"] == 1
        with pytest.raises(FileExistsError):
            pack(manifest, root / "flat.npy", root, root / "train", stats)


def test_split_actions_and_path_boundary(tmp_path):
    with h5py.File(tmp_path / "episode.h5", "w") as f:
        for part, dim in zip(ACTION_PARTS, (6, 1, 6, 1)):
            f[f"action/{part}"] = np.ones((3, dim), np.float32)
    assert raw_actions(tmp_path / "episode.h5").shape == (3, 14)
    with pytest.raises(ValueError):
        source_path(tmp_path, "../outside.h5")
