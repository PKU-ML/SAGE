"""Episode-indexed RoboTwin reader backed by its native compressed JPEG HDF5.

The reader keeps RoboTwin's source files in place and materializes only a
small JSON split manifest.  This avoids inflating three 224px camera streams
into hundreds of GB of decoded HDF5 while preserving the same episode/window
contract used by the tail-aligned LEWM trainer.
"""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from stable_worldmodel.data.dataset import Dataset


VIEW_MAP = {
    "pixels": "vision/cam_head/colors",
    "left_wrist_pixels": "vision/cam_left_wrist/colors",
    "right_wrist_pixels": "vision/cam_right_wrist/colors",
}
_PARTS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)


def _joint_vector(source: h5py.File, group: str) -> np.ndarray:
    direct = f"{group}/joint_states"
    if direct in source:
        values = np.asarray(source[direct][:], dtype=np.float32)
    else:
        values = np.concatenate(
            [np.asarray(source[f"{group}/{part}"][:], dtype=np.float32) for part in _PARTS], axis=1
        )
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"{source.filename}: expected 14-D {group}, got {values.shape}")
    return values


def _decode_rgb(value: np.bytes_) -> np.ndarray:
    with Image.open(io.BytesIO(bytes(value))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


class RobotWinCompressedDataset(Dataset):
    """Native RoboTwin episodes with the standard LEWM Dataset API."""

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform=None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        max_open_files: int = 16,
        **_: object,
    ) -> None:
        self.manifest_path = Path(name)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "episodes" not in payload:
            raise ValueError(f"{name} is not a RoboTwin LEWM split manifest")
        self.source_root = Path(payload["source_root"])
        self.episodes = payload["episodes"]
        self._keys = keys_to_load or [*VIEW_MAP, "action", "proprio"]
        unsupported = set(self._keys) - {*VIEW_MAP, "action", "proprio", "task_id", "action_mode"}
        if unsupported:
            raise KeyError(f"Unsupported RoboTwin columns: {sorted(unsupported)}")
        self.max_open_files = int(max_open_files)
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._column_cache: dict[str, np.ndarray] = {}
        lengths = np.asarray([int(entry["frames"]) for entry in self.episodes], dtype=np.int32)
        offsets = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def _handle(self, episode: int) -> h5py.File:
        handle = self._handles.pop(episode, None)
        if handle is None:
            path = self.source_root / self.episodes[episode]["source"]
            handle = h5py.File(path, "r", swmr=True)
        self._handles[episode] = handle
        while len(self._handles) > self.max_open_files:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        source = self._handle(ep_idx)
        item: dict[str, torch.Tensor] = {}
        for key in self._keys:
            if key in VIEW_MAP:
                encoded = source[VIEW_MAP[key]][start:end:self.frameskip]
                frames = np.stack([_decode_rgb(value) for value in encoded], axis=0)
                item[key] = torch.from_numpy(frames).permute(0, 3, 1, 2)
            elif key == "action":
                item[key] = torch.from_numpy(_joint_vector(source, "action")[start:end])
            elif key == "proprio":
                item[key] = torch.from_numpy(_joint_vector(source, "state")[start:end:self.frameskip])
            else:
                length = len(range(start, end, self.frameskip))
                item[key] = torch.zeros((length, 1), dtype=torch.int64)
        return self.transform(item) if self.transform else item

    def get_col_data(self, col: str) -> np.ndarray:
        if col in self._column_cache:
            return self._column_cache[col]
        if col in ("task_id", "action_mode"):
            dtype = np.int16 if col == "task_id" else np.int8
            values = np.zeros((int(self.lengths.sum()), 1), dtype=dtype)
        elif col in ("action", "proprio"):
            values = np.concatenate(
                [_joint_vector(self._handle(index), "action" if col == "action" else "state") for index in range(len(self.episodes))],
                axis=0,
            )
        else:
            raise KeyError(f"Cannot materialize compressed image column {col!r}")
        self._column_cache[col] = values
        return values

    def get_dim(self, col: str) -> int:
        if col in ("action", "proprio"):
            return 14
        if col in ("task_id", "action_mode"):
            return 1
        raise KeyError(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        indices = np.atleast_1d(row_idx).astype(np.int64)
        episode_ids = np.searchsorted(self.offsets[1:], indices, side="right")
        local = indices - self.offsets[episode_ids]
        result = {}
        for col in ("action", "proprio", "task_id", "action_mode"):
            values = self.get_col_data(col)
            result[col] = values[indices]
        result["episode_id"] = episode_ids
        result["episode_step"] = local
        return result


__all__ = ["RobotWinCompressedDataset"]
