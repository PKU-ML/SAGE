import io
import json

import h5py
import numpy as np
from PIL import Image
import pytest
import torch

from sage.train.robotwin_cache import FrameDataset, VIEWS, preprocess


def test_frame_order_rgb_and_camera_layout(tmp_path):
    episodes = []
    decoded = []
    for episode, length in enumerate((2, 3)):
        frames = []
        for step in range(length):
            rgb = np.full((8, 9, 3), 20 + 30 * episode + step, np.uint8)
            stream = io.BytesIO()
            Image.fromarray(rgb).save(stream, format='JPEG')
            frames.append(stream.getvalue())
            with Image.open(io.BytesIO(frames[-1])) as image:
                decoded.append(torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1)[None])
        with h5py.File(tmp_path / f'{episode}.h5', 'w') as source:
            for column in VIEWS.values():
                source[column] = np.array(frames, dtype=f'S{max(map(len, frames))}')
        episodes.append(dict(source=f'{episode}.h5', frames=length))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(episodes=episodes)))
    dataset = FrameDataset(manifest, tmp_path, size=16)
    assert len(dataset) == 5
    for index in range(5):
        result = dataset[index]
        assert list(result) == list(VIEWS)
        for frame in result.values():
            torch.testing.assert_close(frame, preprocess(decoded[index], 16), rtol=0, atol=0)
    for handle in dataset.handles.values():
        handle.close()
    episodes[0]['sha256'] = '0' * 64
    manifest.write_text(json.dumps(dict(episodes=episodes)))
    with pytest.raises(ValueError, match='checksum'):
        FrameDataset(manifest, tmp_path)
