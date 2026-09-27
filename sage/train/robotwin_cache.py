"""Encode all three camera frames at raw stride 1 in manifest order."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sage.native_models import load_world_model
from sage.provenance import sha256_file
from sage.train.robotwin_pack import source_path

VIEWS = dict(pixels='vision/cam_head/colors', left_wrist_pixels='vision/cam_left_wrist/colors',
             right_wrist_pixels='vision/cam_right_wrist/colors')


def preprocess(frames, size=224):
    # Historical cache normalizes before a bilinear resize without antialiasing.
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225))])
    value = transform(frames)
    return F.interpolate(value.float(), size=(size, size), mode='bilinear', align_corners=False)


class FrameDataset(Dataset):
    def __init__(self, manifest, root, size=224):
        self.episodes = json.loads(Path(manifest).read_text())['episodes']
        self.root, self.size = Path(root), size
        self.lengths = np.array([int(ep['frames']) for ep in self.episodes], dtype=np.int64)
        if not len(self.lengths) or (self.lengths <= 0).any():
            raise ValueError('Empty or invalid episode lengths')
        self.ends = self.lengths.cumsum()
        self.handles = OrderedDict()
        for ep, length in zip(self.episodes, self.lengths):
            path = source_path(self.root, ep['source'])
            if ep.get('sha256') and sha256_file(path) != ep['sha256']:
                raise ValueError(f"Episode checksum mismatch: {ep['source']}")
            with h5py.File(path, 'r') as f:
                if any(len(f[column]) != length for column in VIEWS.values()):
                    raise ValueError('Manifest/camera frame count mismatch')

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        state = dict(self.__dict__)
        state['handles'] = OrderedDict()
        return state

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.ends, index, side='right'))
        local = index - (int(self.ends[episode - 1]) if episode else 0)
        handle = self.handles.pop(episode, None)
        if handle is None:
            handle = h5py.File(source_path(self.root, self.episodes[episode]['source']), 'r', swmr=True)
        self.handles[episode] = handle
        while len(self.handles) > 16:
            _, old = self.handles.popitem(last=False)
            old.close()
        result = {}
        for name, column in VIEWS.items():
            with Image.open(io.BytesIO(bytes(handle[column][local]))) as image:
                array = np.array(image.convert('RGB'), dtype=np.uint8)
            frames = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            result[name] = preprocess(frames, self.size)
        return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--world-model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.json').exists():
        raise FileExistsError(args.out)
    model = load_world_model(args.world_model, args.device)
    if list(model.view_keys) != list(VIEWS):
        raise ValueError('This cache requires the paper three-view RoboTwin world model')
    dataset = FrameDataset(args.manifest, args.data_root)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache, written = None, 0
    dtype = next(model.parameters()).dtype
    for batch in loader:
        inputs = {key: value.to(device=args.device, dtype=dtype, non_blocking=True)
                  for key, value in batch.items()}
        values = model.encode(inputs)['emb'].squeeze(1).float().cpu().numpy().astype(np.float16)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite encoder output')
        if cache is None:
            cache = np.lib.format.open_memmap(args.out, mode='w+', dtype=np.float16,
                shape=(len(dataset), values.shape[-1]))
        cache[written:written + len(values)] = values
        written += len(values)
        if written % (args.batch_size * 100) == 0:
            print(f'cached={written}/{len(dataset)}', flush=True)
    if cache is None or written != len(dataset):
        raise RuntimeError('Incomplete frame cache')
    cache.flush()
    metadata = dict(complete=True, frames=written, latent_dim=cache.shape[-1], cache_stride=1,
        views=list(VIEWS), manifest_sha256=sha256_file(args.manifest),
        checkpoint_sha256=sha256_file(args.world_model),
        model_config_sha256=sha256_file(args.world_model.parent / 'config.json'),
        cache_sha256=sha256_file(args.out))
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
