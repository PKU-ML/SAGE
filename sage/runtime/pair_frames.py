"""Raw-frame indexing for variable-horizon supervised pairs."""

import numpy as np
import torch


def configure_pair_windows(dataset, context_len, min_offset):
    # A window needs history and its own endpoint, not the largest far goal.
    dataset.span = (int(context_len) - 1) * dataset.frameskip + int(min_offset) + 1
    dataset.clip_indices = [
        (ep, start)
        for ep, length in enumerate(dataset.lengths)
        for start in range(max(0, int(length) - dataset.span + 1))
    ]
    return dataset


def pair_context(dataset, spec, context_len, action_offset):
    fs = int(dataset.frameskip)
    stop = int(spec.start) + (int(context_len) - 1) * fs + int(action_offset)
    if stop >= int(dataset.lengths[spec.local_episode]):
        raise IndexError("Pair endpoint is outside its source episode")
    item = dataset._load_slice(spec.local_episode, spec.start, stop)
    for key, value in item.items():
        if key == "action":
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite action inside the supervised chunk")
            tokens = value.reshape(-1, fs * value.shape[-1])
            # Padding is outside tau and never contributes to the masked loss.
            padded = tokens.new_zeros((dataset.num_steps - 1, tokens.shape[-1]))
            padded[:len(tokens)] = tokens
            item[key] = padded
        elif torch.is_tensor(value):
            item[key] = value[:int(context_len)]
    return item


def raw_goal_pixels(dataset, spec, raw_offset):
    frame = int(spec.start) + int(raw_offset)
    if frame >= int(dataset.lengths[spec.local_episode]):
        raise IndexError("Goal frame is outside its source episode")
    return dataset._load_slice(spec.local_episode, frame, frame + 1)["pixels"]


def validate_full_frame_cache(cache, dataset):
    counts = np.diff(np.append(cache.episode_base, len(cache.latents)))
    if not np.array_equal(counts, np.asarray(dataset.lengths)):
        raise ValueError("Variable-horizon training requires every raw frame, including terminal frames")
