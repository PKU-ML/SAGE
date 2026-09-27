"""CPU regression against the real Cube HDF5, including terminal endpoints."""

import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sage.runtime.lewm import load_swm_dataset, WindowSpec, target_action_chunk
from sage.runtime.pair_frames import configure_pair_windows
from sage.train.cube_generator import SubgoalPairDataset
from sage.train.cube_action_prior import VariableActionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    path = parser.parse_args().dataset
    ds = load_swm_dataset(path, frameskip=5, num_steps=8, keys_to_load=["pixels", "action", "observation"])
    configure_pair_windows(ds, 3, 15)
    base = np.r_[0, np.cumsum(ds.lengths)[:-1]]
    assert len(ds.clip_indices) == 1760000
    assert (0, 175) in ds.clip_indices
    rows = []
    for ep, start, horizon, tau in [(0, 20, 150, 25), (42, 175, 15, 15),
                                    (42, 170, 20, 20), (9999, 165, 25, 25)]:
        spec = WindowSpec(0, ep, ep, start, 0)
        kwargs = dict(context_len=3, frameskip=5)
        gen = SubgoalPairDataset(ds, [spec], [0], [horizon], [tau], **kwargs)[0]
        cached = SubgoalPairDataset(ds, [spec], [0], [horizon], [tau], episode_base=base, **kwargs)[0]
        row = dict(spec_index=0, action_offset=tau, goal_offset=horizon)
        prior = VariableActionDataset(ds, [spec], [row], **kwargs)[0]
        prior_cached = VariableActionDataset(ds, [spec], [row], episode_base=base, **kwargs)[0]
        expected_history = base[ep] + start + np.array([0, 5, 10])
        for example in (cached, prior_cached):
            np.testing.assert_array_equal(example["history_frame_indices"], expected_history)
            torch.testing.assert_close(example["observation"][2], ds._load_slice(ep, start+10, start+11)["observation"][0])
        assert int(cached["goal_frame_index"]) == base[ep]+start+10+horizon
        assert int(cached["subgoal_frame_index"]) == base[ep]+start+10+tau
        assert int(prior_cached["far_goal_frame_index"]) == base[ep]+start+10+horizon
        assert int(prior_cached["goal_frame_index"]) == base[ep]+start+10+tau
        expected_far = ds._load_slice(ep, start+10+horizon, start+11+horizon)["pixels"]
        expected_local = ds._load_slice(ep, start+10+tau, start+11+tau)["pixels"]
        torch.testing.assert_close(gen["goal_pixels"], expected_far, rtol=0, atol=0)
        torch.testing.assert_close(gen["subgoal_pixels"], expected_local, rtol=0, atol=0)
        torch.testing.assert_close(prior["far_goal_pixels"], expected_far, rtol=0, atol=0)
        torch.testing.assert_close(prior["goal_pixels"], expected_local, rtol=0, atol=0)
        chunk = target_action_chunk({"action": prior["action"][None]}, 3, 5)[0]
        truth = ds._load_slice(ep, start+10, start+10+tau)["action"].reshape(tau//5, 25)
        torch.testing.assert_close(chunk[:tau//5], truth, rtol=0, atol=0)
        torch.testing.assert_close(prior_cached["action"], prior["action"], rtol=0, atol=0)
        assert torch.isfinite(chunk).all()
        assert torch.count_nonzero(chunk[tau//5:]) == 0
        rows.append(dict(episode=ep, current=start+10, horizon=horizon, tau=tau,
                         history=(expected_history-base[ep]).tolist(), endpoint=start+10+tau))
    print(json.dumps(dict(status="PASS", all_windows=len(ds.clip_indices), checks=rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
