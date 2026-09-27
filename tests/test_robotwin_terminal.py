import numpy as np
import torch

from sage.train.robotwin_terminal import PairDataset, build_terminal_staircase_specs


def test_terminal_windows_keep_tail_and_all_integer_horizons():
    specs = build_terminal_staircase_specs([34, 49], 15, None, False, 42)
    expected = [(e, t, length - 1 - t, length - 1)
                for e, length in enumerate([34, 49]) for t in range(length - 30)]
    assert sorted(map(tuple, specs.tolist())) == expected
    np.testing.assert_array_equal(specs,
        build_terminal_staircase_specs([34, 49], 15, None, False, 42))
    prior = build_terminal_staircase_specs([34], 15, None, True, 42)
    assert sorted(prior[:, 2]) == list(range(15, 34))
    assert len(build_terminal_staircase_specs([15], 15, None, True, 42)) == 0


def test_history_and_action_alignment(tmp_path):
    latents = np.arange(40 * 3, dtype=np.float16).reshape(1, 40, 3)
    actions = np.arange(40 * 14, dtype=np.float16).reshape(1, 40, 14)
    np.save(tmp_path / 'latents.npy', latents)
    np.save(tmp_path / 'actions.npy', actions)
    specs = np.array([[0, 1, 38, 39], [0, 3, 36, 39], [0, 24, 15, 39]])
    data = PairDataset(tmp_path, specs)
    np.testing.assert_array_equal(data[0]['history'], latents[0, [1, 1, 1]])
    np.testing.assert_array_equal(data[1]['history'], latents[0, [1, 2, 3]])
    np.testing.assert_array_equal(data[2]['actions'], actions[0, 24:39])
    torch.testing.assert_close(data[2]['local_goal'], data[2]['far_goal'])
