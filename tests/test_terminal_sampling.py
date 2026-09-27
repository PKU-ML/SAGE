"""Boundary contracts of the published terminal-conditioned training recipe."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from sage.models.action_prior import PushtVariableTransformerGoalPrior
from sage.train.libero_generator import (
    build_terminal_staircase_specs, Scene2CachedSubgoalPairDataset,
)
from sage.train.libero_prior import forward_prior


class TerminalSamplingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.path = root / 'episodes.h5'
        with h5py.File(self.path, 'w') as f:
            f['ep_len'] = [46, 61]
            f['ep_offset'] = [0, 46]
            f['pixels'] = np.zeros((107, 2, 2, 3), dtype=np.uint8)
            f['action_mode'] = np.zeros(107, dtype=np.int16)
            f['action_mode'][46 + 20] = 2
        self.cache = root / 'latents.npy'
        np.save(self.cache, np.arange(107, dtype=np.float16)[:, None].repeat(4, axis=1))
        self.cache.with_suffix('.json').write_text(json.dumps({
            'complete': True, 'frame_count': 107, 'latent_dim': 4}))

    def test_no_cross_episode_or_random_action_targets(self):
        specs = build_terminal_staircase_specs(self.path, 15, None,
            include_direct_final=True, allowed_local_action_modes=[0, 1])
        for episode, start, horizon, duration, terminal in specs:
            self.assertEqual(horizon, terminal - start)
            self.assertLessEqual(start + duration, terminal)
            if episode == 1:
                self.assertFalse(start <= 20 < start + duration)
        self.assertTrue(any(np.array_equal(row, [0, 30, 15, 15, 45]) for row in specs))

    def test_generator_excludes_direct_goal_and_repeats_short_history(self):
        specs = build_terminal_staircase_specs(self.path, 15, None,
                                              include_direct_final=False)
        self.assertTrue(np.all(specs[:, 2] >= 30))
        dataset = Scene2CachedSubgoalPairDataset(self.path,
            np.asarray([[1, 4, 56, 15, 60]]), self.cache, 15)
        row = dataset[0]
        self.assertEqual(row['history_latents'][:, 0].tolist(), [50., 50., 50.])
        self.assertEqual(row['goal_latents'][0].item(), 106.)
        self.assertEqual(row['subgoal_latents'][0].item(), 65.)

    def test_gmm_loss_backward(self):
        torch.manual_seed(7)
        model = PushtVariableTransformerGoalPrior(latent_dim=4, lowdim_dim=0,
            action_dim=35, max_plan_horizon=3, hidden_dim=16, num_heads=4,
            depth=1, num_modes=8, dropout=0., max_goal_offset=100)
        prepared = (torch.randn(2, 3, 4), torch.randn(2, 1, 4),
            torch.randn(2, 1, 4), torch.randn(2, 3, 35),
            torch.tensor([30., 60.]), torch.tensor([15., 15.]))
        result = forward_prior(model, prepared, use_bf16=False)
        loss = model.nll(result, prepared[3]) + 0.05 * model.best_mode_l1(result, prepared[3])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
