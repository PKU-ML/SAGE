"""Gaussian CEM must retain the historical per-environment RNG order."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from sage.eval.pusht import GaussianCEM, PriorInitializedCEM, SAGECostModel
from stable_worldmodel.solver.cem import CEMSolver


class QuadraticCost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def get_cost(self, info, actions):
        target = info['target'][..., None, None]
        return (actions - target).square().mean(dim=(-1, -2))


class GoalPreprocessingTest(unittest.TestCase):
    def test_normalized_goal_above_two_is_not_normalized_again(self):
        model = SampleCost()
        goal = torch.full((1, 8, 1, 3, 6, 6), 2.64)
        with patch('sage.eval.pusht.image_batch_to_lewm') as preprocess, \
             patch('sage.eval.pusht.encode_lewm_context', side_effect=lambda wm, x: x):
            actual = model._final_goal_latents({'goal': goal})
        preprocess.assert_not_called()
        self.assertTrue(torch.equal(actual, goal[:, 0]))


class GaussianOrderTest(unittest.TestCase):
    def solver(self):
        solver = GaussianCEM(QuadraticCost(), candidates=8, rounds=3,
                             elites=3, seed=42, device=torch.device('cpu'))
        solver.configure(action_space=Box(-1, 1, shape=(2, 2), dtype=np.float32),
                         n_envs=2, config=SimpleNamespace(horizon=3, action_block=1))
        return solver

    def test_batched_matches_sequential_calls(self):
        info = {'target': torch.tensor([0.2, -0.8])}
        actual = self.solver().solve(info)
        reference = self.solver()
        expected = [reference.solve({'target': info['target'][i:i+1]}) for i in range(2)]
        self.assertTrue(torch.equal(actual['actions'], torch.cat([x['actions'] for x in expected])))
        self.assertEqual(actual['costs'], [x['costs'][0] for x in expected])

    def test_no_warm_start(self):
        with self.assertRaisesRegex(ValueError, 'forbid warm starts'):
            self.solver().solve({'target': torch.zeros(2)}, init_action=torch.zeros(2, 3, 2))

    def test_bf16_matches_original_zero_gaussian_solver(self):
        old_model = QuadraticCost().to(dtype=torch.bfloat16)
        new_model = QuadraticCost().to(dtype=torch.bfloat16)
        old = CEMSolver(old_model, batch_size=1, num_samples=300,
                        n_steps=30, topk=30, seed=32)
        new = GaussianCEM(new_model, candidates=300, rounds=30,
                          elites=30, seed=32, device=torch.device('cpu'))
        config = SimpleNamespace(horizon=5, action_block=5)
        space = Box(-1, 1, shape=(2, 2), dtype=np.float32)
        for solver in (old, new):
            solver.configure(action_space=space, n_envs=2, config=config)
        info = {'target': torch.tensor([0.2, -0.4], dtype=torch.bfloat16)}
        for _ in range(2):
            expected, actual = old.solve(info), new.solve(info)
            self.assertTrue(torch.equal(expected['actions'], actual['actions']))


class SampleCost(SAGECostModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.lewm = QuadraticCost()
        self.action_prior = SimpleNamespace(action_dim=2)

    def _local_goal_latents(self, info):
        return info['target']

    def sample_candidates(self, info, *, num_samples, action_horizon, generator):
        return torch.randn(len(info['target']), num_samples, action_horizon, 2,
                           generator=generator)


class PriorChunkOrderTest(unittest.TestCase):
    def run_solver(self, batch):
        solver = PriorInitializedCEM(SampleCost(), candidates=8, rounds=3,
            elites=3, seed=42, device=torch.device('cpu'))
        solver.configure(action_space=Box(-1, 1, shape=(2, 2), dtype=np.float32),
            n_envs=2, config=SimpleNamespace(horizon=3, action_block=1))
        with patch.dict('os.environ', {'SAGE_ENV_BATCH': str(batch)}):
            return solver.solve({'target': torch.tensor([0.2, -0.8])})

    def test_cost_chunking_preserves_samples_and_solution(self):
        small, large = self.run_solver(1), self.run_solver(2)
        self.assertTrue(torch.equal(small['actions'], large['actions']))
        self.assertEqual(small['costs'], large['costs'])


if __name__ == '__main__':
    unittest.main()
