from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from sage.eval.cube import GaussianCEM as CubeGaussianCEM
from sage.eval.cube import PriorTopMode as CubePriorTopMode
from sage.eval.pusht import GaussianCEM as PushTGaussianCEM
from sage.eval.pusht import PriorTopMode as PushTPriorTopMode
from stable_worldmodel.solver.solver import Solver


class PushTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.action_dim = 10

    def get_cost(self, info, actions):
        del info
        return actions.float().square().mean(dim=(-1, -2))

    def top_candidate(self, info, *, action_horizon):
        del info
        return torch.zeros(2, action_horizon, self.action_dim)


class CubeModel(PushTModel):
    device = torch.device("cpu")
    dtype = torch.float32

    def top_prior(self, info, horizon):
        return self.top_candidate(info, action_horizon=horizon)


def setup():
    info = {"x": torch.zeros(2, 1)}
    action_space = Box(-1, 1, shape=(2, 2), dtype=np.float32)
    config = SimpleNamespace(horizon=5, action_block=5)
    return info, action_space, config


def test_true_gaussian_cem_shapes():
    info, action_space, config = setup()
    solvers = (
        PushTGaussianCEM(
            PushTModel(),
            candidates=8,
            rounds=2,
            elites=2,
            seed=1,
            device=torch.device("cpu"),
        ),
        CubeGaussianCEM(
            CubeModel(),
            candidates=8,
            rounds=2,
            elites=2,
            seed=1,
            score_batch_size=4,
        ),
    )
    for solver in solvers:
        solver.configure(action_space=action_space, n_envs=2, config=config)
        assert solver(info)["actions"].shape == (2, 5, 10)


def test_prior_top_shapes():
    info, action_space, config = setup()
    for solver in (PushTPriorTopMode(PushTModel()), CubePriorTopMode(CubeModel())):
        solver.configure(action_space=action_space, n_envs=2, config=config)
        assert isinstance(solver, Solver)
        assert solver(info)["actions"].shape == (2, 5, 10)


def test_generator_goal_is_cached_before_candidate_expansion():
    class GeneratorModel(PushTModel):
        generator = object()

        def __init__(self):
            super().__init__()
            self.cached = False

        def _local_goal_latents(self, info):
            assert "_proposal_pixels_raw" in info
            self.cached = True

        def get_cost(self, info, actions):
            assert self.cached
            return super().get_cost(info, actions)

    _, action_space, config = setup()
    model = GeneratorModel()
    solver = PushTGaussianCEM(
        model,
        candidates=8,
        rounds=1,
        elites=2,
        seed=1,
        device=torch.device("cpu"),
    )
    solver.configure(action_space=action_space, n_envs=2, config=config)
    info = {
        "x": torch.zeros(2, 1),
        "_proposal_pixels_raw": np.zeros((2, 3, 8, 8, 3), dtype=np.uint8),
    }
    assert solver(info)["actions"].shape == (2, 5, 10)
