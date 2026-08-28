import torch

from sage.models.action_prior import PushtVariableTransformerGoalPrior
from sage.models.subgoal import PushtSubgoalPrior


def test_generator_and_prior_shapes():
    batch, tokens, latent_dim = 2, 4, 16
    history = torch.randn(batch, tokens, latent_dim)
    far_goal = torch.randn(batch, tokens, latent_dim)
    lowdim = torch.randn(batch, 6)
    goal_offset = torch.tensor([75.0, 125.0])
    option_duration = torch.tensor([15.0, 25.0])

    generator = PushtSubgoalPrior(
        latent_dim=latent_dim,
        lowdim_dim=6,
        hidden_dim=32,
        num_heads=4,
        depth=2,
        pooling="decoder",
        predict_residual_from="goal",
    )
    local_goal = generator(
        history, far_goal, lowdim, goal_offset, option_duration
    )["prediction"]
    assert local_goal.shape == far_goal.shape

    prior = PushtVariableTransformerGoalPrior(
        latent_dim=latent_dim,
        lowdim_dim=6,
        action_dim=10,
        max_plan_horizon=5,
        hidden_dim=32,
        num_heads=4,
        depth=2,
        num_modes=3,
    )
    output = prior(
        history,
        local_goal,
        lowdim,
        action_horizon=3,
        far_goal_latents=far_goal,
        goal_offset_steps=goal_offset,
        subgoal_offset_steps=option_duration,
    )
    assert output["means"].shape == (batch, 3, 5, 10)
    samples = prior.sample(
        history,
        local_goal,
        lowdim,
        7,
        action_horizon=3,
        far_goal_latents=far_goal,
        goal_offset_steps=goal_offset,
        subgoal_offset_steps=option_duration,
    )
    assert samples.shape == (batch, 7, 3, 10)
