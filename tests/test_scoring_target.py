from types import SimpleNamespace
import os
from unittest.mock import patch

import torch

from sage.eval.cube import CubeSAGEModel
from sage.eval.pusht import SAGECostModel


def test_final_scoring_changes_only_cost_payload():
    final = torch.randn(2, 1, 4)
    actions = torch.randn(2, 3, 5, 10)
    observed = []

    def cost(info, candidates):
        observed.append(info)
        assert candidates is actions
        torch.testing.assert_close(info["goal_emb"], final)
        assert "_private" not in info
        return candidates.square().mean((-1, -2))

    wm = SimpleNamespace(extra_encoders={}, get_cost=cost)
    for cls, goal_method in ((SAGECostModel, "_final_goal_latents"), (CubeSAGEModel, "_goal")):
        model = SimpleNamespace(lewm=wm, score_final_goal=True)
        setattr(model, goal_method, lambda info: final)
        info = {"pixels": torch.zeros(2, 3, 3, 8, 8), "_private": 1}
        # This test isolates scoring-target conversion; chunk invariance has
        # a separate test with a fully bound model and proposal RNG.
        with patch.dict(os.environ, {'SAGE_ENV_BATCH': '2'}):
            result = cls.get_cost(model, info, actions)
        assert result.shape == (2, 3)
        assert "goal_emb" not in info
    assert len(observed) == 2
