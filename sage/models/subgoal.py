"""Goal-conditioned latent subgoal prior for Push-T.

This module is intentionally small: it predicts the next reachable LeWM latent
subgoal from the current history, a farther goal, low-dimensional state, and a
horizon token. The online planner can later use the predicted subgoal as the
goal for a local action prior / CEM call.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class PushtSubgoalPrior(nn.Module):
    """Predict a local future latent from history and a farther goal latent."""

    def __init__(
        self,
        latent_dim: int,
        lowdim_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 3,
        max_goal_offset: int = 200,
        predict_residual_from: str = "goal",
        pooling: str = "attention",
        goal_condition_mode: str = "goal",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.lowdim_dim = int(lowdim_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.depth = int(depth)
        self.max_goal_offset = int(max_goal_offset)
        self.predict_residual_from = str(predict_residual_from)
        self.pooling = str(pooling)
        self.goal_condition_mode = str(goal_condition_mode)

        if self.predict_residual_from not in {"goal", "current", "zero"}:
            raise ValueError(
                "predict_residual_from must be one of goal/current/zero"
            )
        if self.goal_condition_mode not in {"goal", "zero", "current"}:
            raise ValueError(
                "goal_condition_mode must be one of goal/zero/current"
            )
        if self.pooling not in {"attention", "mean", "decoder"}:
            raise ValueError("pooling must be attention, mean, or decoder")

        self.visual_norm = nn.LayerNorm(self.latent_dim)
        self.visual_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.history_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.target_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        if self.pooling == "attention":
            self.pool_query = nn.Parameter(
                torch.zeros(1, 1, self.hidden_dim)
            )
            self.pool = nn.MultiheadAttention(
                self.hidden_dim,
                int(num_heads),
                batch_first=True,
            )
        else:
            self.register_parameter("pool_query", None)
            self.pool = None

        if self.pooling == "decoder":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=4 * self.hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=max(int(depth), 1),
            )
            self.offset_token_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.decoder = None
            self.offset_token_proj = None

        if self.lowdim_dim > 0:
            self.lowdim_encoder = nn.Sequential(
                nn.LayerNorm(self.lowdim_dim),
                nn.Linear(self.lowdim_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.lowdim_encoder = None

        self.offset_encoder = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        cond_dim = 3 * self.hidden_dim
        if self.lowdim_encoder is not None:
            cond_dim += self.hidden_dim
        cond_layers: list[nn.Module] = [
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, self.hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(max(self.depth - 1, 0)):
            cond_layers.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                ]
            )
        self.cond_trunk = nn.Sequential(*cond_layers)

        self.token_block = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(self.hidden_dim, self.latent_dim)

        for parameter in (
            self.history_type,
            self.goal_type,
            self.target_type,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        if self.pool_query is not None:
            nn.init.trunc_normal_(self.pool_query, std=0.02)
        nn.init.zeros_(self.delta_head.bias)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool is None:
            return tokens.mean(dim=1)
        query = self.pool_query.expand(tokens.size(0), -1, -1)
        pooled, _ = self.pool(query, tokens, tokens, need_weights=False)
        return pooled.squeeze(1)

    def _offset_features(
        self,
        goal_offset_steps: torch.Tensor,
        subgoal_offset_steps: torch.Tensor,
    ) -> torch.Tensor:
        goal = goal_offset_steps.float().view(-1, 1)
        subgoal = subgoal_offset_steps.float().view(-1, 1)
        denom = max(float(self.max_goal_offset), 1.0)
        ratio = subgoal / torch.clamp(goal, min=1.0)
        features = torch.cat([goal / denom, subgoal / denom, ratio], dim=-1)
        return self.offset_encoder(features)

    def _base_latents(
        self,
        history_latents: torch.Tensor,
        goal_latents: torch.Tensor,
    ) -> torch.Tensor:
        if self.predict_residual_from == "goal":
            return goal_latents
        if self.predict_residual_from == "zero":
            return torch.zeros_like(goal_latents)
        target_tokens = goal_latents.size(1)
        if history_latents.size(1) < target_tokens:
            raise ValueError(
                "history_latents has fewer tokens than goal_latents"
            )
        return history_latents[:, -target_tokens:]

    def _condition_goal_latents(
        self,
        history_latents: torch.Tensor,
        goal_latents: torch.Tensor,
    ) -> torch.Tensor:
        if self.goal_condition_mode == "goal":
            return goal_latents
        if self.goal_condition_mode == "zero":
            return torch.zeros_like(goal_latents)
        target_tokens = goal_latents.size(1)
        if history_latents.size(1) < target_tokens:
            raise ValueError(
                "history_latents has fewer tokens than goal_latents"
            )
        return history_latents[:, -target_tokens:]

    def forward(
        self,
        history_latents: torch.Tensor,
        goal_latents: torch.Tensor,
        lowdim: torch.Tensor,
        goal_offset_steps: torch.Tensor,
        subgoal_offset_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        conditioned_goal_latents = self._condition_goal_latents(
            history_latents,
            goal_latents,
        )
        history = (
            self.visual_proj(self.visual_norm(history_latents))
            + self.history_type
        )
        goal = (
            self.visual_proj(self.visual_norm(conditioned_goal_latents))
            + self.goal_type
        )
        offset_feat = self._offset_features(
            goal_offset_steps,
            subgoal_offset_steps,
        )
        if self.decoder is not None:
            memory_parts = [
                history,
                goal,
                self.offset_token_proj(offset_feat)[:, None],
            ]
            if self.lowdim_encoder is not None:
                memory_parts.append(self.lowdim_encoder(lowdim)[:, None])
            memory = torch.cat(memory_parts, dim=1)
            query = goal + self.target_type
            decoded = self.decoder(query, memory)
            token_hidden = self.token_block(torch.cat([query, decoded], dim=-1))
            delta = self.delta_head(token_hidden)
            base = self._base_latents(history_latents, conditioned_goal_latents)
            prediction = base + delta
            return {"prediction": prediction, "delta": delta, "base": base}

        history_pool = self._pool(history)
        goal_pool = self._pool(goal)
        cond_parts = [history_pool, goal_pool, offset_feat]
        if self.lowdim_encoder is not None:
            cond_parts.append(self.lowdim_encoder(lowdim))
        cond = self.cond_trunk(torch.cat(cond_parts, dim=-1))
        cond_tokens = cond[:, None].expand(-1, goal.size(1), -1)
        token_hidden = self.token_block(
            torch.cat([goal + self.target_type, cond_tokens], dim=-1)
        )
        delta = self.delta_head(token_hidden)
        base = self._base_latents(history_latents, conditioned_goal_latents)
        prediction = base + delta
        return {"prediction": prediction, "delta": delta, "base": base}

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        target_latents: torch.Tensor,
        *,
        cosine_weight: float = 0.1,
        smooth_l1_beta: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        prediction = outputs["prediction"]
        reg = F.smooth_l1_loss(
            prediction,
            target_latents,
            beta=float(smooth_l1_beta),
        )
        pred_flat = prediction.flatten(1)
        target_flat = target_latents.flatten(1)
        cosine = 1.0 - F.cosine_similarity(
            pred_flat,
            target_flat,
            dim=-1,
        ).mean()
        total = reg + float(cosine_weight) * cosine
        return {"loss": total, "smooth_l1": reg, "cosine": cosine}


def save_subgoal_prior(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_subgoal_prior(path: str | Path, device: torch.device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("prior_type") != "pusht_subgoal_prior_v1":
        raise ValueError(f"Not a PushT subgoal prior checkpoint: {path}")
    model = PushtSubgoalPrior(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    stats = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in checkpoint.get("stats", {}).items()
    }
    return model, stats, checkpoint
