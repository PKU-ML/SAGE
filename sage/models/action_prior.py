"""Variable-length Push-T action priors."""
from __future__ import annotations
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn


def _flatten_goal_latents(goal_latents: torch.Tensor) -> torch.Tensor:
    if goal_latents.ndim == 4:
        return goal_latents.flatten(1, 2)
    return goal_latents


def _time_features(values: torch.Tensor, max_value: int) -> torch.Tensor:
    values = values.float().view(-1, 1)
    x = values / max(float(max_value), 1.0)
    return torch.cat([x, torch.sin(math.pi * x), torch.cos(math.pi * x), torch.sin(2.0 * math.pi * x), torch.cos(2.0 * math.pi * x)], dim=-1)


def _sample_modes(probabilities: torch.Tensor, num_samples: int, generator=None) -> torch.Tensor:
    if os.environ.get("PUSHT_STRATIFIED_GMM", "0") == "1":
        batch, num_modes = probabilities.shape
        count = int(num_samples)
        if count < num_modes:
            return probabilities.topk(count, dim=-1).indices
        # Reserve one exact candidate for every component, then allocate the
        # remaining budget according to the learned mixture probabilities.
        remaining = count - num_modes
        expected = probabilities.float() * float(remaining)
        allocations = expected.floor().long()
        leftovers = remaining - allocations.sum(dim=-1)
        fractions = expected - allocations.float()
        rows = []
        for row in range(batch):
            extra = int(leftovers[row].item())
            row_alloc = allocations[row].clone()
            if extra > 0:
                indices = fractions[row].topk(extra).indices
                row_alloc[indices] += 1
            mode_ids = [torch.arange(num_modes, device=probabilities.device)]
            mode_ids.append(
                torch.repeat_interleave(
                    torch.arange(num_modes, device=probabilities.device),
                    row_alloc,
                )
            )
            rows.append(torch.cat(mode_ids, dim=0))
        return torch.stack(rows, dim=0)
    if os.environ.get("PUSHT_CPU_MULTINOMIAL", "0") != "1":
        return torch.multinomial(
            probabilities,
            num_samples=int(num_samples),
            replacement=True,
            generator=generator,
        )
    cpu_gen = None
    if generator is not None:
        seed = torch.randint(
            0,
            2**31 - 1,
            (1,),
            device=probabilities.device,
            generator=generator,
        ).item()
        cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.multinomial(
        probabilities.detach().float().cpu(),
        num_samples=int(num_samples),
        replacement=True,
        generator=cpu_gen,
    ).to(probabilities.device)


def _sample_noise(shape, *, dtype, device, generator, exact_prefix: int = 0):
    if os.environ.get("PUSHT_ANTITHETIC_NOISE", "0") != "1":
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)
    batch, count, *tail = shape
    exact_prefix = min(max(int(exact_prefix), 0), count)
    random_count = count - exact_prefix
    pair_count = (random_count + 1) // 2
    seed = torch.randint(
        0,
        2**31 - 1,
        (1,),
        device=device,
        generator=generator,
    ).item()
    cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    base = torch.randn(
        (batch, pair_count, *tail),
        dtype=torch.float32,
        device="cpu",
        generator=cpu_gen,
    )
    paired = torch.cat([base, -base], dim=1)[:, :random_count]
    noise = torch.zeros((batch, count, *tail), dtype=torch.float32, device="cpu")
    if random_count:
        noise[:, exact_prefix:] = paired
    return noise.to(device=device, dtype=dtype)


class PushtVariablePooledGoalPrior(nn.Module):
    """Pooled latent trajectory-GMM prior conditioned on action token length."""
    def __init__(self, latent_dim:int, lowdim_dim:int, action_dim:int=10, max_plan_horizon:int=7, hidden_dim:int=512, num_heads:int=8, depth:int=3, num_modes:int=8, pooling:str="attention", min_log_std:float=-5.0, max_log_std:float=1.0):
        super().__init__()
        self.latent_dim=int(latent_dim); self.lowdim_dim=int(lowdim_dim); self.action_dim=int(action_dim)
        self.plan_horizon=int(max_plan_horizon); self.max_plan_horizon=int(max_plan_horizon)
        self.hidden_dim=int(hidden_dim); self.num_modes=int(num_modes); self.depth=int(depth); self.pooling=str(pooling)
        self.min_log_std=float(min_log_std); self.max_log_std=float(max_log_std)
        if self.pooling not in {"attention", "mean"}: raise ValueError(f"Unsupported pooling mode: {self.pooling}")
        self.visual_norm=nn.LayerNorm(self.latent_dim); self.visual_proj=nn.Linear(self.latent_dim, self.hidden_dim)
        self.history_type=nn.Parameter(torch.zeros(1,1,self.hidden_dim)); self.goal_type=nn.Parameter(torch.zeros(1,1,self.hidden_dim))
        if self.pooling == "attention":
            self.pool_query=nn.Parameter(torch.zeros(1,1,self.hidden_dim)); self.pool=nn.MultiheadAttention(self.hidden_dim, num_heads=int(num_heads), batch_first=True)
        else:
            self.register_parameter("pool_query", None); self.pool=None
        self.length_encoder=nn.Sequential(nn.LayerNorm(5), nn.Linear(5,self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim,self.hidden_dim))
        if self.lowdim_dim > 0:
            self.lowdim_encoder=nn.Sequential(nn.LayerNorm(self.lowdim_dim), nn.Linear(self.lowdim_dim,self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim,self.hidden_dim))
            trunk_input=3*self.hidden_dim
        else:
            self.lowdim_encoder=None; trunk_input=2*self.hidden_dim
        layers=[nn.LayerNorm(trunk_input), nn.Linear(trunk_input,self.hidden_dim), nn.SiLU()]
        for _ in range(max(self.depth-1,0)): layers += [nn.Linear(self.hidden_dim,self.hidden_dim), nn.SiLU()]
        self.trunk=nn.Sequential(*layers)
        out_dim=self.num_modes*self.max_plan_horizon*self.action_dim
        self.logit_head=nn.Linear(self.hidden_dim,self.num_modes); self.mean_head=nn.Linear(self.hidden_dim,out_dim); self.log_std_head=nn.Linear(self.hidden_dim,out_dim)
        params=[self.history_type,self.goal_type]
        if self.pool_query is not None: params.append(self.pool_query)
        for p in params: nn.init.trunc_normal_(p, std=0.02)
        nn.init.zeros_(self.logit_head.bias); nn.init.zeros_(self.mean_head.bias); nn.init.constant_(self.log_std_head.bias, -1.0)

    def _action_horizon_values(self, action_horizon, batch:int, device):
        if action_horizon is None:
            v=torch.full((batch,), float(self.max_plan_horizon), device=device)
        elif torch.is_tensor(action_horizon):
            v=action_horizon.to(device=device, dtype=torch.float32).view(-1)
            if v.numel()==1: v=v.expand(batch)
        else:
            v=torch.full((batch,), float(action_horizon), device=device)
        return v.clamp(1, self.max_plan_horizon)

    def forward(self, history_latents, goal_latents, lowdim, action_horizon=None):
        goal_latents=_flatten_goal_latents(goal_latents)
        history=self.visual_proj(self.visual_norm(history_latents))+self.history_type
        goal=self.visual_proj(self.visual_norm(goal_latents))+self.goal_type
        tokens=torch.cat([history, goal], dim=1)
        if self.pool is None:
            pooled=tokens.mean(dim=1)
        else:
            query=self.pool_query.expand(tokens.size(0), -1, -1); pooled,_=self.pool(query,tokens,tokens,need_weights=False); pooled=pooled.squeeze(1)
        hvals=self._action_horizon_values(action_horizon, tokens.size(0), tokens.device)
        length_feat=self.length_encoder(_time_features(hvals, self.max_plan_horizon))
        parts=[pooled, length_feat]
        if self.lowdim_encoder is not None: parts.append(self.lowdim_encoder(lowdim))
        hidden=self.trunk(torch.cat(parts, dim=-1))
        logits=self.logit_head(hidden)
        means=self.mean_head(hidden).view(-1,self.num_modes,self.max_plan_horizon,self.action_dim)
        log_stds=self.log_std_head(hidden).view_as(means).clamp(self.min_log_std, self.max_log_std)
        return {"logits":logits,"means":means,"log_stds":log_stds}

    def _slice_outputs(self, outputs, action_horizon=None):
        h=self.max_plan_horizon if action_horizon is None else int(action_horizon); h=max(1,min(h,self.max_plan_horizon))
        return {"logits":outputs["logits"],"means":outputs["means"][:,:,:h],"log_stds":outputs["log_stds"][:,:,:h]}

    def component_log_probs(self, outputs, actions):
        means=outputs["means"]; log_stds=outputs["log_stds"]; target=actions.unsqueeze(1); inv_std=torch.exp(-log_stds)
        lp=-0.5*((target-means)*inv_std).pow(2)-log_stds-0.5*math.log(2.0*math.pi)
        return lp.sum(dim=(-1,-2))

    def nll(self, outputs, target):
        outputs=self._slice_outputs(outputs, target.size(1)); comp=self.component_log_probs(outputs,target); log_mix=F.log_softmax(outputs["logits"], dim=-1)
        return -torch.logsumexp(log_mix+comp, dim=-1).mean()

    def best_mode_l1(self, outputs, target):
        outputs=self._slice_outputs(outputs, target.size(1)); per=torch.abs(outputs["means"]-target.unsqueeze(1)).mean(dim=(-1,-2)); return per.min(dim=1).values.mean()

    @torch.no_grad()
    def sample(self, history_latents, goal_latents, lowdim, num_samples:int, generator=None, action_horizon=None):
        outputs=self._slice_outputs(self(history_latents, goal_latents, lowdim, action_horizon), action_horizon)
        probs=F.softmax(outputs["logits"], dim=-1); mode_ids=_sample_modes(probs, num_samples, generator=generator)
        gather=mode_ids[:,:,None,None].expand(-1,-1,outputs["means"].size(2),self.action_dim)
        means=outputs["means"].gather(1,gather); log_stds=outputs["log_stds"].gather(1,gather)
        noise=torch.randn(means.shape, dtype=means.dtype, device=means.device, generator=generator)
        return means+noise*torch.exp(log_stds)

    @torch.no_grad()
    def top_mode(self, history_latents, goal_latents, lowdim, action_horizon=None):
        outputs=self._slice_outputs(self(history_latents, goal_latents, lowdim, action_horizon), action_horizon)
        mode_ids=outputs["logits"].argmax(dim=-1); gather=mode_ids[:,None,None,None].expand(-1,1,outputs["means"].size(2),self.action_dim)
        return outputs["means"].gather(1,gather).squeeze(1)


class PushtVariableTransformerGoalPrior(nn.Module):
    """Variable-length trajectory GMM prior with action-token decoding.

    The pooled prior compresses visual context first and then predicts a full
    action chunk with an MLP.  This version keeps a sequence of learned action
    queries and decodes them against visual/low-dimensional memory tokens.  The
    mixture mode is shared across the whole trajectory, while each action token
    has its own Gaussian parameters under that mode.
    """

    def __init__(
        self,
        latent_dim: int,
        lowdim_dim: int,
        action_dim: int = 10,
        max_plan_horizon: int = 7,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 3,
        num_modes: int = 8,
        dropout: float = 0.0,
        max_goal_offset: int = 200,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.lowdim_dim = int(lowdim_dim)
        self.action_dim = int(action_dim)
        self.plan_horizon = int(max_plan_horizon)
        self.max_plan_horizon = int(max_plan_horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_modes = int(num_modes)
        self.depth = int(depth)
        self.max_goal_offset = int(max_goal_offset)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.visual_norm = nn.LayerNorm(self.latent_dim)
        self.visual_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.history_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.far_goal_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.length_encoder = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.offset_encoder = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if self.lowdim_dim > 0:
            self.lowdim_encoder = nn.Sequential(
                nn.LayerNorm(self.lowdim_dim),
                nn.Linear(self.lowdim_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.lowdim_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        else:
            self.lowdim_encoder = None
            self.register_parameter("lowdim_type", None)

        self.length_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.offset_type = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.action_queries = nn.Parameter(
            torch.zeros(1, self.max_plan_horizon, self.hidden_dim)
        )
        self.action_pos = nn.Parameter(
            torch.zeros(1, self.max_plan_horizon, self.hidden_dim)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(depth))
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.logit_head = nn.Linear(self.hidden_dim, self.num_modes)
        self.mean_head = nn.Linear(self.hidden_dim, self.num_modes * self.action_dim)
        self.log_std_head = nn.Linear(self.hidden_dim, self.num_modes * self.action_dim)

        for p in [
            self.history_type,
            self.goal_type,
            self.far_goal_type,
            self.length_type,
            self.offset_type,
            self.action_queries,
            self.action_pos,
        ]:
            nn.init.trunc_normal_(p, std=0.02)
        if self.lowdim_type is not None:
            nn.init.trunc_normal_(self.lowdim_type, std=0.02)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_std_head.bias, -1.0)

    def _action_horizon_values(self, action_horizon, batch: int, device):
        if action_horizon is None:
            values = torch.full(
                (batch,), float(self.max_plan_horizon), device=device
            )
        elif torch.is_tensor(action_horizon):
            values = action_horizon.to(device=device, dtype=torch.float32).view(-1)
            if values.numel() == 1:
                values = values.expand(batch)
        else:
            values = torch.full((batch,), float(action_horizon), device=device)
        return values.clamp(1, self.max_plan_horizon)

    def _length_mask(self, hvals: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(self.max_plan_horizon, device=hvals.device)
        return steps[None, :] < hvals.long().clamp(1, self.max_plan_horizon)[:, None]

    def _offset_values(
        self,
        goal_offset_steps,
        subgoal_offset_steps,
        batch: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if goal_offset_steps is None:
            goal = torch.full(
                (batch,),
                float(self.max_goal_offset),
                device=device,
            )
        elif torch.is_tensor(goal_offset_steps):
            goal = goal_offset_steps.to(device=device, dtype=torch.float32).view(-1)
            if goal.numel() == 1:
                goal = goal.expand(batch)
        else:
            goal = torch.full((batch,), float(goal_offset_steps), device=device)
        if subgoal_offset_steps is None:
            subgoal = torch.clamp(goal, max=float(self.max_goal_offset))
        elif torch.is_tensor(subgoal_offset_steps):
            subgoal = subgoal_offset_steps.to(device=device, dtype=torch.float32).view(-1)
            if subgoal.numel() == 1:
                subgoal = subgoal.expand(batch)
        else:
            subgoal = torch.full((batch,), float(subgoal_offset_steps), device=device)
        return goal.clamp(1, self.max_goal_offset), subgoal.clamp(1, self.max_goal_offset)

    def _offset_features(
        self,
        goal_offset_steps,
        subgoal_offset_steps,
        batch: int,
        device,
    ) -> torch.Tensor:
        goal, subgoal = self._offset_values(
            goal_offset_steps,
            subgoal_offset_steps,
            batch,
            device,
        )
        denom = max(float(self.max_goal_offset), 1.0)
        ratio = subgoal / torch.clamp(goal, min=1.0)
        return self.offset_encoder(
            torch.stack([goal / denom, subgoal / denom, ratio], dim=-1)
        )

    def forward(
        self,
        history_latents,
        goal_latents,
        lowdim,
        action_horizon=None,
        far_goal_latents=None,
        goal_offset_steps=None,
        subgoal_offset_steps=None,
    ):
        goal_latents = _flatten_goal_latents(goal_latents)
        if far_goal_latents is None:
            far_goal_latents = goal_latents
        else:
            far_goal_latents = _flatten_goal_latents(far_goal_latents)
        batch = int(history_latents.size(0))
        history = (
            self.visual_proj(self.visual_norm(history_latents)) + self.history_type
        )
        goal = self.visual_proj(self.visual_norm(goal_latents)) + self.goal_type
        far_goal = (
            self.visual_proj(self.visual_norm(far_goal_latents))
            + self.far_goal_type
        )
        hvals = self._action_horizon_values(
            action_horizon, batch, history_latents.device
        )
        length_token = self.length_encoder(
            _time_features(hvals, self.max_plan_horizon)
        ).unsqueeze(1) + self.length_type
        offset_token = self._offset_features(
            goal_offset_steps,
            subgoal_offset_steps,
            batch,
            history_latents.device,
        ).unsqueeze(1) + self.offset_type
        memory_parts = [history, goal, far_goal, length_token, offset_token]
        if self.lowdim_encoder is not None:
            memory_parts.append(self.lowdim_encoder(lowdim).unsqueeze(1) + self.lowdim_type)
        memory = self.memory_norm(torch.cat(memory_parts, dim=1))
        query = (self.action_queries + self.action_pos).expand(batch, -1, -1)
        decoded = self.output_norm(self.decoder(query, memory))
        mask = self._length_mask(hvals).to(decoded.dtype)
        pooled = (decoded * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits = self.logit_head(pooled)
        means = self.mean_head(decoded).view(
            batch, self.max_plan_horizon, self.num_modes, self.action_dim
        ).permute(0, 2, 1, 3).contiguous()
        log_stds = self.log_std_head(decoded).view_as(
            means.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3).contiguous()
        log_stds = log_stds.clamp(self.min_log_std, self.max_log_std)
        return {"logits": logits, "means": means, "log_stds": log_stds}

    def _slice_outputs(self, outputs, action_horizon=None):
        horizon = self.max_plan_horizon if action_horizon is None else int(action_horizon)
        horizon = max(1, min(horizon, self.max_plan_horizon))
        return {
            "logits": outputs["logits"],
            "means": outputs["means"][:, :, :horizon],
            "log_stds": outputs["log_stds"][:, :, :horizon],
        }

    def component_log_probs(self, outputs, actions):
        means = outputs["means"]
        log_stds = outputs["log_stds"]
        target = actions.unsqueeze(1)
        inv_std = torch.exp(-log_stds)
        lp = (
            -0.5 * ((target - means) * inv_std).pow(2)
            - log_stds
            - 0.5 * math.log(2.0 * math.pi)
        )
        return lp.sum(dim=(-1, -2))

    def nll(self, outputs, target):
        outputs = self._slice_outputs(outputs, target.size(1))
        comp = self.component_log_probs(outputs, target)
        log_mix = F.log_softmax(outputs["logits"], dim=-1)
        return -torch.logsumexp(log_mix + comp, dim=-1).mean()

    def best_mode_l1(self, outputs, target):
        outputs = self._slice_outputs(outputs, target.size(1))
        per_mode = torch.abs(outputs["means"] - target.unsqueeze(1)).mean(
            dim=(-1, -2)
        )
        return per_mode.min(dim=1).values.mean()

    @torch.no_grad()
    def sample(
        self,
        history_latents,
        goal_latents,
        lowdim,
        num_samples: int,
        generator=None,
        action_horizon=None,
        far_goal_latents=None,
        goal_offset_steps=None,
        subgoal_offset_steps=None,
    ):
        outputs = self._slice_outputs(
            self(
                history_latents,
                goal_latents,
                lowdim,
                action_horizon,
                far_goal_latents,
                goal_offset_steps,
                subgoal_offset_steps,
            ),
            action_horizon,
        )
        probs = F.softmax(outputs["logits"], dim=-1)
        mode_ids = _sample_modes(
            probs,
            num_samples,
            generator=generator,
        )
        gather = mode_ids[:, :, None, None].expand(
            -1, -1, outputs["means"].size(2), self.action_dim
        )
        means = outputs["means"].gather(1, gather)
        log_stds = outputs["log_stds"].gather(1, gather)
        exact_prefix = self.num_modes if os.environ.get("PUSHT_STRATIFIED_GMM", "0") == "1" else 0
        noise = _sample_noise(
            means.shape,
            dtype=means.dtype,
            device=means.device,
            generator=generator,
            exact_prefix=exact_prefix,
        )
        return means + noise * torch.exp(log_stds)

    @torch.no_grad()
    def top_mode(
        self,
        history_latents,
        goal_latents,
        lowdim,
        action_horizon=None,
        far_goal_latents=None,
        goal_offset_steps=None,
        subgoal_offset_steps=None,
    ):
        outputs = self._slice_outputs(
            self(
                history_latents,
                goal_latents,
                lowdim,
                action_horizon,
                far_goal_latents,
                goal_offset_steps,
                subgoal_offset_steps,
            ),
            action_horizon,
        )
        mode_ids = outputs["logits"].argmax(dim=-1)
        gather = mode_ids[:, None, None, None].expand(
            -1, 1, outputs["means"].size(2), self.action_dim
        )
        return outputs["means"].gather(1, gather).squeeze(1)


def load_action_prior(path: str | Path, device: torch.device):
    """Load the variable-duration action prior used by the paper.

    The public evaluator intentionally accepts only the two variable-duration
    architectures defined in this module.  Older language-conditioned and
    fixed-horizon checkpoints fail explicitly instead of being routed through
    a legacy adapter.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    architecture = config.pop("architecture", "variable_transformer")
    if architecture == "variable_transformer":
        config.pop("pooling", None)
        model = PushtVariableTransformerGoalPrior(**config)
    elif architecture == "variable_pooled":
        model = PushtVariablePooledGoalPrior(**config)
    else:
        raise ValueError(
            "SAGE expects a variable-duration action prior; "
            f"checkpoint architecture is {architecture!r}"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    stats = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in checkpoint["stats"].items()
    }
    return model, stats, checkpoint
