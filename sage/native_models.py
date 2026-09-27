"""Load the paper's multi-view LeWM, subgoal generator and GMM prior bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sage.models.action_prior import PushtVariableTransformerGoalPrior
from sage.models.subgoal import PushtSubgoalPrior


def make_native_encoder(config):
    """Instantiate the published ViT shape without importing training utilities."""
    from transformers import ViTConfig, ViTModel

    config = dict(config)
    if config.pop('_target_') != 'stable_pretraining.backbone.utils.vit_hf':
        raise ValueError('Unexpected native encoder factory')
    size = config.pop('size')
    if config.pop('pretrained'):
        raise ValueError('Native inference loads the released weights, not remote pretrained weights')
    width, heads = {'tiny': (192, 3), 'small': (384, 6)}[size]
    use_mask_token = config.pop('use_mask_token')
    encoder = ViTModel(ViTConfig(hidden_size=width, num_hidden_layers=12,
        num_attention_heads=heads, intermediate_size=4*width, **config),
        add_pooling_layer=False, use_mask_token=use_mask_token)
    encoder.config.interpolate_pos_encoding = True
    return encoder


def load_world_model(checkpoint, device="cpu"):
    from hydra.utils import instantiate

    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint.parent / "config.json").read_text())
    encoder = make_native_encoder(config.pop('encoder'))
    wm = instantiate(config, encoder=encoder)
    wm.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return wm.to(device).eval()


def load_components(directory, device="cpu"):
    directory = Path(directory)
    models = {"world_model": load_world_model(directory / "world_model/weights.pt", device)}
    metadata = {}
    for name, cls in (("generator", PushtSubgoalPrior), ("prior", PushtVariableTransformerGoalPrior)):
        checkpoint = torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
        config = dict(checkpoint["model_config"])
        if name == "prior":
            if config.pop("proposal_type", "gmm") != "gmm" or config.pop("uniform_mixture", False):
                raise ValueError("The paper bundle requires a learned-weight GMM prior")
        model = cls(**config)
        model.load_state_dict(checkpoint["model"], strict=True)
        models[name] = model.to(device).eval()
        metadata[name] = {k: v for k, v in checkpoint.items() if k != "model"}
    if models["generator"].latent_dim != models["prior"].latent_dim:
        raise ValueError("Generator/prior latent dimensions differ")
    return models, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    models, metadata = load_components(args.bundle, args.device)
    generator, prior = models["generator"], models["prior"]
    d = generator.latent_dim
    with torch.inference_mode():
        history = torch.zeros(1, 3, d, device=args.device)
        goal = torch.zeros(1, 1, d, device=args.device)
        lowdim = torch.empty(1, 0, device=args.device)
        delta = torch.tensor([60.], device=args.device)
        tau = torch.tensor([15.], device=args.device)
        predicted = generator(history, goal, lowdim, delta, tau)
        action = prior.top_mode(history, goal, lowdim, action_horizon=prior.max_plan_horizon,
                                far_goal_latents=goal, goal_offset_steps=delta,
                                subgoal_offset_steps=tau)
        if not torch.isfinite(action).all():
            raise RuntimeError("Non-finite prior output")
        for value in predicted.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError("Non-finite generator output")
    print(json.dumps({"strict_load": True, "finite_component_forward": True,
                      "action_shape": list(action.shape),
                      "parameters": {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
                      "selected_epochs": {k: v.get("epoch") for k, v in metadata.items()}}, indent=2))


if __name__ == "__main__":
    main()
