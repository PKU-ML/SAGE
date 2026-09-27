"""Train the paper stride-1 terminal subgoal generator or GMM action prior."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sage.models.action_prior import PushtVariableTransformerGoalPrior
from sage.models.subgoal import PushtSubgoalPrior


def build_terminal_staircase_specs(lengths, subgoal_tokens, max_goal_tokens,
                                   include_direct_final, seed):
    rows = []
    minimum = subgoal_tokens if include_direct_final else 2 * subgoal_tokens
    for episode, length in enumerate(lengths):
        terminal = int(length) - 1
        for start in range(terminal - minimum + 1):
            remaining = terminal - start
            budget = remaining if max_goal_tokens is None else min(max_goal_tokens, remaining)
            rows.append((episode, start, budget, terminal))
    result = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    np.random.default_rng(seed).shuffle(result)
    return result


class PairDataset(Dataset):
    def __init__(self, root, specs, subgoal_tokens=15, cache_stride=1):
        self.root, self.specs = Path(root), specs
        self.subgoal_tokens, self.cache_stride = subgoal_tokens, cache_stride
        self._latents = self._actions = None

    def __len__(self):
        return len(self.specs)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_latents'] = state['_actions'] = None
        return state

    @property
    def latents(self):
        if self._latents is None:
            self._latents = np.load(self.root / 'latents.npy', mmap_mode='r')
        return self._latents

    @property
    def actions(self):
        if self._actions is None:
            self._actions = np.load(self.root / 'actions.npy', mmap_mode='r')
        return self._actions

    def __getitem__(self, index):
        episode, start, remaining, terminal = map(int, self.specs[index])
        local = start + self.subgoal_tokens
        history = [start] * 3 if start < 2 else [start - 2, start - 1, start]
        return dict(
            history=torch.from_numpy(np.array(self.latents[episode, history], dtype=np.float32)),
            far_goal=torch.from_numpy(np.array(self.latents[episode, terminal], dtype=np.float32)),
            local_goal=torch.from_numpy(np.array(self.latents[episode, local], dtype=np.float32)),
            actions=torch.from_numpy(np.array(self.actions[episode, start:local], dtype=np.float32)),
            goal_steps=torch.tensor(remaining * self.cache_stride, dtype=torch.float32))


@torch.no_grad()
def resolve_local_goal(generator, history, far, offsets, tau):
    predicted = generator(history, far, history.new_empty(history.size(0), 0),
                          offsets, torch.full_like(offsets, tau))['prediction']
    return torch.where((offsets <= tau).view(-1, 1, 1), far, predicted)


def prior_forward(model, history, local, far, offsets, tau, horizon):
    return model(history, local, history.new_empty(history.size(0), 0),
        action_horizon=horizon, far_goal_latents=far, goal_offset_steps=offsets,
        subgoal_offset_steps=torch.full_like(offsets, tau))


@torch.no_grad()
def evaluate(model, generator, loader, device, tau, samples=16, max_batches=0):
    model.eval()
    totals = dict(l1=0., mse=0.) if generator is None else dict(objective=0., top_l1=0., best_sample_l1=0.)
    count, buckets = 0, {}
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        history, far = batch['history'].to(device), batch['far_goal'].to(device)[:, None]
        offsets = batch['goal_steps'].to(device)
        empty = history.new_empty(history.size(0), 0)
        durations = torch.full_like(offsets, tau)
        n = len(history)
        if generator is None:
            prediction = model(history, far, empty, offsets, durations)['prediction']
            difference = prediction - batch['local_goal'].to(device)[:, None]
            values = dict(l1=difference.abs().flatten(1).mean(1), mse=difference.square().flatten(1).mean(1))
        else:
            actions = batch['actions'].to(device)
            local = resolve_local_goal(generator, history, far, offsets, tau)
            output = prior_forward(model, history, local, far, offsets, tau, actions.shape[1])
            top = model.top_mode(history, local, empty, actions.shape[1], far, offsets, durations)
            proposals = model.sample(history, local, empty, samples,
                action_horizon=actions.shape[1], far_goal_latents=far,
                goal_offset_steps=offsets, subgoal_offset_steps=durations)
            totals['objective'] += float(model.nll(output, actions)) * n
            values = dict(top_l1=(top - actions).abs().mean(dim=(-1, -2)),
                best_sample_l1=(proposals - actions[:, None]).abs().mean(dim=(-1, -2)).min(dim=1).values)
        count += n
        for name, value in values.items():
            totals[name] += float(value.sum())
        for horizon in offsets.long().unique().tolist():
            mask = offsets.long() == horizon
            bucket = buckets.setdefault(str(horizon), dict(count=0, **{key: 0. for key in values}))
            bucket['count'] += int(mask.sum())
            for name, value in values.items():
                bucket[name] += float(value[mask].sum())
    if not count:
        raise ValueError('No validation batches')
    result = {key: value / count for key, value in totals.items()}
    result['per_horizon'] = {h: {k: v if k == 'count' else v / row['count']
        for k, v in row.items()} for h, row in buckets.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['generator', 'prior'], required=True)
    parser.add_argument('--train-cache', type=Path, required=True)
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--generator-checkpoint', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--recipe', type=Path,
                        help='Verify architecture and ordered training windows against the paper recipe')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-val-batches', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    prior = args.stage == 'prior'
    datasets, metas, specs = [], [], []
    for i, root in enumerate((args.train_cache, args.val_cache)):
        meta = json.loads((root / 'metadata.json').read_text())
        if not meta['complete'] or meta['cache_stride'] != 1 or meta['action_dim'] != 14:
            raise ValueError('A complete raw stride-1 cache with 14D actions is required')
        windows = build_terminal_staircase_specs(meta['lengths'], 15, None, prior, args.seed + i)
        if not len(windows):
            raise ValueError('No eligible terminal windows')
        datasets.append(PairDataset(root, windows))
        metas.append(meta)
        specs.append(windows)
    if metas[0]['latent_dim'] != metas[1]['latent_dim']:
        raise ValueError('Training and validation latent dimensions differ')
    for key in ('encoder_checkpoint_sha256', 'action_stats_sha256'):
        if metas[0].get(key) != metas[1].get(key):
            raise ValueError(f'Training and validation cache {key} differ')
    train_loader = DataLoader(datasets[0], args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(datasets[1], args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    if not len(train_loader):
        raise ValueError('Batch size exceeds the number of training windows')
    config = dict(latent_dim=metas[0]['latent_dim'], lowdim_dim=0,
        max_goal_offset=int(max(s[:, 2].max() for s in specs)))
    generator = None
    if prior:
        if args.generator_checkpoint is None:
            raise ValueError('Prior training requires the selected generator checkpoint')
        saved = torch.load(args.generator_checkpoint, map_location='cpu', weights_only=True)
        generator = PushtSubgoalPrior(**saved['model_config'])
        generator.load_state_dict(saved['model'], strict=True)
        generator = generator.to(device).eval().requires_grad_(False)
        config.update(hidden_dim=512, num_heads=8, depth=3, num_modes=8,
                      dropout=0., action_dim=14, max_plan_horizon=15)
        model = PushtVariableTransformerGoalPrior(**config).to(device)
    else:
        config.update(hidden_dim=896, num_heads=8, depth=4, predict_residual_from='goal',
                      pooling='decoder', goal_condition_mode='goal')
        model = PushtSubgoalPrior(**config).to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / 'run_manifest.json').exists():
        raise FileExistsError(args.out_dir / 'run_manifest.json')
    manifest = dict(args=json.loads(json.dumps(vars(args), default=str)), model_config=config,
        parameter_count=sum(p.numel() for p in model.parameters()),
        windows={split: dict(count=len(s), sha256=hashlib.sha256(s.tobytes()).hexdigest())
                 for split, s in zip(('train', 'validation'), specs)})
    if args.recipe:
        expected = json.loads(args.recipe.read_text())[args.stage]
        for key in ('model_config', 'windows'):
            if manifest[key] != expected[key]:
                raise ValueError(f'Paper recipe mismatch: {key}')
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    (args.out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best, metrics = math.inf, []
    for epoch in range(1, (args.epochs or (4 if prior else 6)) + 1):
        model.train()
        total, seen = 0., 0
        for index, batch in enumerate(train_loader):
            if args.max_train_batches and index >= args.max_train_batches:
                break
            history = batch['history'].to(device, non_blocking=True)
            far = batch['far_goal'].to(device, non_blocking=True)[:, None]
            target = batch['local_goal'].to(device, non_blocking=True)[:, None]
            actions = batch['actions'].to(device, non_blocking=True)
            offsets = batch['goal_steps'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                if generator is None:
                    output = model(history, far, history.new_empty(len(history), 0), offsets,
                                   torch.full_like(offsets, 15))
                    loss = model.loss(output, target, cosine_weight=.1, smooth_l1_beta=.05)['loss']
                else:
                    local = resolve_local_goal(generator, history, far, offsets, 15)
                    output = prior_forward(model, history, local.float(), far, offsets, 15, actions.shape[1])
                    output = {key: value.float() for key, value in output.items()}
                    loss = model.nll(output, actions) + .05 * model.best_mode_l1(output, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(history)
            seen += len(history)
            if index % 250 == 0:
                print(f'epoch={epoch} batch={index} loss={total / seen:.6f}', flush=True)
        validation = evaluate(model, generator, val_loader, device, 15, max_batches=args.max_val_batches)
        score = validation['best_sample_l1' if prior else 'l1']
        row = dict(epoch=epoch, train_loss=total / seen, validation=validation)
        metrics.append(row)
        payload = dict(model=model.state_dict(), model_config=config, epoch=epoch,
                       validation=validation, manifest=manifest)
        torch.save(payload, args.out_dir / 'latest.pt')
        # Preserve the selected generator epoch from the paper independently of best validation.
        if not prior and epoch == 1:
            torch.save(payload, args.out_dir / 'epoch1.pt')
        if score < best:
            best = score
            torch.save(payload, args.out_dir / 'best.pt')
        (args.out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
