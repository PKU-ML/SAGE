import os
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained
from stable_worldmodel.data.formats.robotwin_compressed import (
    RobotWinCompressedDataset,
)


class SquareResize:
    """Force an image sequence to the configured square ViT resolution."""

    def __init__(self, source: str, target: str, size: int):
        self.source = source
        self.target = target
        self.size = int(size)

    def __call__(self, batch):
        images = batch[self.source]
        if images.ndim not in (3, 4):
            raise ValueError(
                f'{self.source} must be CxHxW or TxCxHxW, got {images.shape}'
            )
        sequence = images.unsqueeze(0) if images.ndim == 3 else images
        batch[self.target] = F.interpolate(
            sequence.float(),
            size=(self.size, self.size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0) if images.ndim == 3 else F.interpolate(
            sequence.float(),
            size=(self.size, self.size),
            mode='bilinear',
            align_corners=False,
        )
        return batch


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = SquareResize(source=source, target=target, size=img_size)
    return dt.transforms.Compose(to_image, resize)


def _without_pixel_columns(dataset_cfg: dict) -> dict:
    """Remove image columns when frozen visual latents are supplied."""
    result = dict(dataset_cfg)
    for field in ('keys_to_load', 'keys_to_cache'):
        if field in result:
            result[field] = [
                key for key in result[field] if 'pixels' not in key
            ]
    return result


def _load_dataset(dataset_name, dataset_cfg, transform, cache_dir):
    """Use native compressed RoboTwin episodes only when explicitly requested."""
    dataset_format = dataset_cfg.pop('format', None)
    if dataset_format is None:
        return swm.data.load_dataset(
            dataset_name, transform=transform, cache_dir=cache_dir, **dataset_cfg
        )
    if dataset_format == 'robotwin_compressed':
        return RobotWinCompressedDataset(
            dataset_name, transform=transform, **dataset_cfg
        )
    raise ValueError(f'Unsupported explicit dataset format: {dataset_format!r}')


class FrameLatentDataset(torch.utils.data.Dataset):
    """Attach cached raw-frame latents to an episode-window dataset."""

    def __init__(self, dataset, cache_path, num_steps, frameskip):
        self.dataset = dataset
        self.cache_path = str(cache_path)
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self._latents = None
        probe = np.load(self.cache_path, mmap_mode='r')
        if probe.ndim != 2:
            raise ValueError(
                f'Frame latent cache must be 2D, got {probe.shape}'
            )
        required_rows = int(
            np.max(np.asarray(dataset.offsets) + np.asarray(dataset.lengths))
        )
        if len(probe) < required_rows:
            raise ValueError(
                f'Frame latent cache has {len(probe)} rows but the dataset '
                f'requires at least {required_rows}.'
            )
        self.latent_dim = int(probe.shape[1])
        del probe

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name in {'dataset', '_latents'}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def _cache(self):
        if self._latents is None:
            self._latents = np.load(self.cache_path, mmap_mode='r')
        return self._latents

    def __getitem__(self, index):
        item = self.dataset[index]
        episode, start = self.dataset.clip_indices[index]
        first_row = int(self.dataset.offsets[episode]) + int(start)
        rows = first_row + np.arange(self.num_steps) * self.frameskip
        # Copy out of the read-only memmap before handing the tensor to workers.
        cached = np.array(self._cache()[rows], dtype=np.float32, copy=True)
        item['emb'] = torch.from_numpy(cached)
        return item


class TailAlignedWindowDataset(torch.utils.data.Dataset):
    """Add valid terminal windows without inventing post-terminal dynamics.

    The stock episode dataset uses ``num_steps * frameskip`` as a joint
    observation/action span.  For LeWM's ``history=3, recursive_steps=5``
    loss, the eighth observation only needs the first seven action blocks;
    the eighth block is never consumed.  The stock span therefore excludes
    the last four valid observation windows of every episode.  This wrapper
    restores them, padding only that unused final action token for collation.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.num_steps = int(dataset.num_steps)
        self.frameskip = int(dataset.frameskip)
        if self.num_steps < 2:
            raise ValueError('Tail-aligned windows require at least two frames.')
        self.obs_span = (self.num_steps - 1) * self.frameskip + 1
        self.full_span = self.num_steps * self.frameskip
        self.lengths = dataset.lengths
        self.offsets = dataset.offsets
        self._base_count = len(dataset)
        self.clip_indices = list(dataset.clip_indices)
        added = 0
        for episode, length in enumerate(self.lengths):
            # The base dataset admits starts through L-full_span.  Real
            # observations remain available through L-obs_span inclusive.
            for start in range(
                max(0, int(length) - self.full_span + 1),
                int(length) - self.obs_span + 1,
            ):
                self.clip_indices.append((episode, start))
                added += 1
        self.added_tail_windows = added

    def __len__(self):
        return len(self.clip_indices)

    def __getattr__(self, name):
        if name in {'dataset', 'clip_indices'}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self._base_count:
            return self.dataset[index]

        episode, start = self.clip_indices[index]
        # _load_slice applies the base dataset's transforms, including action
        # normalization, before we remove the action block after the terminal
        # frame.  That removed block is not part of any LeWM target.
        item = self.dataset._load_slice(episode, start, start + self.obs_span)
        if 'action' not in item:
            raise KeyError('Tail-aligned LeWM windows require action data.')
        action = item['action'][: (self.num_steps - 1) * self.frameskip]
        action = action.reshape(self.num_steps - 1, -1)
        # Preserve the ordinary batch shape. Trainer losses only read actions
        # through index num_steps-2, so this token has no effect on gradients.
        item['action'] = torch.cat([action, torch.zeros_like(action[:1])], dim=0)
        return item


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            config_key='model',
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    recursive_steps = int(cfg.loss.get('recursive_rollout_steps', 0))
    recursive_weight = float(cfg.loss.get('recursive_rollout_weight', 0.0))

    # Lightning recursively calls train() on the full model. Keep frozen
    # modules in inference mode as well so BatchNorm buffers and dropout do
    # not move the latent coordinate system during adaptation.
    for module_name in cfg.get('freeze_modules', []):
        getattr(self.model, module_name).eval()
    if cfg.get('freeze_modules', []):
        # pred_proj maps predictions into the frozen target space. Its affine
        # parameters remain trainable, but its BatchNorm running statistics
        # must stay fixed or the planner's cost coordinates drift.
        self.model.pred_proj.eval()
    if recursive_steps > 0:
        # The planner rolls out deterministic inference-mode dynamics. Keep
        # dropout disabled and BatchNorm buffers fixed during the recursive
        # objective while still allowing gradients through their parameters.
        self.model.predictor.eval()

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    if 'emb' in batch:
        output = dict(batch)
        output['emb'] = batch['emb']
        output['act_emb'] = self.model.action_encoder(batch['action'])
        if hasattr(self.model, 'view_keys'):
            output['view_emb'] = output['emb'].unflatten(
                -1, (len(self.model.view_keys), -1)
            )
    else:
        output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    if recursive_steps > 0:
        if n_preds != 1:
            raise ValueError(
                'Recursive rollout training currently requires wm.num_preds=1.'
            )
        required_steps = ctx_len + recursive_steps
        if emb.size(1) < required_steps:
            raise ValueError(
                f'Recursive rollout training needs at least {required_steps} '
                f'frames, but the batch contains {emb.size(1)}. Set '
                'data.dataset.num_steps and data.val_dataset.num_steps to '
                f'{required_steps}.'
            )

        # Preserve the ordinary teacher-forced one-step objective over every
        # available transition in the longer training window.
        one_step_preds = []
        one_step_targets = []
        for start in range(emb.size(1) - ctx_len):
            stop = start + ctx_len
            pred = self.model.predict(
                emb[:, start:stop], act_emb[:, start:stop]
            )[:, -1]
            one_step_preds.append(pred)
            one_step_targets.append(emb[:, stop].detach())
        one_step_preds = torch.stack(one_step_preds, dim=1)
        one_step_targets = torch.stack(one_step_targets, dim=1)
        output['pred_loss'] = (
            one_step_preds - one_step_targets
        ).pow(2).mean()

        # Free-running recursive prediction: after the
        # initial history, each prediction becomes input to the next step.
        rollout_embs = list(emb[:, :ctx_len].unbind(dim=1))
        rollout_losses = []
        for step in range(recursive_steps):
            rollout_history = torch.stack(
                rollout_embs[-ctx_len:], dim=1
            )
            action_history = act_emb[:, step : step + ctx_len]
            next_pred = self.model.predict(
                rollout_history, action_history
            )[:, -1]
            target = emb[:, ctx_len + step].detach()
            rollout_losses.append((next_pred - target).pow(2).mean())
            rollout_embs.append(next_pred)
        output['recursive_rollout_loss'] = torch.stack(
            rollout_losses
        ).mean()
    else:
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]  # label
        pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred
        output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()

        if 'view_emb' in output:
            view_emb = output['view_emb']
            num_views = view_emb.size(2)
            pred_views = pred_emb.unflatten(-1, (num_views, -1))
            target_views = tgt_emb.unflatten(-1, (num_views, -1))
            for view_index, view_key in enumerate(self.model.view_keys):
                output[f'pred_loss_{view_key}'] = (
                    pred_views[:, :, view_index]
                    - target_views[:, :, view_index]
                ).pow(2).mean()

    if 'view_emb' in output:
        view_sigreg = []
        for view_index, view_key in enumerate(self.model.view_keys):
            value = self.sigreg(
                output['view_emb'][:, :, view_index].transpose(0, 1)
            )
            output[f'sigreg_loss_{view_key}'] = value
            view_sigreg.append(value)
        output['sigreg_loss'] = torch.stack(view_sigreg).mean()
    else:
        output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']
    if recursive_steps > 0:
        output['loss'] = (
            output['loss']
            + recursive_weight * output['recursive_rollout_loss']
        )

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='robotwin_a2')
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)
    #########################
    ##       dataset       ##
    #########################

    frame_cache_cfg = cfg.data.get('frame_latent_cache')
    use_frame_cache = frame_cache_cfg is not None
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    if use_frame_cache:
        dataset_cfg = _without_pixel_columns(dataset_cfg)
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = _load_dataset(dataset_name, dataset_cfg, transform=None, cache_dir=cache_dir)
    normalizer_dataset = dataset
    normalizer_dataset_cfg = cfg.data.get("normalizer_dataset")
    if normalizer_dataset_cfg is not None:
        resolved_normalizer_cfg = OmegaConf.to_container(
            normalizer_dataset_cfg, resolve=True
        )
        normalizer_dataset_name = resolved_normalizer_cfg.pop("name")
        if use_frame_cache:
            resolved_normalizer_cfg = _without_pixel_columns(
                resolved_normalizer_cfg
            )
        print(
            f'Loading normalization reference "{normalizer_dataset_name}" from '
            f'{"local cache: " + cache_dir if cache_dir else "default location"}'
        )
        normalizer_dataset = _load_dataset(
            normalizer_dataset_name,
            resolved_normalizer_cfg,
            transform=None,
            cache_dir=cache_dir,
        )
    pixel_columns = [
        col
        for col in cfg.data.dataset.keys_to_load
        if col == 'pixels' or col.endswith('_pixels')
    ]
    if not use_frame_cache and 'pixels' not in pixel_columns:
        raise ValueError("LeWM training requires a 'pixels' observation")
    transforms = []
    if not use_frame_cache:
        transforms = [
            get_img_preprocessor(
                source=col, target=col, img_size=cfg.img_size
            )
            for col in pixel_columns
        ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col in pixel_columns:
                continue

            normalizer = get_column_normalizer(normalizer_dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    val_dataset_cfg = cfg.data.get('val_dataset')
    if val_dataset_cfg is None:
        train_set, val_set = spt.data.random_split(
            dataset,
            lengths=[cfg.train_split, 1 - cfg.train_split],
            generator=rnd_gen,
        )
    else:
        resolved_val_cfg = OmegaConf.to_container(
            val_dataset_cfg, resolve=True
        )
        val_dataset_name = resolved_val_cfg.pop('name')
        if use_frame_cache:
            resolved_val_cfg = _without_pixel_columns(resolved_val_cfg)
        print(
            f'Loading validation dataset "{val_dataset_name}" from '
            f'{"local cache: " + cache_dir if cache_dir else "default location"}'
        )
        val_set = _load_dataset(
            val_dataset_name,
            resolved_val_cfg,
            transform=transform,
            cache_dir=cache_dir,
        )
        train_set = dataset

    if bool(cfg.data.get('tail_aligned_windows', False)):
        train_set = TailAlignedWindowDataset(train_set)
        val_set = TailAlignedWindowDataset(val_set)
        print(
            'Using tail-aligned windows: '
            f'train +{train_set.added_tail_windows}, '
            f'validation +{val_set.added_tail_windows}.'
        )

    if use_frame_cache:
        train_cache = str(frame_cache_cfg.train)
        val_cache = str(frame_cache_cfg.validation)
        train_set = FrameLatentDataset(
            train_set,
            train_cache,
            cfg.data.dataset.num_steps,
            cfg.data.dataset.frameskip,
        )
        val_set = FrameLatentDataset(
            val_set,
            val_cache,
            cfg.data.val_dataset.num_steps,
            cfg.data.val_dataset.frameskip,
        )
        expected_dim = int(cfg.model.predictor.input_dim)
        if train_set.latent_dim != expected_dim:
            raise ValueError(
                f'Train cache latent dim {train_set.latent_dim} does not '
                f'match predictor input dim {expected_dim}.'
            )
        if val_set.latent_dim != expected_dim:
            raise ValueError(
                f'Validation cache latent dim {val_set.latent_dim} does not '
                f'match predictor input dim {expected_dim}.'
            )
        print(
            f'Using frame latent cache: train={train_cache}, '
            f'validation={val_cache}, dim={expected_dim}'
        )

    train_loader_cfg = dict(cfg.loader)
    train = torch.utils.data.DataLoader(
        train_set,
        **train_loader_cfg,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint:
        pretrained = swm.wm.utils.load_pretrained(str(init_checkpoint))
        try:
            world_model.load_state_dict(pretrained.state_dict(), strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                f"Failed to initialize LeWM exactly from {init_checkpoint}; "
                "refusing to continue with a partially loaded model."
            ) from error
        print(f"Initialized model weights from {init_checkpoint}")

    freeze_modules = list(cfg.get('freeze_modules', []))
    for module_name in freeze_modules:
        if not hasattr(world_model, module_name):
            raise ValueError(
                f'Cannot freeze unknown model module {module_name!r}.'
            )
        module = getattr(world_model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    if freeze_modules:
        trainable = sum(
            parameter.numel()
            for parameter in world_model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in world_model.parameters())
        print(
            f'Frozen modules: {freeze_modules}; trainable parameters: '
            f'{trainable:,}/{total:,}'
        )

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.02 * total_steps)),
                'max_steps': total_steps,
            },
            # Both scheduler arguments are expressed in optimizer updates.
            # Stepping this scheduler per epoch would leave long runs near
            # the warmup learning rate for their entire training budget.
            'interval': 'step',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    callbacks = []
    if bool(cfg.get('save_checkpoints', True)):
        callbacks.append(
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg,
                epoch_interval=int(cfg.get('checkpoint_interval', 1)),
            )
        )

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.setdefault('enable_checkpointing', True)
    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()
