# RoboTwin A2B Training

This recipe trains the world model, terminal subgoal generator and GMM action
prior used for `place_a2b_left`. Run commands from the repository root on Linux
with NVIDIA GPUs. Evaluation and downloads are described in
[QUICKSTART_NATIVE.md](QUICKSTART_NATIVE.md).

## Train on the Released Data

Download and extract the A2B train and validation archives using the native
download guide. Set `DATA` to the extraction root containing the relative
episode paths in `data/training/robotwin_a2b/train.json` and `validation.json`.

```bash
conda env create -f runtime/robotwin/training_environment.yml
conda activate sage-robotwin-train
pip install --no-deps -e .
export DATA=/path/to/robotwin_a2b
export RUN=/path/to/new-training-run

# Optional bounded integration test; use a separate output directory.
python -m sage.train_robotwin --data-root "$DATA" --out "${RUN}-smoke" \
  --gpus 0 --workers 0 --smoke

# Complete recipe: Stage A -> latent cache -> Stage B -> generator -> prior.
python -m sage.train_robotwin --data-root "$DATA" --out "$RUN" \
  --gpus 0,1,2,3 --workers 4
```

The wrapper preserves manifest order and global batches when changing GPU
count. Use 1, 2, 3 or 4 GPUs; smaller GPU counts require more memory per GPU.
The original training used four GPUs. `--dry-run` prints the commands without
training. `--stages` runs selected stages, for example `--stages generator prior
export` after the cache and world model have finished. Use a fresh output
directory for a new run; existing caches are not overwritten.

| Component | Recipe |
| --- | --- |
| Data | 8,000 train / 1,000 validation / 1,000 test, success-filtered |
| Observations | Head and both wrists, 224 x 224, raw stride 1 |
| Action | 14 joint/gripper coordinates, one raw action per frame |
| World model | 89.3M, shared ViT-small image encoder, three-view latent |
| Stage A | 2 epochs, global batch 96, AdamW LR 5e-5, weight decay 0.001 |
| Stage B | 2 epochs, global batch 288, same optimizer, frozen encoder/projector |
| B objective | Teacher-forced prediction plus 5-step recursive loss, weight 0.1 |
| Generator | 63.2M, hidden 896 / depth 4 / heads 8, 6 epochs |
| Action prior | 13.9M GMM, hidden 512 / depth 3 / heads 8 / modes 8, 4 epochs |
| Planning chunk | 15 raw actions |
| Seed | 42 |

The exact architecture and ordered-window checks are in
`configs/robotwin_training.json`; the saved A/B configurations are in
`sage/train/config/robotwin_{a2,b2}.yaml`. The generator and prior enumerate
eligible terminal-goal windows, rather than sampling only multiples of 15.
History has three observations; the far goal is the episode's terminal frame.
The local target is 15 steps ahead; short remaining horizons use the far goal
directly. Generator windows start at remaining horizon 30, prior windows at 15.
Action normalization is fitted on the training split only.
Generator/prior both use batch 512 and AdamW with learning rate and weight
decay `1e-4`. Generator loss is smooth L1 (beta 0.05) plus cosine loss (weight
0.1); prior loss is GMM NLL plus best-mode L1 (weight 0.05). At the first two
history positions the current latent is repeated three times, matching the
released training code.

For released A2B data there are 948,823 / 118,500 generator windows and
1,068,823 / 133,500 prior windows (train / validation). The prior is trained
with the selected epoch-1 generator. `bundle/` exports Stage-B epoch 2,
generator epoch 1, prior epoch 4, model configuration and action statistics.
An independently trained bundle has its own latent space: never mix its
generator, prior or statistics with the released checkpoints.

## Generate A2B Data

Collection uses a separate environment. Install the locked runtime as described
in the native guide first; keep its complete asset inventory. Collection also
requires a CUDA toolkit compatible with PyTorch, `nvcc`, a C++ compiler and
the pinned cuRobo source revision:

```bash
conda env create -f runtime/robotwin/collection_environment.yml
conda activate sage-robotwin-collect
pip install --no-deps -e .
# Point to a CUDA 12.x development toolkit compatible with torch.version.cuda.
export CUDA_HOME=/path/to/cuda-12
export PATH="$CUDA_HOME/bin:$PATH"
git clone https://github.com/NVlabs/curobo.git .runtime/curobo
git -C .runtime/curobo checkout 0a50de1ba72db304195d59d9d0b1ed269696047f
pip install --no-build-isolation -e .runtime/curobo
export ROBOTWIN_ROOT="$PWD/.runtime/robotwin-a2b"
python -m sage.robotwin_runtime --out "$ROBOTWIN_ROOT"
export COLLECT=/path/to/a2b-collection

# Two accepted episodes, including raw-stride-1 HDF5 writing.
python -m sage.collect_robotwin --robotwin-root "$ROBOTWIN_ROOT" \
  --curobo-root "$PWD/.runtime/curobo" --out "$COLLECT" \
  --split train --shard 0 --episodes 2 --gpu 0
```

Use a separate smoke output directory if keeping the full collection directory
untouched. To collect the full dataset, run train shards 0 through 7 with
`--episodes 1000`, then validation shard 0 and test shard 0, each with 1,000
episodes. Independent shards may run on separate GPUs. Do not run two writers
against the same split/shard directory.

The collector links the runtime's `assets/` into cuRobo's standard asset and
robot-config resource roots, so relative URDF and collision-sphere paths resolve
correctly. Use a writable, dedicated cuRobo checkout for this runtime; source,
simulator assets and camera settings are not rewritten.

Each shard first generates oracle joint paths and retains episodes passing
`plan_success` and task success. It then replays those paths and writes three
camera views at every raw action step, checking task success again. No added
action noise or recovery is enabled. Seed starts are 10,000,000 / 20,000,000 /
30,000,000 for train / validation / test, plus 100,000 times the shard index.
Failed attempts do not count toward the requested number of successful episodes.

```bash
python scripts/build_robotwin_collection_manifests.py \
  --root "$COLLECT/data/episodes" --out-dir "$COLLECT/manifests"
python -m sage.train_robotwin --data-root "$COLLECT/data/episodes" \
  --train-manifest "$COLLECT/manifests/train.json" \
  --validation-manifest "$COLLECT/manifests/validation.json" \
  --out /path/to/new-collected-data-run --gpus 0,1,2,3 --new-task
```

`--new-task` disables the released-data window-count/hash checks, not the model
or training recipe. Newly collected episodes need not be byte-identical to the
released data across GPU drivers. Use the published test manifests and matched
released model bundle to reproduce the paper evaluation; do not substitute
newly sampled episodes for those queries. For a different task, its environment,
task configuration, collection logic and validation-based checkpoint choice
must also be adapted; the supplied collector is specifically A2B.
