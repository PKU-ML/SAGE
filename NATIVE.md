# Native Manipulation Suites

Run from the repository root on Linux with an NVIDIA GPU. Model weights,
datasets and simulator assets are installed separately from the source code.
Native assets currently require a supplied local mirror.

## Environment

Use separate environments: the released LIBERO and RoboTwin encoders require
different Transformers checkpoint layouts. Do not bypass strict checkpoint
loading or mix the two dependency sets.

```bash
conda env create -f runtime/libero/environment.yml
conda activate sage-libero
pip install --no-deps -e .
python scripts/check_native_runtime.py libero --gpu --numerics
```

For RoboTwin, replace `libero` with `robotwin` in the three relevant names.
The environment files pin the dependencies. Rendering and replay additionally
depend on the GPU driver and simulator assets; version checks alone do not
validate them.

LIBERO additionally requires its asset-bearing checkout at the commit in
`runtime/native_contracts.json`. Pass it with `--libero-root`. Configuration is
created in the output directory; the evaluator does not modify `~/.libero`.

RoboTwin requires the content-locked A2B runtime and its original scene assets;
pass their common root with `--robotwin-root`. A random upstream checkout is not
interchangeable with the collection runtime. Texture counts and object assets
affect seeded initialization, so do not prune them to save space.

The A2B source snapshot is a small separate MIT-licensed archive. Its 12.31 GB
simulator-asset inventory preserves the original texture sampling universe.
Install the locally staged source and assets without changing filenames:

```bash
python -m sage.assets --registry runtime/robotwin/source_archive.json \
  --source-dir "$WEIGHT_MIRROR" --out-dir downloads
mkdir -p "$ROBOTWIN_ROOT"
tar -xzf downloads/robotwin_a2b_runtime.tar.gz -C "$ROBOTWIN_ROOT"
python -m sage.assets --registry runtime/robotwin/assets.json \
  --source-dir "$ROBOTWIN_ASSET_MIRROR" --out-dir "$ROBOTWIN_ROOT"
```

The asset mirror is rooted above `assets/`. These are required scene/robot
assets, not additional training episodes. Output and cuRobo configuration paths
are relocated. The source snapshot supports recorded-path evaluation without
initializing unused cuRobo arm planners; native MPlib TOPP and gripper
interpolation are unchanged. New arm-path generation still requires cuRobo.
File-level provenance and the evaluation patch accompany `source_files.json`.

SAPIEN 3.0.0b1 invokes `nvidia-smi` before checking explicit driver settings.
If that probe hangs, an optional isolated overlay honors existing ICD settings
first, without modifying installed packages or simulator binaries:

```bash
python scripts/prepare_sapien_overlay.py \
  --package-dir "$CONDA_PREFIX/lib/python3.10/site-packages/sapien" \
  --out .runtime/sapien-overlay
export PYTHONPATH="$PWD/.runtime/sapien-overlay${PYTHONPATH:+:$PYTHONPATH}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

Use the actual NVIDIA ICD paths on your host. The helper preserves bundled
`sapien.libs` and records patch hashes. This fixes startup autodetection only;
it does not fix a broken driver or certify rendering.

## Checkpoints

The local weight mirror has one directory per native suite. Install, then verify:

```bash
python -m sage.assets --suite libero_scene2 --source-dir "$WEIGHT_MIRROR" --out-dir checkpoints
python -m sage.assets --suite libero_scene2 --out-dir checkpoints --verify-only
python -m sage.native_models checkpoints/libero_scene2
```

If supplied as a separate native-weight tar archive, extract it into the
checkpoint directory first, then run the same `--verify-only` and model-load
commands. The archive contains registered inference weights and their companion
configuration files only; it is not a dataset or a simulator asset package.

Use `libero_caddy` or `robotwin_a2b` for the other bundles. The filenames in
`configs/assets.json` are also the planned HF filenames. The loader never treats
an unpublished checkpoint as an available download. Configurations and action
statistics ship with the code. Optimizers and private training paths do not ship
with the inference weights.

## Data

Each suite's `data/manifests/<suite>/assets.json` lists the exact files needed by
its locked queries. The data mirror is rooted at that suite's directory:

```bash
python -m sage.assets --registry data/manifests/libero_scene2/assets.json \
  --source-dir "$SCENE2_DATA_MIRROR" --out-dir datasets/libero_scene2
python -m sage.native_data --queries data/manifests/libero_scene2/tail.json \
  --data-root datasets/libero_scene2
```

Installation verifies every SHA256. The second command checks query dependency
paths, not simulator replay. Missing files are not replaced with new episodes.

Archive filenames, sizes and SHA256 hashes are listed in `data/archives.json`.
For a supplied dataset tar archive, install it directly with file-level checksum
verification against the repository's locked registry:

```bash
python scripts/install_native_dataset.py --suite libero_caddy --split evaluation \
  --archive libero_caddy_evaluation.tar --out datasets/libero_caddy
```

Use the corresponding suite name for Scene2 or RoboTwin. Training archives use
`--split train` or `--split validation`; install both under the same data root.
The installer preserves verified existing files and rejects mismatched files or
archive links. Training indices are rebuilt using the commands below.

| Evaluation dependencies | Files | Uncompressed size |
|---|---:|---:|
| Scene2 | 39 HDF5 shards | 19.34 GB |
| Caddy | 6 HDF5 shards | 8.64 GB |
| RoboTwin A2B | 750 files | 1.36 GB |

These are evaluation inputs, not complete training datasets. Some locked Scene2
queries use additional test shards absent from the original 8k-training package.
RoboTwin includes joint-path pickles and terminal images as well as stride-1
HDF5 episodes. Load only trusted trajectory pickle files verified against the
manifest. Simulator assets are separate from these data sizes.

## Evaluation

```bash
python -m sage.reproduce_native --suite libero_scene2 \
  --libero-root "$LIBERO_ROOT" --data-root datasets/libero_scene2 \
  --checkpoints checkpoints --out results/libero_scene2
```

Default methods: `base prior_top sage`. Default horizons: `30 60 90 120`.
Each horizon has three locked groups of 50 queries. `--groups 0` selects one
group; `--dry-run` prints commands without loading data or models. For LIBERO
full episodes add `--full-episode` and use a separate output directory.
RoboTwin groups use proposal seeds 42/43/44, matching the archived runs.
LIBERO Scene2 uses proposal seed 20260829; Caddy uses 42. Both retain their
original per-query/per-stage seed formula across all three query groups.

For A2B use `--suite robotwin_a2b --robotwin-root "$ROBOTWIN_ROOT"` instead of
the LIBERO arguments. A2B exposes only the paper's tail evaluation.

After all three groups finish, aggregate a table with:

```bash
python -m sage.summarize_native --suite libero_scene2 \
  --results results/libero_scene2 --out results/libero_scene2.csv
```

The summarizer checks query order, seeds, checkpoint hashes and planner settings;
it rejects incomplete groups and runtime errors. Use matching `--methods`,
`--horizons`, or `--full-episode` options for a smaller completed table. The CSV
contains all three group counts and their sample standard deviation in percentage
points. These are evaluation groups, not independent training seeds.

The direct RoboTwin evaluator saves its proposal RNG at query boundaries.
Repeating the identical command resumes only an ordered, error-free prefix with
matching input hashes. Older outputs without RNG state require a fresh output.
The RoboTwin table launcher recycles the process after eight new queries and
resumes from the saved proposal RNG to release simulator resources. LIBERO
currently requires a fresh output path.

Native SAGE uses K64, a single LeWM ranking pass, 15 raw actions per replan, and
120 recovery actions. Prior Top selects the highest-weight GMM component mean.
LIBERO checks success every action and stops after the current chunk; RoboTwin
stops immediately on official success. Final goals always remain the stored
episode-terminal images.

The archived Base definitions are not interchangeable: LIBERO uses a cumulative
Gaussian residual proposal anchored at the preceding action; RoboTwin anchors
it at the currently observed joint state. Both use standard deviation equal to
5% of the native action range per increment. Neither is IID zero Gaussian.

The evaluator replays the collection prefix and checks handoff observations.
RoboTwin checks joint commands against the stored stride-1 record and reports
three-view image differences. JPEG compression and rendering can produce
nonzero pixel differences; these diagnostics are not an exact-physics
certificate. Cross-machine full-grid acceptance remains pending.
For Caddy, the collection runtime requires NumPy 2.2.6 and MuJoCo 2.3.7.
An environment with the same name but newer MuJoCo is not interchangeable:
small physics differences amplify at contact. Do not relax replay tolerances
or assume an environment name identifies its installed packages. Record
`--numerics` output with an evaluation run; it is diagnostic, not a certificate.
Full cross-machine Caddy tail acceptance is still pending.
The candidate also supports `--handoff-anchors <index.json>` for SHA-locked
prefix corrections exported from the collection runtime. These verify the
query, compiled model, integration-state layout, and replay time before
restoring physics/controller state. Images are rendered afresh and checked
against the unchanged certificate. Observation clocks are preserved during
that check. Anchors still require prefix replay; they are not generic state
banks. The complete 600-query Caddy tail anchor inventory has been exported;
full online acceptance of that inventory remains pending.
To create a small acceptance slice on the collection-compatible machine:

```bash
python scripts/export_libero_anchors.py \
  --queries data/manifests/libero_caddy/tail.json \
  --data-root datasets/libero_caddy --libero-root "$LIBERO_ROOT" \
  --out-dir anchors/libero_caddy --query-start 0 --num-queries 1 --physics-only
```

Export verifies every prefix proprio frame exactly. `--physics-only` avoids
rendering during export, not during online acceptance. Transfer the resulting
directory intact and pass its `index.json` to the evaluator. Missing anchors
are errors, never silently replaced with uncorrected queries.

It does not assume flattened simulator state is a complete cross-machine
snapshot.

## LIBERO Component Training

The terminal recipe enumerates clean local action windows with the episode's
last frame as the far goal. The generator uses remaining horizons of at least
30 steps; the prior additionally includes the direct-final 15-step tail.
It does not round remaining horizons to multiples of 15. Local action modes
0 and 1 are retained; random intervention mode 2 is excluded from the target
action chunk. The world model stays frozen during component training.

Use the matching full training dataset, not the evaluation-only mirrors above.
Rebuild portable indices from the ordered, SHA-checked shard lists. For Caddy:

```bash
python -m sage.build_native_index --manifest data/training/libero_caddy/train_shards.json \
  --data-root datasets/libero_caddy_train --output datasets/libero_caddy_train/indices/train.h5
python -m sage.build_native_index --manifest data/training/libero_caddy/validation_shards.json \
  --data-root datasets/libero_caddy_train --output datasets/libero_caddy_train/indices/validation.h5
```

The data directory must contain the `episodes/` paths listed in those manifests.
For Scene2 use the corresponding lists in `data/training/libero_scene2/`.
Move the indices and episodes together; the index does not embed server paths.
Then run these stages in order:

```bash
python -m sage.train_native --suite libero_scene2 --stage cache \
  --train "$TRAIN_H5" --validation "$VAL_H5" \
  --world-model checkpoints/libero_scene2/world_model/weights.pt --out runs/scene2
python -m sage.train_native --suite libero_scene2 --stage generator \
  --train "$TRAIN_H5" --validation "$VAL_H5" \
  --world-model checkpoints/libero_scene2/world_model/weights.pt --out runs/scene2
python -m sage.train_native --suite libero_scene2 --stage prior \
  --train "$TRAIN_H5" --validation "$VAL_H5" \
  --world-model checkpoints/libero_scene2/world_model/weights.pt --out runs/scene2
```

The launcher checks training-window counts and ordered-array SHA256. Both suites use a
768-wide, four-layer generator for six epochs and a 512-wide, three-layer GMM8
prior for four epochs. The prior uses the generator selected by validation,
not automatically its last epoch. `configs/native_training.json` records the
selected historical epochs and window counts. Cache construction and training
entry points are integrated; full training-data packaging and GPU training
acceptance are still pending. Both LIBERO component CLIs have passed a one-batch
training/validation/checkpoint handoff test; this is not a full retraining result.

## RoboTwin Component Training

Use the ordered stride-1 training and validation manifests and the matching
three-view world model. These entry points preserve every raw frame, terminal
goal, and eligible 15-action window; they do not expand a stride-5 cache.
Ordered manifests are in `data/training/robotwin_a2b/`; downloadable archives
are still being prepared. Set `TRAIN_MANIFEST` and `VAL_MANIFEST` to the two
JSON files there.

```bash
python -m sage.train.robotwin_cache --manifest "$TRAIN_MANIFEST" --data-root "$ROBOTWIN_DATA" \
  --world-model checkpoints/robotwin_a2b/world_model/weights.pt --out cache/train_flat.npy
python -m sage.train.robotwin_cache --manifest "$VAL_MANIFEST" --data-root "$ROBOTWIN_DATA" \
  --world-model checkpoints/robotwin_a2b/world_model/weights.pt --out cache/validation_flat.npy
python -m sage.train.robotwin_pack --manifest "$TRAIN_MANIFEST" --data-root "$ROBOTWIN_DATA" \
  --flat-latents cache/train_flat.npy --out-dir cache/train \
  --action-stats cache/action_stats.npz --fit-action-stats
python -m sage.train.robotwin_pack --manifest "$VAL_MANIFEST" --data-root "$ROBOTWIN_DATA" \
  --flat-latents cache/validation_flat.npy --out-dir cache/validation \
  --action-stats cache/action_stats.npz
python -m sage.train.robotwin_terminal --stage generator --recipe configs/robotwin_training.json \
  --train-cache cache/train --val-cache cache/validation --out-dir runs/robotwin_generator
python -m sage.train.robotwin_terminal --stage prior --recipe configs/robotwin_training.json \
  --train-cache cache/train --val-cache cache/validation --out-dir runs/robotwin_prior \
  --generator-checkpoint runs/robotwin_generator/epoch1.pt
```

The generator is 896-wide with four layers and eight heads (six epochs).
The GMM8 prior is 512-wide with three layers and eight heads (four epochs).
Both use batch size 512 and AdamW at learning rate and weight decay `1e-4`.
The historical released bundle uses generator epoch 1 and prior epoch 4;
`best.pt` is not an interchangeable replacement. The prior is conditioned on
the frozen generator's predictions, with direct final-goal routing at H=15.
Validation uses training action statistics. Ordered sampling and cache packing
have passed exact comparisons with the source recipe; both training entry
points have passed one-batch CPU training, validation and checkpoint tests.
This is not evidence of full retraining equivalence.
