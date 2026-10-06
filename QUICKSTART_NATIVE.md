# Reproducing LIBERO and RoboTwin

[Code](https://github.com/PKU-ML/SAGE) provides environments, evaluation commands,
and locked query manifests. [Models](https://huggingface.co/CLTRAY/SAGE) provides
matching LeWM, subgoal generator, and GMM prior checkpoints.
[Datasets](https://huggingface.co/datasets/CLTRAY/SAGE-data) provides evaluation,
training, and validation archives. Evaluation alone does not require training data.

## 1. Install

Run on Linux with an NVIDIA GPU:

```bash
git clone https://github.com/PKU-ML/SAGE.git
cd SAGE
conda env create -f runtime/libero/environment.yml
conda activate sage-libero
pip install --no-deps -e .
pip install huggingface_hub
```

Install LIBERO with its scene assets at the commit recorded in
`runtime/native_contracts.json`:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
git -C external/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install --no-deps -e external/LIBERO
export LIBERO_ROOT="$PWD/external/LIBERO"
python scripts/check_native_runtime.py libero --gpu --numerics
```

Follow LIBERO's asset installation instructions if assets are absent. Simulator
assets are separate from the trajectory archives below. Preserve the pinned
MuJoCo, NumPy, and robosuite versions; see [runtime details](NATIVE.md#environment).

## 2. Download Matching Models and Evaluation Data

Choose `libero_scene2` or `libero_caddy`:

```bash
export SUITE=libero_scene2
python -m sage.assets --suite "$SUITE" --out-dir checkpoints
python - "$SUITE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="CLTRAY/SAGE-data", repo_type="dataset",
    filename=f"evaluation/{sys.argv[1]}_evaluation.tar",
    local_dir="downloads",
)
PY
python scripts/install_native_dataset.py --suite "$SUITE" --split evaluation \
  --archive "downloads/evaluation/${SUITE}_evaluation.tar" --out "datasets/$SUITE"
```

The model installer checks SHA256 and installs:

```text
checkpoints/<suite>/
  world_model/config.json
  world_model/weights.pt
  generator.pt
  prior.pt
  action_stats.json       # LIBERO
```

`weights.pt` is the world model only. Do not mix components between suites.
Evaluation data sizes are approximately 19.34 GB for Scene2, 8.64 GB for Caddy,
and 1.36 GB for RoboTwin A2B. Allow additional space for extraction.

## 3. Evaluate

Optionally check checkpoint loading without running a simulator:

```bash
python -m sage.native_models "checkpoints/$SUITE"
```

Run SAGE on the locked full-episode LIBERO queries (three groups of 50):

```bash
python -m sage.reproduce_native --suite "$SUITE" --methods sage \
  --libero-root "$LIBERO_ROOT" --data-root "datasets/$SUITE" \
  --checkpoints checkpoints --full-episode --out "results/${SUITE}_full"
python -m sage.summarize_native --suite "$SUITE" --methods sage \
  --full-episode --results "results/${SUITE}_full" --out "results/${SUITE}_full.csv"
```

For the tail-horizon table, replace `--full-episode` with
`--horizons 30 60 90 120` in both commands and choose a new output directory.
Use `--methods base prior_top sage` in both commands to include all components.
`--groups 0` selects the first 50 queries for a smaller run; the table summarizer
requires all three groups.
`--dry-run` on the evaluator prints commands without loading models or data.

Evaluation uses the settings in the released code and the queries in
`data/manifests/<suite>/`; do not resample queries. Use a fresh output directory
for a new LIBERO run. See [evaluation details](NATIVE.md#evaluation) for runtime
and replay requirements.

## RoboTwin A2B

Use the separate environment in `runtime/robotwin/environment.yml` and set
`SUITE=robotwin_a2b`. Download models and evaluation data using step 2.
Install the matching RoboTwin source and scene assets as described in
[NATIVE.md](NATIVE.md#environment), and set `ROBOTWIN_ROOT` to their common root.
These simulator dependencies are not included in the HF trajectory archives.

```bash
python -m sage.reproduce_native --suite robotwin_a2b --methods sage \
  --robotwin-root "$ROBOTWIN_ROOT" --data-root datasets/robotwin_a2b \
  --checkpoints checkpoints --horizons 30 60 90 120 --out results/robotwin_a2b_tail
python -m sage.summarize_native --suite robotwin_a2b --methods sage \
  --horizons 30 60 90 120 --results results/robotwin_a2b_tail \
  --out results/robotwin_a2b_tail.csv
```

The released RoboTwin A2B protocol is tail-only, not full-episode.

## Training Data

Training and validation archives are under `training/` in
[SAGE-data](https://huggingface.co/datasets/CLTRAY/SAGE-data).
Download only `training/<suite>_*`. Training archives are binary-split files:
use the dataset page's `.parts.json` merge and SHA256-check example before
installation. Then install both splits into the same training directory:

```bash
python scripts/install_native_dataset.py --suite "$SUITE" --split train \
  --archive "sage-data/training/${SUITE}_train.tar" --out "datasets/${SUITE}_train"
python scripts/install_native_dataset.py --suite "$SUITE" --split validation \
  --archive "sage-data/training/${SUITE}_validation.tar" --out "datasets/${SUITE}_train"
```

See [LIBERO training](NATIVE.md#libero-component-training) and
[RoboTwin training](NATIVE.md#robotwin-component-training) for index construction,
latent caching, and generator/prior training commands.
