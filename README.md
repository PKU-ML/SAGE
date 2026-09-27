# SAGE

Official implementation of **[SAGE: Subgoal-Conditioned Action Generation for
Latent World Model Planning](https://arxiv.org/abs/2607.17973)**.

[Paper](https://arxiv.org/abs/2607.17973) | [Checkpoints](https://huggingface.co/CLTRAY/SAGE)

SAGE plans with a frozen latent world model. A subgoal generator predicts
intermediate targets, an action prior proposes goal-directed action sequences,
and the world model evaluates and refines them before execution.

## Installation

```bash
conda env create -f environment.yml
conda activate sage
pip install -e .
```

Run from the repository root on Linux. LIBERO and RoboTwin use separate
[simulator environments](NATIVE.md#environment).

## Reproduce Results

Prepare the PushT Lance and OGBench Cube HDF5 datasets, then run:

```bash
PUSHT_DATASET=/path/to/pusht_expert_train.lance \
CUBE_DATASET=/path/to/cube_single_expert.h5 \
bash scripts/reproduce_main.sh
```

This downloads the original SAGE component checkpoints, runs the PushT/Cube
component evaluations, and saves results under `results/`.
Fixed splits and query manifests are included in `data/`.

To evaluate selected methods:

```bash
export PUSHT_DATASET=/path/to/pusht_expert_train.lance
METHODS="base_cem sage" bash scripts/eval_pusht.sh
```

| Experiments | Instructions |
|:---|:---|
| PushT / OGBench Cube, visual-only, DINO-WM | [Protocols](REPRODUCING.md) |
| LIBERO / RoboTwin | [Setup and evaluation](NATIVE.md) |
| Datasets and manifests | [Data layout](DATA.md) |

## Training

```bash
bash scripts/train_pusht.sh
bash scripts/train_cube.sh
```

These recipes train the subgoal generator and action priors on a frozen world
model. Architectures and sampling settings are in `configs/training.json`;
additional recipes are documented in [Protocols](REPRODUCING.md#training)
and [Native Suites](NATIVE.md).

## Citation

```bibtex
@article{cheng2026sage,
  title={SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning},
  author={Cheng, Letian and Zhang, Qi and Wang, Qixun and Wang, Yisen},
  journal={arXiv preprint arXiv:2607.17973},
  year={2026}
}
```
