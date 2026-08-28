# SAGE

Official implementation of **[SAGE: Subgoal-Conditioned Action Generation for
Latent World Model Planning](https://arxiv.org/abs/2607.17973)**.

Checkpoints: [CLTRAY/SAGE](https://huggingface.co/CLTRAY/SAGE)

## Overview

SAGE plans with a frozen latent world model. A subgoal generator first predicts
a reachable local target, then a trajectory-level GMM proposes coherent action
options. The world model ranks and refines these options with CEM.

This release reproduces the five PushT and OGBench Cube component methods in
the paper.

## Installation

```bash
conda env create -f environment.yml
conda activate sage
pip install -e .
```

The paper environment uses Python 3.10, PyTorch 2.5.1, CUDA 12.1, and cuDNN 9.

## Data

Prepare the PushT Lance dataset and OGBench Cube HDF5 dataset, then set:

```bash
export PUSHT_DATASET=/path/to/pusht_expert_train.lance
export CUBE_DATASET=/path/to/cube_single_expert.h5
```

Fixed splits and evaluation manifests are included under `data/`.

## Reproduce Results

One command downloads verified checkpoints, evaluates all methods, and writes
the JSON, CSV, and Markdown tables to `results/`:

```bash
PUSHT_DATASET=/path/to/pusht_expert_train.lance \
CUBE_DATASET=/path/to/cube_single_expert.h5 \
bash scripts/reproduce_main.sh
```

To download checkpoints only:

```bash
python scripts/download_checkpoints.py --repo-id CLTRAY/SAGE
```

## Training

```bash
bash scripts/train_pusht.sh
bash scripts/train_cube.sh
```

Each recipe trains the subgoal generator, local-goal action prior, and matched
far-goal action prior. Exact architectures, sampling rules, and selected epochs
are recorded in `configs/training.json`.

## Evaluation

```bash
bash scripts/eval_pusht.sh
bash scripts/eval_cube.sh
python scripts/summarize_results.py
```

The paper protocol uses 300 candidates, 30 CEM rounds, 30 elites, three-frame
history, and seeds 32, 42, and 52. Full schedules are in `configs/paper.json`.

| Method | Goal | Action proposal | World model |
|:---|:---|:---|:---|
| Base CEM | final | Gaussian | rank and refine |
| Far-Goal Prior + CEM | final | far-goal prior | rank and refine |
| LeWM + Generator | generated | Gaussian | rank and refine |
| Generator + Prior Top | generated | top prior mode | unused |
| SAGE | generated | local prior | rank and refine |

Evaluate selected rows with `METHODS`, for example:

```bash
METHODS="base_cem sage" bash scripts/eval_pusht.sh
```

## SAGE Results

Mean success rate over seeds 32, 42, and 52:

| Benchmark | H25 | H50 | H75 | H100 | H125 | H150 |
|:---|---:|---:|---:|---:|---:|---:|
| PushT | 94.0 | 81.3 | 81.3 | 72.7 | 68.7 | 64.7 |
| OGBench Cube | 98.7 | 76.0 | 86.0 | 85.3 | 77.3 | 67.3 |

## Citation

```bibtex
@article{cheng2026sage,
  title={SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning},
  author={Cheng, Letian and Zhang, Qi and Wang, Qixun and Wang, Yisen},
  journal={arXiv preprint arXiv:2607.17973},
  year={2026}
}
```
