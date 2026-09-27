# Protocols

Run commands from the repository root. Simulator environments are Linux-based.
Do not mix checkpoints from different world-model encoders.
Native LIBERO/RoboTwin protocols are specified separately in [NATIVE.md](NATIVE.md).

Cube needs a working MuJoCo OpenGL backend. The DINO Cube results use OSMesa;
install the native OSMesa library in addition to the Python packages and set
`MUJOCO_GL=osmesa`. A working EGL run is not evidence of matching that renderer.

| Suite | Queries | Candidates / CEM | Execution budget |
|---|---|---|---|
| LeWM PushT | 3 seeds x 50 per horizon | 300 / 30 | 2H |
| LeWM Cube | 3 seeds x 50 per horizon | 300 / 30 | H |
| Visual-only PushT / Cube | same fixed manifests | 300 / 30 | same as above |
| DINO-WM PushT / Cube | same fixed manifests | 128 / 6 | same as above |

Horizon schedules are fixed in the corresponding `configs/paper*.json`.
The Base planner starts from Gaussian proposals, without an action prior.
Training-split action normalization is explicit in `data/stats/`; it is not
computed from evaluation queries. PushT DINO-WM additionally uses the native
world-model action/proprioception coordinates in its predictor.

`generator_prior_top` uses the highest-weight GMM component mean.
`final_goal_scoring` retains subgoal-conditioned proposals and changes only the
LeWM scoring target to the final goal. It is not the far-goal prior.

## Training

The original component recipes are `scripts/train_pusht.sh` and
`scripts/train_cube.sh`. Visual-only recipes have the `_visual_only` suffix.
For DINO-WM, use `scripts/train_pusht_dinowm_visual_only.sh` or
`scripts/train_cube_dinowm.sh` with the corresponding world-model checkpoint.

DINO Cube uses the full 28-dimensional observation for its generator and prior;
it is not a visual-only result. Its generator trains for six epochs on 3M sampled
pairs; its prior trains for four epochs on 750k examples. The released selected
checkpoints are generator epoch 5 and prior epoch 4.

## Outputs

Each evaluation writes a result JSON with query identifiers, checkpoint identity,
schedule, and per-query success. A reduced query count is marked `subset`, not
`paper`. Keep output directories separate for different models and protocols.

Unit tests run with `python -m pytest tests`. The dataset-backed temporal check is:

```bash
python tests/test_pair_temporal_contract.py --dataset "$CUBE_DATASET"
```

It checks raw-action alignment, terminal-frame indexing, and cached/uncached
sample parity using the original Cube dataset. Numerical paper reproduction
requires the full evaluation, not just these checks.
