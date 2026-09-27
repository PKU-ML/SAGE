#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python}"
: "${CUBE_DATASET:?Set CUBE_DATASET}"
: "${DINOWM_POLICY:?Set DINOWM_POLICY to the released Cube world model}"
: "${OUTPUT_ROOT:=runs/dino-cube}"
: "${DEVICE:=cuda:0}"
cache_args=()
if [[ -n "${FRAME_LATENT_CACHE:-}" ]]; then
  cache_args=(--frame-latent-cache "$FRAME_LATENT_CACHE")
fi
common=(--dataset "$CUBE_DATASET" --policy "$DINOWM_POLICY"
  --split data/splits/ogbench_cube_single_split_seed42.json
  --context-len 3 --frameskip 5 --image-size 224 --lowdim-keys observation
  --goal-offsets 15 20 25 30 40 45 50 60 65 75 90 100 115 125 140 150
  --hidden-dim 768 --depth 4 --num-heads 8
  --dense-joint-sampling --dense-balance-goals --dense-allow-repeats
  --max-train-windows 600000 --max-val-windows 60000
  --no-pin-memory --lr 1e-4 --weight-decay 1e-4 --grad-clip 1.0
  --device "$DEVICE" --bf16 --no-resume)

"$PYTHON" -m sage.train.cube_generator "${common[@]}" "${cache_args[@]}" \
  --subgoal-offset 25 --subgoal-offsets 15 20 25 --pooling decoder \
  --predict-residual-from goal --smooth-l1-beta 0.05 --cosine-weight 0.1 \
  --batch-size 512 --num-workers 8 --seed 922 \
  --out-dir "$OUTPUT_ROOT/generator" --epochs 6 \
  --max-train-pairs 3000000 --max-val-pairs 120000

"$PYTHON" -m sage.train.cube_action_prior "${common[@]}" "${cache_args[@]}" \
  --action-offsets 15 20 25 --num-modes 8 --prior-goal-source local \
  --generated-subgoal-ratio 1.0 --eval-use-generated-subgoal \
  --mode-l1-weight 0.05 --coverage-threshold 0.1 --eval-samples 64 \
  --batch-size 256 --num-workers 6 --seed 925 \
  --subgoal-generator-checkpoint "$OUTPUT_ROOT/generator/best.pt" \
  --out-dir "$OUTPUT_ROOT/prior" --epochs 4 \
  --max-train-examples 750000 --max-val-examples 120000
