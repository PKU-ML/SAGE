#!/usr/bin/env bash
set -euo pipefail

: "${CUBE_DATASET:?Set CUBE_DATASET to the OGBench Cube dataset}"
: "${LEWM_POLICY:=quentinll/lewm-cube}"
: "${OUTPUT_ROOT:=outputs/cube}"
: "${DEVICE:=cuda:0}"
: "${FRAME_CACHE:=}"

COMMON_OFFSETS=(15 20 25 30 40 45 50 60 65 75 90 100 115 125 140 150)
CACHE_ARGS=()
if [[ -n "$FRAME_CACHE" ]]; then
  CACHE_ARGS=(--frame-latent-cache "$FRAME_CACHE")
fi

python -m sage.train.cube_generator \
  --dataset "$CUBE_DATASET" \
  --split data/splits/ogbench_cube_single_split_seed42.json \
  --policy "$LEWM_POLICY" \
  --out-dir "$OUTPUT_ROOT/generator" \
  --device "$DEVICE" \
  "${CACHE_ARGS[@]}" \
  --epochs 6 \
  --batch-size 128 \
  --num-workers 4 \
  --no-pin-memory \
  --hidden-dim 512 \
  --num-heads 8 \
  --depth 4 \
  --pooling decoder \
  --predict-residual-from goal \
  --context-len 3 \
  --frameskip 5 \
  --subgoal-offsets 15 20 25 \
  --goal-offsets "${COMMON_OFFSETS[@]}" \
  --max-train-windows 200000 \
  --max-val-windows 20000 \
  --max-train-pairs 400000 \
  --max-val-pairs 40000 \
  --dense-joint-sampling \
  --dense-balance-goals \
  --dense-allow-repeats \
  --seed 422 \
  --bf16 \
  --no-resume

python -m sage.train.cube_action_prior \
  --dataset "$CUBE_DATASET" \
  --split data/splits/ogbench_cube_single_split_seed42.json \
  --policy "$LEWM_POLICY" \
  --out-dir "$OUTPUT_ROOT/action_prior" \
  --device "$DEVICE" \
  "${CACHE_ARGS[@]}" \
  --epochs 5 \
  --save-epochs 3 \
  --batch-size 128 \
  --num-workers 4 \
  --no-pin-memory \
  --hidden-dim 512 \
  --num-heads 8 \
  --depth 3 \
  --num-modes 8 \
  --context-len 3 \
  --frameskip 5 \
  --action-offsets 15 20 25 \
  --goal-offsets "${COMMON_OFFSETS[@]}" \
  --subgoal-generator-checkpoint "$OUTPUT_ROOT/generator/latest.pt" \
  --generated-subgoal-ratio 1.0 \
  --eval-use-generated-subgoal \
  --max-train-examples 400000 \
  --max-val-examples 40000 \
  --max-train-windows 200000 \
  --max-val-windows 20000 \
  --dense-joint-sampling \
  --dense-balance-goals \
  --dense-allow-repeats \
  --seed 423 \
  --bf16 \
  --no-resume

python -m sage.train.cube_action_prior \
  --dataset "$CUBE_DATASET" \
  --split data/splits/ogbench_cube_single_split_seed42.json \
  --policy "$LEWM_POLICY" \
  --out-dir "$OUTPUT_ROOT/far_action_prior" \
  --device "$DEVICE" \
  "${CACHE_ARGS[@]}" \
  --epochs 3 \
  --save-epochs 3 \
  --batch-size 128 \
  --num-workers 4 \
  --no-pin-memory \
  --hidden-dim 512 \
  --num-heads 8 \
  --depth 3 \
  --num-modes 8 \
  --context-len 3 \
  --frameskip 5 \
  --action-offsets 15 20 25 \
  --goal-offsets "${COMMON_OFFSETS[@]}" \
  --prior-goal-source far \
  --max-train-examples 400000 \
  --max-val-examples 40000 \
  --max-train-windows 300000 \
  --max-val-windows 30000 \
  --dense-joint-sampling \
  --dense-balance-goals \
  --dense-allow-repeats \
  --seed 423 \
  --bf16 \
  --no-resume

echo "Paper checkpoints:"
echo "  $OUTPUT_ROOT/generator/latest.pt (epoch 6)"
echo "  $OUTPUT_ROOT/action_prior/epoch_003.pt"
echo "  $OUTPUT_ROOT/far_action_prior/epoch_003.pt"
