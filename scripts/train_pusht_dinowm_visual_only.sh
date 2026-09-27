#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python}"
: "${PUSHT_DATASET:?Set PUSHT_DATASET to the PushT Lance dataset}"
: "${DINOWM_POLICY:?Set DINOWM_POLICY to the converted official DINO-WM checkpoint}"
: "${OUTPUT_ROOT:=outputs/pusht_dinowm_visual_only}"
: "${DEVICE:=cuda:0}"

COMMON_OFFSETS=(15 20 25 30 40 45 50 60 65 75 90 100 115 125 140 150)

"$PYTHON" -m sage.train.pusht_generator \
  --dataset "$PUSHT_DATASET" \
  --split data/splits/pusht_episode_split_seed42.json \
  --policy "$DINOWM_POLICY" \
  --out-dir "$OUTPUT_ROOT/generator" \
  --device "$DEVICE" \
  --epochs 6 \
  --batch-size 128 \
  --num-workers 4 \
  --no-pin-memory \
  --no-persistent-workers \
  --prefetch-factor 1 \
  --hidden-dim 512 \
  --num-heads 8 \
  --depth 4 \
  --pooling decoder \
  --predict-residual-from goal \
  --goal-condition-mode goal \
  --history-len 4 \
  --image-normalization dinowm \
  --frameskip 5 \
  --subgoal-offsets 15 20 25 \
  --goal-offsets "${COMMON_OFFSETS[@]}" \
  --max-train-windows 0 \
  --max-val-windows 0 \
  --max-train-pairs 400000 \
  --max-val-pairs 40000 \
  --dense-joint-sampling \
  --dense-balance-goals \
  --no-lowdim \
  --seed 148 \
  --bf16 \
  --no-resume

"$PYTHON" -m sage.train.pusht_action_prior \
  --dataset "$PUSHT_DATASET" \
  --split data/splits/pusht_episode_split_seed42.json \
  --policy "$DINOWM_POLICY" \
  --out-dir "$OUTPUT_ROOT/action_prior" \
  --device "$DEVICE" \
  --epochs 4 \
  --save-epochs 4 \
  --batch-size 128 \
  --num-workers 4 \
  --no-pin-memory \
  --no-persistent-workers \
  --prefetch-factor 1 \
  --architecture transformer \
  --hidden-dim 512 \
  --num-heads 8 \
  --depth 3 \
  --num-modes 8 \
  --history-len 4 \
  --image-normalization dinowm \
  --frameskip 5 \
  --action-offsets 15 20 25 \
  --goal-offsets "${COMMON_OFFSETS[@]}" \
  --subgoal-generator-checkpoint "$OUTPUT_ROOT/generator/latest.pt" \
  --generated-subgoal-ratio 1.0 \
  --eval-use-generated-subgoal \
  --max-train-examples 400000 \
  --max-val-examples 40000 \
  --max-train-windows 0 \
  --max-val-windows 0 \
  --dense-joint-sampling \
  --dense-balance-goals \
  --no-lowdim \
  --seed 149 \
  --bf16 \
  --no-resume

echo "DINO-WM visual-only checkpoints:"
echo "  $OUTPUT_ROOT/generator/latest.pt"
echo "  $OUTPUT_ROOT/action_prior/epoch_004.pt"
