#!/usr/bin/env bash
set -euo pipefail

: "${PUSHT_DATASET:?Set PUSHT_DATASET to the PushT Lance dataset}"
: "${LEWM_POLICY:=pusht/lewm}"
: "${OUTPUT_ROOT:=outputs/pusht}"
: "${DEVICE:=cuda:0}"

COMMON_OFFSETS=(15 20 25 30 40 45 50 60 65 75 90 100 115 125 140 150)

python -m sage.train.pusht_generator \
  --dataset "$PUSHT_DATASET" \
  --split data/splits/pusht_episode_split_seed42.json \
  --policy "$LEWM_POLICY" \
  --out-dir "$OUTPUT_ROOT/generator" \
  --device "$DEVICE" \
  --epochs 6 \
  --batch-size 128 \
  --num-workers 4 \
  --no-pin-memory \
  --hidden-dim 512 \
  --num-heads 8 \
  --depth 4 \
  --pooling decoder \
  --predict-residual-from goal \
  --goal-condition-mode goal \
  --history-len 3 \
  --frameskip 5 \
  --subgoal-offsets 15 20 25 \
  --goal-offsets "${COMMON_OFFSETS[@]}" \
  --max-train-windows 0 \
  --max-val-windows 0 \
  --max-train-pairs 400000 \
  --max-val-pairs 40000 \
  --dense-joint-sampling \
  --dense-balance-goals \
  --seed 148 \
  --bf16 \
  --no-resume

python -m sage.train.pusht_action_prior \
  --dataset "$PUSHT_DATASET" \
  --split data/splits/pusht_episode_split_seed42.json \
  --policy "$LEWM_POLICY" \
  --out-dir "$OUTPUT_ROOT/action_prior" \
  --device "$DEVICE" \
  --epochs 5 \
  --save-epochs 3 \
  --batch-size 128 \
  --num-workers 4 \
  --no-pin-memory \
  --architecture transformer \
  --hidden-dim 512 \
  --num-heads 8 \
  --depth 3 \
  --num-modes 8 \
  --history-len 3 \
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
  --seed 149 \
  --bf16 \
  --no-resume

python -m sage.train.pusht_action_prior \
  --dataset "$PUSHT_DATASET" \
  --split data/splits/pusht_episode_split_seed42.json \
  --policy "$LEWM_POLICY" \
  --out-dir "$OUTPUT_ROOT/far_action_prior" \
  --device "$DEVICE" \
  --epochs 3 \
  --save-epochs 3 \
  --batch-size 128 \
  --num-workers 4 \
  --no-pin-memory \
  --architecture transformer \
  --hidden-dim 512 \
  --num-heads 8 \
  --depth 3 \
  --num-modes 8 \
  --pooling attention \
  --history-len 3 \
  --frameskip 5 \
  --action-offsets 15 20 25 \
  --goal-offsets "${COMMON_OFFSETS[@]}" \
  --conditioning-goal-source far \
  --max-train-examples 400000 \
  --max-val-examples 40000 \
  --max-train-windows 0 \
  --max-val-windows 0 \
  --dense-joint-sampling \
  --dense-balance-goals \
  --seed 149 \
  --bf16 \
  --no-resume

echo "Paper checkpoints:"
echo "  $OUTPUT_ROOT/generator/latest.pt (epoch 6)"
echo "  $OUTPUT_ROOT/action_prior/epoch_003.pt"
echo "  $OUTPUT_ROOT/far_action_prior/epoch_003.pt"
