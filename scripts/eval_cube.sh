#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python}"
: "${CUBE_DATASET:?Set CUBE_DATASET to the OGBench Cube dataset}"
: "${LEWM_POLICY:=quentinll/lewm-cube}"
: "${GENERATOR:=checkpoints/cube_generator.pt}"
: "${ACTION_PRIOR:=checkpoints/cube_action_prior.pt}"
: "${FAR_ACTION_PRIOR:=checkpoints/cube_far_action_prior.pt}"
: "${OUTPUT_ROOT:=results/cube}"
: "${DEVICE:=cuda:0}"
: "${METHODS:=base_cem far_goal_prior_cem lewm_generator generator_prior_top sage}"
: "${SEEDS:=32 42 52}"
: "${HORIZONS:=25 50 75 100 125 150}"

for method in $METHODS; do
  generator_args=()
  prior="$ACTION_PRIOR"
  case "$method" in
    base_cem) ;;
    far_goal_prior_cem) prior="$FAR_ACTION_PRIOR" ;;
    lewm_generator|generator_prior_top|sage)
      generator_args=(--generator "$GENERATOR")
      ;;
    *) echo "Unknown method: $method" >&2; exit 2 ;;
  esac
  for seed in $SEEDS; do
    for horizon in $HORIZONS; do
      "$PYTHON" -m sage.eval.cube \
        --method "$method" \
        --dataset "$CUBE_DATASET" \
        --policy "$LEWM_POLICY" \
        "${generator_args[@]}" \
        --action-prior "$prior" \
        --manifest "data/manifests/cube/seed${seed}/h${horizon}.json" \
        --paper-config configs/paper.json \
        --seed "$seed" \
        --device "$DEVICE" \
        --out-dir "$OUTPUT_ROOT/$method/seed${seed}/h${horizon}"
    done
  done
done
