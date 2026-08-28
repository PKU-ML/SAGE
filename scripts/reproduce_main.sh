#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python}"
: "${SAGE_HF_REPO:=CLTRAY/SAGE}"
: "${SAGE_HF_REVISION:=main}"
: "${PUSHT_DATASET:?Set PUSHT_DATASET to the PushT Lance dataset}"
: "${CUBE_DATASET:?Set CUBE_DATASET to the OGBench Cube dataset}"

"$PYTHON" scripts/audit_release.py
"$PYTHON" scripts/download_checkpoints.py \
  --repo-id "$SAGE_HF_REPO" \
  --revision "$SAGE_HF_REVISION"

bash scripts/eval_pusht.sh
bash scripts/eval_cube.sh
"$PYTHON" scripts/summarize_results.py \
  --root results \
  --out results/component_table.json
