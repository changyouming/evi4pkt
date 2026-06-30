#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
SEEDS="0 1 2 3 4 5 6 7 8 9"
BACKBONES="DKT DKVMN SAKT AKT qDKT QIKT SimpleKT SparseKT"

mkdir -p runs
$PY scripts/run_ablation_ladder.py \
  --levels v0 v5 \
  --backbones $BACKBONES \
  --seeds $SEEDS \
  --epochs 8 \
  --batch-size 64 \
  --device cuda \
  --resume
