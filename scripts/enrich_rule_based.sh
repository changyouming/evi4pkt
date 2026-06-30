#!/usr/bin/env bash
# Rule-based enrichment: preprocess -> +Process -> +Mechanism (no LLM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
COHORT="${1:-f19}"

if [[ "$COHORT" == "f19" ]]; then
  RAW="data/raw/F19/All"
  OUT="data/processed"
elif [[ "$COHORT" == "s19" ]]; then
  RAW="data/raw/S19/All"
  OUT="data/processed_s19"
else
  echo "Usage: $0 {f19|s19}" >&2
  exit 1
fi

mkdir -p "$OUT"
$PY scripts/preprocess_csedm.py \
  --csedm-root "$RAW" \
  --prompts-csv data/metadata/problem_prompts.csv \
  --submission-mode first \
  --out-dir "$OUT"

$PY scripts/enrich_framework_logs_process.py \
  --in-path "$OUT/framework_logs_first.jsonl"

$PY scripts/enrich_framework_logs_error_mechanism.py \
  --in-path "$OUT/framework_logs_first_process.jsonl" \
  --out-path "$OUT/framework_logs_first_process_mechanism.jsonl"

echo "Wrote: $OUT/framework_logs_first_process_mechanism.jsonl"
