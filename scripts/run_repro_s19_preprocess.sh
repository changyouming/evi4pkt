#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/enrich_rule_based.sh f19
bash scripts/enrich_rule_based.sh s19
