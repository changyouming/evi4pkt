# CSEDM / CodeWorkout Data Access

The raw CSEDM/CodeWorkout F19 and S19 programming-learning datasets are available from the original dataset provider, subject to its access conditions.

After obtaining the releases:

1. Place F19 ProgSnap2 under `data/raw/F19/` (e.g. `MainTable.csv`, `CodeStates.csv` under `All/`).
2. Place S19 under `data/raw/S19/` (full tier only).
3. Run `bash scripts/enrich_rule_based.sh f19` (and `s19` for full tier).

This repository does **not** ship raw CSEDM files, processed framework logs, LLM caches, or model checkpoints.
