# Evi4PKT — Reproduction Guide


Obtain CSEDM F19/S19 data (not shipped here). **Student split lists** (seeds 0–9) are bundled under `data/splits/`; see [data/splits/README.md](data/splits/README.md).

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[repro]"
pytest tests/ -q
```

## 2. Obtain CSEDM data

See [supplementary/DATA_ACCESS.md](supplementary/DATA_ACCESS.md).

## 3. Build framework logs (rule-based)

```bash
# F19
bash scripts/enrich_rule_based.sh f19

# S19 (full tier)
bash scripts/enrich_rule_based.sh s19
```

Output: `data/processed/framework_logs_first_process_mechanism.jsonl` (F19) or `data/processed_s19/framework_logs_first_process_mechanism.jsonl` (S19).

## 4. Train eight backbones (TABLE II backbone vs Full repro)

```bash
bash scripts/run_repro_eight_backbone.sh
```

Or manually:

```bash
python scripts/run_ablation_ladder.py \
  --levels v0 v5 \
  --backbones DKT DKVMN SAKT AKT qDKT QIKT SimpleKT SparseKT \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --epochs 8 --batch-size 64 --device cuda --resume
```

- **(Backbone):** `problem_onehot`
- **(Full repro):** `problem_plus_q_process_code_mechanism` on rule-enriched logs

## 5. Summarize results

```bash
python scripts/build_eight_backbone_plugplay_table.py
python scripts/build_ablation_plan_table.py
```

## 6. Specialized PKT baselines (optional)

After building F19/S19 rule-based logs and code2vec/codebert caches:

```bash
python scripts/run_classic_pkt_f19_s19.py --cohorts f19 s19 --resume
python scripts/build_classic_pkt_summary.py
```

## References

- [docs/experiment_protocol.md](docs/experiment_protocol.md)
- [docs/ablation_ladder.md](docs/ablation_ladder.md)
- [supplementary/evidence_schema.md](supplementary/evidence_schema.md)
