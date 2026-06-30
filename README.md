# Evi4PKT

Evidence-enhanced **Programming Knowledge Tracing** on CSEDM CodeWorkout logs. This repository is the official reproduction bundle for experiments reported in the Evi4PKT paper.

## Overview

Evi4PKT augments standard KT backbones (DKT, SAKT, AKT, etc.) with **Q-matrix–anchored evidence** extracted from programming submissions. Four evidence channels are aligned to the knowledge components (KCs) required by each problem:

| Evidence | Role |
|----------|------|
| **Task (Q)** | Defines the active KC set for the problem |
| **Code** | Signals KCs reflected in the submitted code |
| **Error** | Compile/test feedback tied to task-required KCs |
| **Process** | Historical KC exposure and success before the attempt |

In this bundle, evidence is **early-fused** (concatenated) with problem features and fed into eight KT backbones. Rule-based Process and Error (mechanism) channels are included; LLM-derived Code/Error channels are not redistributed (see [supplementary/evidence_schema.md](supplementary/evidence_schema.md)).

## Quick start

**Requirements:** Python ≥ 3.12, PyTorch ≥ 2.0, CUDA (recommended for training).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[repro]"
pytest tests/ -q
```

Obtain CSEDM F19/S19 data (not shipped here), then follow the step-by-step guide in **[REPRODUCE.md](REPRODUCE.md)**.

## Repository layout

```
evipkt/          Core library (models, evidence, training)
scripts/         Preprocessing, enrichment, experiment runners
tests/           Smoke tests
docs/            Experiment protocol and ablation ladder
supplementary/   Data access, Q matrix, evidence schema
data/metadata/   Q matrix and problem prompts (small metadata only)
data/splits/     Precomputed train/valid/test student lists (F19 & S19, seeds 0–9)
```

Raw CSEDM files, processed framework logs, LLM caches, and checkpoints are **not** included — see [supplementary/DATA_ACCESS.md](supplementary/DATA_ACCESS.md). **Student split lists** for the paper protocol **are** included under `data/splits/`.

## Reproduction highlights

- **Cohorts:** F19 (primary), S19 (full tier)
- **Split:** student-level 80/10/10, seeds 0–9 — precomputed JSON under [`data/splits/`](data/splits/)
- **Backbones:** DKT, DKVMN, SAKT, AKT, qDKT, QIKT, SimpleKT, SparseKT
- **Backbone baseline:** `problem_onehot` (v0)
- **Full repro:** v5 on rule-enriched logs (Q + Process + mechanism Error)

Details: [docs/experiment_protocol.md](docs/experiment_protocol.md), [docs/ablation_ladder.md](docs/ablation_ladder.md).

## Train/validation/test splits

Precomputed split lists ship in **[`data/splits/`](data/splits/)** (also available from this repository):

| Cohort | File pattern | Train / valid / test students |
|--------|--------------|-------------------------------|
| F19 | `f19/seed_{0..9}.json` | 404 / 50 / 52 |
| S19 | `s19/seed_{0..9}.json` | 330 / 41 / 42 |

Training loads the matching file automatically from `--seed` and the framework-log path (`data/processed/` → F19, `data/processed_s19/` → S19). Each JSON lists hashed `subject_id` values; implementation: `evipkt.dataset.resolve_student_split`.

To regenerate after rebuilding logs:

```bash
python scripts/export_student_splits.py
```

Rules: [supplementary/filtering_rules.md](supplementary/filtering_rules.md), [data/splits/README.md](data/splits/README.md).

## License

See repository license file (if present) or contact the authors.
