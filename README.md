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
```

Raw CSEDM files, processed framework logs, precomputed train/validation/test **split files**, LLM caches, and checkpoints are **not** included — see [supplementary/DATA_ACCESS.md](supplementary/DATA_ACCESS.md).

## Reproduction highlights

- **Cohorts:** F19 (primary), S19 (full tier)
- **Split:** student-level 80/10/10, seeds 0–9 — **not** shipped as downloadable split lists; reproduced deterministically at runtime (see below)
- **Backbones:** DKT, DKVMN, SAKT, AKT, qDKT, QIKT, SimpleKT, SparseKT
- **Backbone baseline:** `problem_onehot` (v0)
- **Full repro:** v5 on rule-enriched logs (Q + Process + mechanism Error)

Details: [docs/experiment_protocol.md](docs/experiment_protocol.md), [docs/ablation_ladder.md](docs/ablation_ladder.md).

## Train/validation/test splits

Precomputed split list files are **not** provided in the paper supplementary materials or in this repository. Instead, splits are **reproduced in code** when you run training:

- Function: `evipkt.dataset.split_students` (80% train / 10% validation / 10% test, PyTorch `randperm` with fixed seed)
- Seeds: `--seed 0` … `9` (same as the paper’s 10-run protocol)
- Scope: student IDs taken from your framework logs for each cohort (F19 and S19 are split **separately**; do not merge cohorts)

Given the same enriched logs and seed, you obtain the same train/valid/test partition as our experiments. Rules: [supplementary/filtering_rules.md](supplementary/filtering_rules.md).

## License

See repository license file (if present) or contact the authors.
