# Student train / validation / test splits

Precomputed **student-level** partitions for the paper’s 10-seed protocol (80% / 10% / 10%).

| Cohort | Students | Train / valid / test (each seed) | Files |
|--------|---------:|----------------------------------|-------|
| F19 | 506 | 404 / 50 / 52 | `f19/seed_0.json` … `seed_9.json` |
| S19 | 413 | 330 / 41 / 42 | `s19/seed_0.json` … `seed_9.json` |

Each JSON file lists hashed `subject_id` values for train, validation, and test. Training code loads these automatically when the framework log path indicates F19 (`data/processed/…`) or S19 (`data/processed_s19/…`) and `--seed` matches the filename.

**Regenerate** (after rebuilding framework logs from CSEDM):

```bash
python scripts/export_student_splits.py
```

Requires enriched logs at `data/processed/framework_logs_first_process_mechanism.jsonl` and `data/processed_s19/framework_logs_first_process_mechanism.jsonl`.
