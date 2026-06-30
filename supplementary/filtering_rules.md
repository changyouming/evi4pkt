# Filtering and Split Rules

- **Submission mode:** `first` — earliest `Run.Program` per (student, problem) for main experiments.
- **Label:** `pkt_label = 1` iff unit-test score ≥ 1.0 (all tests pass).
- **Split:** student-level 80% train / 10% validation / 10% test within each semester cohort (F19 and S19 logs are split separately).
- **Seeds:** 10 independent splits (seeds 0–9).
- **Split files:** not distributed as standalone CSV/JSON lists (neither with the paper supplementary materials nor in this repo). Reproduce via `evipkt.dataset.split_students(students, seed=…)` when loading framework logs; same logs + same seed → same partition.
- **Training:** 8 epochs, batch size 64, checkpoint selected by validation AUC.
