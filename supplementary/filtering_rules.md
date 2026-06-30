# Filtering and Split Rules

- **Submission mode:** `first` — earliest `Run.Program` per (student, problem) for main experiments.
- **Label:** `pkt_label = 1` iff unit-test score ≥ 1.0 (all tests pass).
- **Split:** student-level 80% train / 10% validation / 10% test within each semester cohort.
- **Seeds:** 10 independent splits (seeds 0–9).
- **Training:** 8 epochs, batch size 64, checkpoint selected by validation AUC.
