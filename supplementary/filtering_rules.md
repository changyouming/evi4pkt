# Filtering and Split Rules

- **Submission mode:** `first` — earliest `Run.Program` per (student, problem) for main experiments.
- **Label:** `pkt_label = 1` iff unit-test score ≥ 1.0 (all tests pass).
- **Split:** student-level 80% train / 10% validation / 10% test within each semester cohort (F19 and S19 logs are split separately).
- **Seeds:** 10 independent splits (seeds 0–9).
- **Split files:** JSON lists in `data/splits/{f19,s19}/seed_{0..9}.json` (downloadable from this repo). Training uses `evipkt.dataset.resolve_student_split`; regenerate with `python scripts/export_student_splits.py` after rebuilding logs.
- **Training:** 8 epochs, batch size 64, checkpoint selected by validation AUC.
