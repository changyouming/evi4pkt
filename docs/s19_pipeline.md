# S19 (Spring 2019) processed logs

**Raw:** `data/S19_All_Release_2_10_22/`  
**Primary training file (v8 Full):** `framework_logs_first_process_mechanism_v8_misused_v8.jsonl`

## Pipeline

```bash
# 1. Preprocess (done)
.venv/bin/python scripts/preprocess_csedm.py \
  --csedm-root data/S19_All_Release_2_10_22 \
  --submission-mode first \
  --out-dir data/processed_s19

# 2. Enrich (or: bash scripts/enrich_s19_v8.sh)
#    process → mechanism v8 → misused v8 (LLM)
```

## Stats (first-attempt)

| | S19 | F19 (reference) |
|---|-----|-----------------|
| Students | 413 | 506 |
| Interactions | 16,179 | 24,032 |
| Correct rate | 40.3% | 35.1% |
| Executable failures (Code v8 eligible) | 2,687 | 4,939 |

## Cross-semester

Train on F19 (`data/processed/…`) / test on S19 logs here, or vice versa.  
Do **not** merge student splits across semesters.
