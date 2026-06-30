# Framework Log Evidence Schema (rule-based channels)

One JSON object per line (`Run.Program` interaction). Fields used by the public reproduction bundle:

| Field | Role |
|-------|------|
| `subject_id` | Hashed student identifier from CSEDM |
| `problem_id` | CodeWorkout problem ID |
| `pkt_label` | Binary correctness (1 iff all unit tests pass) |
| `programming_task` | Task prompt + expert `q_kc` + `kc_catalog` |
| `student_code` | Submitted Java source |
| `code_issues` | Outcome typing (`compile_error`, `partial_pass`, …) |
| `process_evidence` | Pre-attempt KC exposure/success (rule-derived) |
| `error_evidence.mechanism_v8` | Compile-error mechanism tags M1–M12 (rule-derived) |

**Fair protocol:** at step *t*, prediction uses history through *t−1* plus Task/Process evidence available before the attempt; compile mechanism attaches to the current step for state updates at *t+1*.

**Not included in this bundle:** LLM-generated Code/Error alignment artifacts (`code_evidence.llm`, `misused_v8`, `error_evidence.llm`). The paper's Full Evi4PKT uses additional LLM channels that are not redistributed.
