from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

from .csedm_io import (
    DEFAULT_CSEDM_ROOT,
    DEFAULT_PROMPTS_CSV,
    SubmissionMode,
    iter_run_program_events,
    load_code_states,
    load_compile_index,
    load_problem_prompts,
    normalize_event_id,
    resolve_compile_context,
)
from .code_evidence import CODE_EVIDENCE_SOURCE, build_code_evidence
from .kc_catalog import DEFAULT_KC_CATALOG
from .labels import build_code_issues
from .q_matrix import build_q_matrix_from_prompts, export_dataset_q_matrix, write_q_matrix_csv


@dataclass
class PreprocessConfig:
    csedm_root: Path
    prompts_csv: Path
    submission_mode: SubmissionMode = "all"
    label_threshold: float = 1.0
    include_compile_only: bool = False  # reserved; Run.Program anchors each interaction


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_interaction_record(
    row: dict,
    *,
    code: str,
    task: dict,
    compile_by_run: Dict[str, dict],
    cerr_by_compile: Dict[str, List[dict]],
    label_threshold: float,
    problem_attempt: int,
    student_timestep: int,
    prev_same_problem: dict | None,
) -> dict:
    run_event_id = normalize_event_id(row["EventID"])
    score = _safe_float(row.get("Score"), 0.0)
    compile_result, compile_messages = resolve_compile_context(
        run_event_id, compile_by_run, cerr_by_compile
    )
    has_compile_error = compile_result == "Error" or bool(compile_messages)

    code_issues = build_code_issues(
        score,
        has_compile_error=has_compile_error,
        compile_messages=compile_messages,
        threshold=label_threshold,
    )

    pid = int(row["ProblemID"])
    programming_task = {
        "problem_id": pid,
        "assignment_id": int(float(row["AssignmentID"])) if row.get("AssignmentID") else task.get("assignment_id"),
        "prompt": task.get("prompt", ""),
        "q_kc": task.get("q_kc", [0.0] * len(DEFAULT_KC_CATALOG)),
        "kc_catalog": DEFAULT_KC_CATALOG,
    }

    student_code = {
        "code_state_id": str(row.get("CodeStateID") or ""),
        "code": code,
    }
    code_evidence = build_code_evidence(code, task.get("q_kc"))

    trajectory = {
        "student_timestep": student_timestep,
        "problem_attempt": problem_attempt,
        "server_timestamp": str(row.get("ServerTimestamp") or ""),
        "order": int(float(row["Order"])) if row.get("Order") else None,
        "run_event_id": run_event_id,
        "prev_same_problem_run_event_id": (
            prev_same_problem["trajectory"]["run_event_id"] if prev_same_problem else None
        ),
        "prev_same_problem_pkt_label": (
            prev_same_problem["code_issues"]["pkt_label"] if prev_same_problem else None
        ),
    }

    return {
        "subject_id": str(row["SubjectID"]),
        "problem_id": pid,
        "programming_task": programming_task,
        "student_code": student_code,
        "code_evidence": code_evidence,
        "code_issues": code_issues,
        "trajectory": trajectory,
        # Flat fields for PKT pipelines
        "pkt_label": code_issues["pkt_label"],
        "score": code_issues["score"],
    }


def iter_framework_logs(cfg: PreprocessConfig) -> Iterator[dict]:
    """
    Yield framework-aligned learning log records (one per Run.Program interaction).

    Maps framework.png inputs:
      1) programming_task  2) student_code  3) code_evidence  4) code_issues (+ pkt_label)  5) trajectory
    """
    root = cfg.csedm_root
    main_path = root / "Data" / "MainTable.csv"
    code_path = root / "Data" / "CodeStates" / "CodeStates.csv"
    if not main_path.exists():
        raise FileNotFoundError(main_path)

    prompts = load_problem_prompts(cfg.prompts_csv)
    compile_by_run, cerr_by_compile = load_compile_index(main_path)

    # Pass 1: collect Run.Program rows (streaming grouped by student would need sort;
    # load into memory grouped — 125k rows is fine).
    by_student: Dict[str, List[dict]] = {}
    needed_code_ids: set[str] = set()
    for row in iter_run_program_events(main_path):
        sid = str(row["SubjectID"])
        by_student.setdefault(sid, []).append(row)
        cid = str(row.get("CodeStateID") or "")
        if cid:
            needed_code_ids.add(cid)

    codes = load_code_states(code_path, needed_code_ids)

    for sid, rows in by_student.items():
        rows.sort(key=lambda r: (int(float(r["Order"])), str(r.get("ServerTimestamp") or "")))

        if cfg.submission_mode == "first":
            seen_prob: set[int] = set()
            filtered = []
            for r in rows:
                pid = int(r["ProblemID"])
                if pid not in seen_prob:
                    seen_prob.add(pid)
                    filtered.append(r)
            rows = filtered

        problem_attempt_counter: Dict[int, int] = {}
        prev_by_problem: Dict[int, dict] = {}

        for t, row in enumerate(rows):
            pid = int(row["ProblemID"])
            problem_attempt_counter[pid] = problem_attempt_counter.get(pid, 0) + 1
            task = prompts.get(pid, {"prompt": "", "q_kc": [0.0] * len(DEFAULT_KC_CATALOG)})
            code = codes.get(str(row.get("CodeStateID") or ""), "")

            prev_same = prev_by_problem.get(pid)
            rec = _build_interaction_record(
                row,
                code=code,
                task=task,
                compile_by_run=compile_by_run,
                cerr_by_compile=cerr_by_compile,
                label_threshold=cfg.label_threshold,
                problem_attempt=problem_attempt_counter[pid],
                student_timestep=t,
                prev_same_problem=prev_same,
            )
            prev_by_problem[pid] = rec
            yield rec


def run_preprocess(cfg: PreprocessConfig, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"framework_logs_{cfg.submission_mode}.jsonl"
    q_matrix_path = out_dir / "q_matrix.csv"

    records: List[dict] = []
    label_counts = {0: 0, 1: 0}
    outcome_counts: Dict[str, int] = {}
    missing_prompts: set[int] = set()

    with jsonl_path.open("w", encoding="utf-8") as out_f:
        for rec in iter_framework_logs(cfg):
            records.append(rec)
            label_counts[rec["pkt_label"]] = label_counts.get(rec["pkt_label"], 0) + 1
            ot = rec["code_issues"]["outcome_type"]
            outcome_counts[ot] = outcome_counts.get(ot, 0) + 1
            if not rec["programming_task"]["prompt"]:
                missing_prompts.add(rec["problem_id"])
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Dataset expert Q-matrix (problem_prompts.csv KC columns; no LLM).
    metadata_q_path = cfg.prompts_csv.parent / "q_matrix.csv"
    q_map = build_q_matrix_from_prompts(cfg.prompts_csv)
    export_dataset_q_matrix(cfg.prompts_csv, metadata_q_path)
    write_q_matrix_csv(q_map, q_matrix_path)

    missing_code = sum(1 for r in records if not (r["student_code"].get("code") or "").strip())

    summary = {
        "submission_mode": cfg.submission_mode,
        "label_threshold": cfg.label_threshold,
        "label_rule": f"score_ge_{cfg.label_threshold}",
        "total_interactions": len(records),
        "students": len({r["subject_id"] for r in records}),
        "problems": len({r["problem_id"] for r in records}),
        "missing_code_state_joins": missing_code,
        "pkt_label_distribution": {
            "incorrect_0": label_counts.get(0, 0),
            "correct_1": label_counts.get(1, 0),
            "correct_rate": round(label_counts.get(1, 0) / max(1, len(records)), 4),
        },
        "outcome_type_distribution": outcome_counts,
        "missing_prompts": sorted(missing_prompts),
        "task_q_source": "dataset_expert",
        "code_evidence_source": CODE_EVIDENCE_SOURCE,
        "outputs": {
            "framework_logs": str(jsonl_path),
            "q_matrix_processed": str(q_matrix_path),
            "q_matrix_dataset": str(metadata_q_path),
        },
    }
    summary_path = out_dir / f"preprocess_summary_{cfg.submission_mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def default_preprocess_config(
    project_root: Path,
    *,
    submission_mode: SubmissionMode = "all",
    label_threshold: float = 1.0,
) -> PreprocessConfig:
    return PreprocessConfig(
        csedm_root=project_root / DEFAULT_CSEDM_ROOT,
        prompts_csv=project_root / DEFAULT_PROMPTS_CSV,
        submission_mode=submission_mode,
        label_threshold=label_threshold,
    )
