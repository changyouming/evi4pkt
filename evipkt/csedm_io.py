from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Literal

from .kc_catalog import DEFAULT_KC_CATALOG, PROMPT_CSV_KC_COLUMNS

SubmissionMode = Literal["first", "all"]
DEFAULT_CSEDM_ROOT = "data/F19_Release_All_05_23_22/All"
DEFAULT_PROMPTS_CSV = "data/metadata/problem_prompts.csv"


def normalize_event_id(value) -> str:
    """Normalize ProgSnap2 EventID (F19 numeric vs S19 strings like '1-69176')."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def load_problem_prompts(path: Path) -> Dict[int, dict]:
    """Load per-problem task text and expert KC vector (q_t)."""
    if not path.exists():
        raise FileNotFoundError(path)

    out: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["ProblemID"])
            prompt = str(row.get("Requirement") or row.get("prompt") or "").strip()
            q_row = [0.0] * len(DEFAULT_KC_CATALOG)
            for csv_col, kc_name in PROMPT_CSV_KC_COLUMNS.items():
                if csv_col not in row:
                    continue
                val = row[csv_col]
                if val is None or str(val).strip() == "":
                    continue
                try:
                    active = float(val) > 0
                except ValueError:
                    active = str(val).strip().lower() in {"1", "true", "yes"}
                if active:
                    idx = DEFAULT_KC_CATALOG.index(kc_name)
                    q_row[idx] = 1.0
            out[pid] = {
                "prompt": prompt,
                "q_kc": q_row,
                "assignment_id": int(row["AssignmentID"]) if row.get("AssignmentID") else None,
            }
    return out


def load_code_states(code_states_path: Path, needed_ids: set[str]) -> Dict[str, str]:
    """Stream CodeStates.csv and return CodeStateID -> Code."""
    found: Dict[str, str] = {}
    if not needed_ids:
        return found
    with code_states_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row["CodeStateID"])
            if cid in needed_ids:
                found[cid] = str(row.get("Code") or "")
                if len(found) >= len(needed_ids):
                    break
    return found


def iter_run_program_events(main_table_path: Path):
    """Yield Run.Program rows from MainTable."""
    with main_table_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("EventType") == "Run.Program":
                yield row


def load_compile_index(main_table_path: Path) -> tuple[Dict[str, dict], Dict[str, List[dict]]]:
    """
    Index Compile events by parent Run.Program EventID, and Compile.Error by Compile EventID.
    """
    compile_by_run: Dict[str, dict] = {}
    cerr_by_compile: Dict[str, List[dict]] = {}
    with main_table_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            et = row.get("EventType")
            if et == "Compile":
                parent = normalize_event_id(row.get("ParentEventID"))
                if parent:
                    compile_by_run[parent] = row
            elif et == "Compile.Error":
                parent = normalize_event_id(row.get("ParentEventID"))
                if parent:
                    cerr_by_compile.setdefault(parent, []).append(row)
    return compile_by_run, cerr_by_compile


def resolve_compile_context(
    run_event_id: str,
    compile_by_run: Dict[str, dict],
    cerr_by_compile: Dict[str, List[dict]],
) -> tuple[str | None, List[str]]:
    """Return (compile_result, compile_error_messages) for a Run.Program event."""
    comp = compile_by_run.get(run_event_id)
    if comp is None:
        return None, []
    result = comp.get("Compile.Result") or None
    comp_id = normalize_event_id(comp["EventID"])
    errors = cerr_by_compile.get(comp_id, [])
    messages = [str(e.get("CompileMessageData") or "").strip() for e in errors]
    messages = [m for m in messages if m]
    return result, messages
