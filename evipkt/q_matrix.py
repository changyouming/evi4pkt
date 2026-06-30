from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from .csedm_io import DEFAULT_PROMPTS_CSV, load_problem_prompts
from .kc_catalog import DEFAULT_KC_CATALOG

# Task evidence q_t comes from bundled expert annotations — no LLM required.
TASK_Q_SOURCE = "dataset_expert"
DEFAULT_Q_MATRIX_PATH = "data/metadata/q_matrix.csv"


def build_q_matrix_from_prompts(prompts_csv: Path) -> Dict[int, List[float]]:
    """Build problem_id -> q_kc vector from problem_prompts.csv expert KC columns."""
    prompts = load_problem_prompts(prompts_csv)
    return {pid: list(info["q_kc"]) for pid, info in prompts.items()}


def write_q_matrix_csv(q_map: Dict[int, List[float]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id"] + DEFAULT_KC_CATALOG)
        for pid in sorted(q_map.keys()):
            row = q_map[pid]
            writer.writerow([pid] + [int(x) for x in row])
    return out_path


def export_dataset_q_matrix(
    prompts_csv: Path,
    out_path: Path | None = None,
) -> Path:
    """
    Export the dataset's built-in expert Q-matrix (not LLM-generated).
    Source: KC columns in data/metadata/problem_prompts.csv.
    """
    out_path = out_path or Path(DEFAULT_Q_MATRIX_PATH)
    q_map = build_q_matrix_from_prompts(prompts_csv)
    return write_q_matrix_csv(q_map, out_path)


def load_q_matrix(csv_path: Path) -> Dict[int, List[float]]:
    """Load q_matrix.csv produced by export_dataset_q_matrix / preprocess."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out: Dict[int, List[float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        k_cols = [c for c in (reader.fieldnames or []) if c != "problem_id"]
        for row in reader:
            pid = int(row["problem_id"])
            out[pid] = [float(row.get(k, 0) or 0) for k in k_cols]
    return out


def resolve_q_matrix(project_root: Path, q_matrix_path: Path | None = None) -> Dict[int, List[float]]:
    """
    Load dataset Q-matrix; export from prompts first if missing.
    """
    path = q_matrix_path or (project_root / DEFAULT_Q_MATRIX_PATH)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if not path.exists():
        prompts = project_root / DEFAULT_PROMPTS_CSV
        export_dataset_q_matrix(prompts, path)
    return load_q_matrix(path)
