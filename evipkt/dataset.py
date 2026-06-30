from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from .code_evidence import (
    CODE_EVIDENCE_VECTOR_DIM,
    code_evidence_backend_for_mode,
    code_evidence_vector,
)
from .error_evidence import ERROR_FEATURE_MODES, error_evidence_vector
from .feature_modes import normalize_feature_mode
from .plugplay_evidence import (
    MODES_USING_CODE as MODES_USING_PLUGPLAY_CODE,
    MODES_USING_MECHANISM as MODES_USING_PLUGPLAY_MECHANISM,
    code_evidence_vector as plugplay_code_vector,
    compile_mechanism_vector,
)

# Backward-compatible names for sequence_dataset and legacy scripts.
MODES_USING_CODE_V8 = MODES_USING_PLUGPLAY_CODE
MODES_USING_MECHANISM_V8 = MODES_USING_PLUGPLAY_MECHANISM
from .code2vec_features import CODE2VEC_VECTOR_DIM
from .codebert_features import CODEBERT_VECTOR_DIM
from .process_evidence import process_evidence_vector

FeatureMode = Literal[
    "problem_onehot",
    "problem_onehot_code2vec",
    "problem_onehot_codebert",
    "problem_plus_q",
    "problem_plus_q_code2vec",
    "q_only",
    "problem_plus_code",
    "problem_plus_q_code_rule",
    "problem_plus_code_llm",
    "problem_plus_q_code_llm",
    "problem_plus_q_process",
    "problem_plus_q_code_llm_process",
    "problem_plus_q_error",
    "problem_plus_q_code_llm_error",
    "problem_plus_q_process_error",
    "problem_plus_q_code_llm_process_error",
    "problem_plus_q_code",
    "problem_plus_q_mechanism",
    "problem_plus_q_code_mechanism",
    "problem_plus_q_process_code",
    "problem_plus_q_process_code_mechanism",
    "problem_plus_q_process_code_mechanism_code2vec",
]

FEATURE_MODES: tuple[str, ...] = (
    "problem_onehot",
    "problem_onehot_code2vec",
    "problem_onehot_codebert",
    "problem_plus_q",
    "problem_plus_q_code2vec",
    "q_only",
    "problem_plus_code",
    "problem_plus_q_code_rule",
    "problem_plus_code_llm",
    "problem_plus_q_code_llm",
    "problem_plus_q_process",
    "problem_plus_q_code_llm_process",
    "problem_plus_q_error",
    "problem_plus_q_code_llm_error",
    "problem_plus_q_process_error",
    "problem_plus_q_code_llm_process_error",
    "problem_plus_q_code",
    "problem_plus_q_mechanism",
    "problem_plus_q_code_mechanism",
    "problem_plus_q_process_code",
    "problem_plus_q_process_code_mechanism",
    "problem_plus_q_process_code_mechanism_code2vec",
)

MODES_USING_Q = frozenset(
    {
        "problem_plus_q",
        "problem_plus_q_code2vec",
        "q_only",
        "problem_plus_q_code_rule",
        "problem_plus_q_code_llm",
        "problem_plus_q_process",
        "problem_plus_q_code_llm_process",
        "problem_plus_q_error",
        "problem_plus_q_code_llm_error",
        "problem_plus_q_process_error",
        "problem_plus_q_code_llm_process_error",
        "problem_plus_q_code",
        "problem_plus_q_mechanism",
        "problem_plus_q_code_mechanism",
        "problem_plus_q_process_code",
        "problem_plus_q_process_code_mechanism",
        "problem_plus_q_process_code_mechanism_code2vec",
    }
)
MODES_USING_CODE = frozenset(
    {
        "problem_plus_code",
        "problem_plus_q_code_rule",
        "problem_plus_code_llm",
        "problem_plus_q_code_llm",
        "problem_plus_q_code_llm_process",
        "problem_plus_q_code_llm_error",
        "problem_plus_q_code_llm_process_error",
    }
)
MODES_USING_PROCESS = frozenset(
    {
        "problem_plus_q_process",
        "problem_plus_q_code_llm_process",
        "problem_plus_q_process_error",
        "problem_plus_q_code_llm_process_error",
        "problem_plus_q_process_code",
        "problem_plus_q_process_code_mechanism",
        "problem_plus_q_process_code_mechanism_code2vec",
    }
)
MODES_USING_ERROR = ERROR_FEATURE_MODES
MODES_USING_CODE2VEC = frozenset(
    {
        "problem_onehot_code2vec",
        "problem_plus_q_code2vec",
        "problem_plus_q_process_code_mechanism_code2vec",
    }
)
MODES_USING_CODEBERT = frozenset(
    {
        "problem_onehot_codebert",
    }
)

import torch
from torch.utils.data import Dataset

DKTSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class SplitData:
    train_students: List[str]
    valid_students: List[str]
    test_students: List[str]


def split_students(students: List[str], seed: int = 0) -> SplitData:
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(students), generator=rng).tolist()
    shuffled = [students[i] for i in idx]
    n = len(students)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    return SplitData(
        train_students=shuffled[:n_train],
        valid_students=shuffled[n_train : n_train + n_valid],
        test_students=shuffled[n_train + n_valid :],
    )


def split_students_stratified(students_by_group: Dict[str, List[str]], seed: int = 0) -> SplitData:
    """80/10/10 within each group, then concatenate (e.g. per-semester cohort)."""
    train: List[str] = []
    valid: List[str] = []
    test: List[str] = []
    for _group, group_students in sorted(students_by_group.items()):
        part = split_students(sorted(group_students), seed=seed)
        train.extend(part.train_students)
        valid.extend(part.valid_students)
        test.extend(part.test_students)
    return SplitData(train_students=train, valid_students=valid, test_students=test)


DEFAULT_SPLIT_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


def infer_cohort_from_logs_path(logs_path: str | Path) -> str | None:
    p = str(logs_path).replace("\\", "/").lower()
    if "processed_s19" in p or "/s19/" in p:
        return "s19"
    if "processed" in p or "/f19/" in p:
        return "f19"
    return None


def split_file_path(cohort: str, seed: int, split_dir: Path | None = None) -> Path:
    root = split_dir or DEFAULT_SPLIT_DIR
    return root / cohort / f"seed_{seed}.json"


def split_to_dict(
    split: SplitData,
    *,
    cohort: str,
    seed: int,
    source_logs: str = "",
) -> dict:
    n = len(split.train_students) + len(split.valid_students) + len(split.test_students)
    return {
        "cohort": cohort,
        "seed": seed,
        "protocol": "student 80/10/10",
        "n_students": n,
        "source_logs": source_logs,
        "train_students": split.train_students,
        "valid_students": split.valid_students,
        "test_students": split.test_students,
    }


def save_student_split(
    split: SplitData,
    *,
    cohort: str,
    seed: int,
    split_dir: Path | None = None,
    source_logs: str = "",
) -> Path:
    path = split_file_path(cohort, seed, split_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = split_to_dict(split, cohort=cohort, seed=seed, source_logs=source_logs)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_student_split(cohort: str, seed: int, split_dir: Path | None = None) -> SplitData:
    path = split_file_path(cohort, seed, split_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitData(
        train_students=[str(s) for s in payload["train_students"]],
        valid_students=[str(s) for s in payload["valid_students"]],
        test_students=[str(s) for s in payload["test_students"]],
    )


def resolve_student_split(
    students: List[str],
    *,
    seed: int,
    logs_path: str | Path | None = None,
    cohort: str | None = None,
    split_dir: Path | None = None,
) -> SplitData:
    """Use bundled split JSON when available; otherwise fall back to split_students."""
    resolved_cohort = cohort or (
        infer_cohort_from_logs_path(logs_path) if logs_path is not None else None
    )
    if resolved_cohort:
        path = split_file_path(resolved_cohort, seed, split_dir)
        if path.is_file():
            split = load_student_split(resolved_cohort, seed, split_dir)
            expected = set(students)
            actual = set(split.train_students) | set(split.valid_students) | set(split.test_students)
            if expected != actual:
                raise ValueError(
                    f"Split file {path} does not match framework logs: "
                    f"logs have {len(expected)} students, split file covers {len(actual)}."
                )
            return split
    return split_students(students, seed=seed)


def load_framework_logs(jsonl_path: Path) -> List[dict]:
    records: List[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def _code_vec_tensor(record: dict, feature_mode: str) -> torch.Tensor:
    backend = code_evidence_backend_for_mode(feature_mode)
    return torch.tensor(code_evidence_vector(record, backend=backend), dtype=torch.float32)


def _plugplay_code_vec_tensor(record: dict) -> torch.Tensor:
    return torch.tensor(plugplay_code_vector(record), dtype=torch.float32)


def _plugplay_mechanism_vec_tensor(record: dict) -> torch.Tensor:
    return torch.tensor(compile_mechanism_vector(record), dtype=torch.float32)


def _student_code_text(record: dict) -> str:
    sc = record.get("student_code") or {}
    return str(sc.get("code") or "")


def _code2vec_vec_tensor(record: dict, code2vec_cache: dict[str, list[float]] | None) -> torch.Tensor:
    if not code2vec_cache:
        raise ValueError("code2vec feature modes require code2vec_cache.")
    from .code2vec_features import code_cache_key

    key = code_cache_key(_student_code_text(record))
    vec = code2vec_cache.get(key)
    if vec is None:
        vec = [0.0] * CODE2VEC_VECTOR_DIM
    return torch.tensor(vec, dtype=torch.float32)


def _codebert_vec_tensor(record: dict, codebert_cache: dict[str, list[float]] | None) -> torch.Tensor:
    if not codebert_cache:
        raise ValueError("codebert feature modes require codebert_cache.")
    from .codebert_features import code_cache_key

    key = code_cache_key(_student_code_text(record))
    vec = codebert_cache.get(key)
    if vec is None:
        vec = [0.0] * CODEBERT_VECTOR_DIM
    return torch.tensor(vec, dtype=torch.float32)


_code_v8_vec_tensor = _plugplay_code_vec_tensor
_mechanism_v8_vec_tensor = _plugplay_mechanism_vec_tensor


def _interaction_vector(
    problem_id: int,
    pkt_label: int,
    problem_id_to_index: Dict[int, int],
    num_problems: int,
    q_vec: torch.Tensor | None = None,
    code_vec: torch.Tensor | None = None,
    plugplay_code_vec: torch.Tensor | None = None,
    plugplay_mechanism_vec: torch.Tensor | None = None,
    process_vec: torch.Tensor | None = None,
    error_vec: torch.Tensor | None = None,
    code2vec_vec: torch.Tensor | None = None,
    codebert_vec: torch.Tensor | None = None,
    feature_mode: FeatureMode = "problem_onehot",
) -> torch.Tensor:
    y = int(pkt_label)
    if feature_mode == "q_only":
        if q_vec is None:
            raise ValueError("q_only feature_mode requires q_vec.")
        return torch.cat(
            [q_vec.to(dtype=torch.float32), torch.tensor([float(y)], dtype=torch.float32)],
            dim=0,
        )

    feat = torch.zeros(2 * num_problems, dtype=torch.float32)
    pidx = problem_id_to_index[int(problem_id)]
    offset = num_problems if y == 1 else 0
    feat[pidx + offset] = 1.0
    if feature_mode in MODES_USING_Q and feature_mode != "q_only":
        if q_vec is None:
            raise ValueError(f"{feature_mode} requires q_vec.")
        feat = torch.cat([feat, q_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_CODE:
        if code_vec is None:
            raise ValueError(f"{feature_mode} requires code_vec.")
        feat = torch.cat([feat, code_vec.to(dtype=torch.float32)], dim=0)
    mode = normalize_feature_mode(feature_mode)
    if mode in MODES_USING_PLUGPLAY_CODE:
        if plugplay_code_vec is None:
            raise ValueError(f"{feature_mode} requires plugplay code evidence.")
        feat = torch.cat([feat, plugplay_code_vec.to(dtype=torch.float32)], dim=0)
    if mode in MODES_USING_PLUGPLAY_MECHANISM:
        if plugplay_mechanism_vec is None:
            raise ValueError(f"{feature_mode} requires plugplay mechanism evidence.")
        feat = torch.cat([feat, plugplay_mechanism_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_PROCESS:
        if process_vec is None:
            raise ValueError(f"{feature_mode} requires process_vec.")
        feat = torch.cat([feat, process_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_ERROR:
        if error_vec is None:
            raise ValueError(f"{feature_mode} requires error_vec.")
        feat = torch.cat([feat, error_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_CODE2VEC:
        if code2vec_vec is None:
            raise ValueError(f"{feature_mode} requires code2vec_vec.")
        feat = torch.cat([feat, code2vec_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_CODEBERT:
        if codebert_vec is None:
            raise ValueError(f"{feature_mode} requires codebert_vec.")
        feat = torch.cat([feat, codebert_vec.to(dtype=torch.float32)], dim=0)
    return feat


def build_interaction_from_record(
    row: dict,
    *,
    problem_id_to_index: Dict[int, int],
    num_problems: int,
    problem_q_map: Dict[int, List[float]] | None,
    feature_mode: FeatureMode,
    code2vec_cache: dict[str, list[float]] | None = None,
    codebert_cache: dict[str, list[float]] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, int] | None:
    """Build (interaction_vec, target_vec, pkt_label) for one trajectory row."""
    pid = int(row["problem_id"])
    if pid not in problem_id_to_index:
        return None
    y = int(row["pkt_label"] if "pkt_label" in row else row["code_issues"]["pkt_label"])
    q_vec = None
    if feature_mode in MODES_USING_Q:
        if problem_q_map is None:
            return None
        q_row = problem_q_map.get(pid)
        if q_row is None:
            return None
        q_vec = torch.tensor(q_row, dtype=torch.float32)
    code_vec = (
        _code_vec_tensor(row, feature_mode) if feature_mode in MODES_USING_CODE else None
    )
    plugplay_code_vec = (
        _plugplay_code_vec_tensor(row)
        if feature_mode in MODES_USING_PLUGPLAY_CODE
        else None
    )
    plugplay_mechanism_vec = (
        _plugplay_mechanism_vec_tensor(row)
        if feature_mode in MODES_USING_PLUGPLAY_MECHANISM
        else None
    )
    process_vec = (
        torch.tensor(process_evidence_vector(row), dtype=torch.float32)
        if feature_mode in MODES_USING_PROCESS
        else None
    )
    error_vec = (
        torch.tensor(error_evidence_vector(row), dtype=torch.float32)
        if feature_mode in MODES_USING_ERROR
        else None
    )
    code2vec_vec = (
        _code2vec_vec_tensor(row, code2vec_cache)
        if feature_mode in MODES_USING_CODE2VEC
        else None
    )
    codebert_vec = (
        _codebert_vec_tensor(row, codebert_cache)
        if feature_mode in MODES_USING_CODEBERT
        else None
    )
    interaction = _interaction_vector(
        pid,
        y,
        problem_id_to_index,
        num_problems,
        q_vec,
        code_vec,
        plugplay_code_vec,
        plugplay_mechanism_vec,
        process_vec,
        error_vec,
        code2vec_vec,
        codebert_vec,
        feature_mode=feature_mode,
    )
    target = _target_vector(
        pid, problem_id_to_index, num_problems, q_vec, feature_mode=feature_mode
    )
    return interaction, target, y


def _target_vector(
    problem_id: int,
    problem_id_to_index: Dict[int, int],
    num_problems: int,
    q_vec: torch.Tensor | None = None,
    feature_mode: FeatureMode = "problem_onehot",
) -> torch.Tensor:
    if feature_mode == "q_only":
        if q_vec is None:
            raise ValueError("q_only feature_mode requires q_vec.")
        return q_vec.to(dtype=torch.float32)

    feat = torch.zeros(num_problems, dtype=torch.float32)
    feat[problem_id_to_index[int(problem_id)]] = 1.0
    if feature_mode in MODES_USING_Q and feature_mode != "q_only":
        if q_vec is None:
            raise ValueError(f"{feature_mode} requires q_vec.")
        feat = torch.cat([feat, q_vec.to(dtype=torch.float32)], dim=0)
    return feat


def build_dkt_samples(
    records: List[dict],
    students: List[str],
    problem_id_to_index: Dict[int, int],
    problem_q_map: Dict[int, List[float]] | None = None,
    feature_mode: FeatureMode = "problem_onehot",
    code2vec_cache: dict[str, list[float]] | None = None,
    codebert_cache: dict[str, list[float]] | None = None,
) -> List[DKTSample]:
    """
    Next-step DKT samples: history [0:t] + target problem at t -> predict pkt_label at step t.
    """
    if feature_mode in MODES_USING_Q and problem_q_map is None:
        raise ValueError(f"feature_mode='{feature_mode}' requires problem_q_map.")
    feature_mode = normalize_feature_mode(feature_mode)  # type: ignore[assignment]
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode='{feature_mode}'.")
    if feature_mode in MODES_USING_CODE2VEC and not code2vec_cache:
        raise ValueError(f"feature_mode='{feature_mode}' requires code2vec_cache.")
    if feature_mode in MODES_USING_CODEBERT and not codebert_cache:
        raise ValueError(f"feature_mode='{feature_mode}' requires codebert_cache.")

    num_problems = len(problem_id_to_index)
    student_set = set(students)
    by_student: Dict[str, List[dict]] = {}
    for rec in records:
        sid = str(rec["subject_id"])
        if sid not in student_set:
            continue
        by_student.setdefault(sid, []).append(rec)

    samples: List[DKTSample] = []
    for rows in by_student.values():
        rows.sort(key=lambda r: int(r["trajectory"]["student_timestep"]))
        feats: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        labels: List[int] = []
        for row in rows:
            pid = int(row["problem_id"])
            if pid not in problem_id_to_index:
                continue
            y = int(row["pkt_label"] if "pkt_label" in row else row["code_issues"]["pkt_label"])
            q_vec = None
            if feature_mode in MODES_USING_Q:
                q_row = problem_q_map.get(pid)
                if q_row is None:
                    continue
                q_vec = torch.tensor(q_row, dtype=torch.float32)
            code_vec = (
                _code_vec_tensor(row, feature_mode) if feature_mode in MODES_USING_CODE else None
            )
            plugplay_code_vec = (
                _plugplay_code_vec_tensor(row)
                if feature_mode in MODES_USING_PLUGPLAY_CODE
                else None
            )
            plugplay_mechanism_vec = (
                _plugplay_mechanism_vec_tensor(row)
                if feature_mode in MODES_USING_PLUGPLAY_MECHANISM
                else None
            )
            process_vec = (
                torch.tensor(process_evidence_vector(row), dtype=torch.float32)
                if feature_mode in MODES_USING_PROCESS
                else None
            )
            error_vec = (
                torch.tensor(error_evidence_vector(row), dtype=torch.float32)
                if feature_mode in MODES_USING_ERROR
                else None
            )
            code2vec_vec = (
                _code2vec_vec_tensor(row, code2vec_cache)
                if feature_mode in MODES_USING_CODE2VEC
                else None
            )
            codebert_vec = (
                _codebert_vec_tensor(row, codebert_cache)
                if feature_mode in MODES_USING_CODEBERT
                else None
            )
            feats.append(
                _interaction_vector(
                    pid,
                    y,
                    problem_id_to_index,
                    num_problems,
                    q_vec,
                    code_vec,
                    plugplay_code_vec,
                    plugplay_mechanism_vec,
                    process_vec,
                    error_vec,
                    code2vec_vec,
                    codebert_vec,
                    feature_mode=feature_mode,
                )
            )
            targets.append(
                _target_vector(
                    pid, problem_id_to_index, num_problems, q_vec, feature_mode=feature_mode
                )
            )
            labels.append(y)
        if len(feats) < 2:
            continue
        for t in range(1, len(feats)):
            hist = torch.stack(feats[:t], dim=0)
            samples.append(
                (
                    hist,
                    torch.tensor(t, dtype=torch.long),
                    targets[t],
                    torch.tensor(float(labels[t])),
                )
            )
    return samples


class DKTPrefixDataset(Dataset):
    def __init__(self, samples: List[DKTSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_dkt_batch(batch):
    seqs, lengths, targets, labels = zip(*batch)
    max_len = max(int(x.item()) for x in lengths)
    feat_dim = seqs[0].shape[-1]
    padded = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s
    return (
        padded,
        torch.stack(lengths).long(),
        torch.stack(targets).float(),
        torch.stack(labels).float(),
    )
