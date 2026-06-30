from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .dataset import (
    FEATURE_MODES,
    FeatureMode,
    MODES_USING_CODE,
    MODES_USING_CODE_V8,
    MODES_USING_ERROR,
    MODES_USING_MECHANISM_V8,
    MODES_USING_PROCESS,
    MODES_USING_Q,
    _code_vec_tensor,
    _code_v8_vec_tensor,
    _mechanism_v8_vec_tensor,
    _target_vector,
)
from .feature_modes import normalize_feature_mode
from .error_evidence import error_evidence_vector
from .process_evidence import process_evidence_vector

SequenceSample = tuple[torch.Tensor, torch.Tensor]
DKVMNSample = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _query_vector(
    problem_id: int,
    problem_id_to_index: dict[int, int],
    num_problems: int,
    *,
    q_vec: torch.Tensor | None = None,
    code_vec: torch.Tensor | None = None,
    code_v8_vec: torch.Tensor | None = None,
    mechanism_v8_vec: torch.Tensor | None = None,
    process_vec: torch.Tensor | None = None,
    error_vec: torch.Tensor | None = None,
    feature_mode: FeatureMode = "problem_onehot",
) -> torch.Tensor:
    feat = torch.zeros(num_problems, dtype=torch.float32)
    feat[problem_id_to_index[int(problem_id)]] = 1.0
    if feature_mode in MODES_USING_Q and feature_mode != "q_only":
        if q_vec is None:
            raise ValueError(f"{feature_mode} requires q_vec.")
        feat = torch.cat([feat, q_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_CODE:
        if code_vec is None:
            raise ValueError(f"{feature_mode} requires code_vec.")
        feat = torch.cat([feat, code_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_CODE_V8:
        if code_v8_vec is None:
            raise ValueError(f"{feature_mode} requires code_v8_vec.")
        feat = torch.cat([feat, code_v8_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_MECHANISM_V8:
        if mechanism_v8_vec is None:
            raise ValueError(f"{feature_mode} requires mechanism_v8_vec.")
        feat = torch.cat([feat, mechanism_v8_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_PROCESS:
        if process_vec is None:
            raise ValueError(f"{feature_mode} requires process_vec.")
        feat = torch.cat([feat, process_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_ERROR:
        if error_vec is None:
            raise ValueError(f"{feature_mode} requires error_vec.")
        feat = torch.cat([feat, error_vec.to(dtype=torch.float32)], dim=0)
    return feat


def _write_vector(
    problem_id: int,
    problem_id_to_index: dict[int, int],
    num_problems: int,
    *,
    q_vec: torch.Tensor | None = None,
    code_vec: torch.Tensor | None = None,
    code_v8_vec: torch.Tensor | None = None,
    mechanism_v8_vec: torch.Tensor | None = None,
    process_vec: torch.Tensor | None = None,
    error_vec: torch.Tensor | None = None,
    feature_mode: FeatureMode = "problem_onehot",
) -> torch.Tensor:
    """Memory-write features after observing the outcome (no correctness bit)."""
    feat = torch.zeros(num_problems, dtype=torch.float32)
    feat[problem_id_to_index[int(problem_id)]] = 1.0
    if feature_mode in MODES_USING_Q and feature_mode != "q_only":
        if q_vec is None:
            raise ValueError(f"{feature_mode} requires q_vec.")
        feat = torch.cat([feat, q_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_CODE:
        if code_vec is None:
            raise ValueError(f"{feature_mode} requires code_vec.")
        feat = torch.cat([feat, code_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_CODE_V8:
        if code_v8_vec is None:
            raise ValueError(f"{feature_mode} requires code_v8_vec.")
        feat = torch.cat([feat, code_v8_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_MECHANISM_V8:
        if mechanism_v8_vec is None:
            raise ValueError(f"{feature_mode} requires mechanism_v8_vec.")
        feat = torch.cat([feat, mechanism_v8_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_PROCESS:
        if process_vec is None:
            raise ValueError(f"{feature_mode} requires process_vec.")
        feat = torch.cat([feat, process_vec.to(dtype=torch.float32)], dim=0)
    if feature_mode in MODES_USING_ERROR:
        if error_vec is None:
            raise ValueError(f"{feature_mode} requires error_vec.")
        feat = torch.cat([feat, error_vec.to(dtype=torch.float32)], dim=0)
    return feat


def build_dkvmn_samples(
    records: list[dict],
    students: list[str],
    problem_id_to_index: dict[int, int],
    problem_q_map: dict[int, list[float]] | None = None,
    feature_mode: FeatureMode = "problem_onehot",
) -> list[DKVMNSample]:
    """
    Leakage-free DKVMN sequences.

    predict_queries[t]: target problem (+ task Q) only — used before observing y_t.
    write_vectors[t]: problem (+ Q + evidence) without correctness — used after y_t.
    """
    if feature_mode in MODES_USING_Q and problem_q_map is None:
        raise ValueError(f"{feature_mode} requires problem_q_map.")
    feature_mode = normalize_feature_mode(feature_mode)  # type: ignore[assignment]
    if feature_mode not in FEATURE_MODES or feature_mode == "q_only":
        raise ValueError(f"Unsupported DKVMN feature_mode='{feature_mode}'.")

    num_problems = len(problem_id_to_index)
    student_set = set(students)
    by_student: dict[str, list[dict]] = {}
    for rec in records:
        sid = str(rec["subject_id"])
        if sid in student_set:
            by_student.setdefault(sid, []).append(rec)

    samples: list[DKVMNSample] = []
    for rows in by_student.values():
        rows.sort(key=lambda r: int(r["trajectory"]["student_timestep"]))
        predict_queries: list[torch.Tensor] = []
        write_vectors: list[torch.Tensor] = []
        labels: list[float] = []
        for row in rows:
            pid = int(row["problem_id"])
            if pid not in problem_id_to_index:
                continue
            q_vec = None
            if feature_mode in MODES_USING_Q:
                q_row = problem_q_map.get(pid) if problem_q_map is not None else None
                if q_row is None:
                    continue
                q_vec = torch.tensor(q_row, dtype=torch.float32)
            code_vec = (
                _code_vec_tensor(row, feature_mode) if feature_mode in MODES_USING_CODE else None
            )
            code_v8_vec = (
                _code_v8_vec_tensor(row) if feature_mode in MODES_USING_CODE_V8 else None
            )
            mechanism_v8_vec = (
                _mechanism_v8_vec_tensor(row)
                if feature_mode in MODES_USING_MECHANISM_V8
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
            y = float(row["pkt_label"] if "pkt_label" in row else row["code_issues"]["pkt_label"])
            predict_queries.append(
                _target_vector(
                    pid,
                    problem_id_to_index,
                    num_problems,
                    q_vec,
                    feature_mode=feature_mode,
                )
            )
            write_vectors.append(
                _write_vector(
                    pid,
                    problem_id_to_index,
                    num_problems,
                    q_vec=q_vec,
                    code_vec=code_vec,
                    code_v8_vec=code_v8_vec,
                    mechanism_v8_vec=mechanism_v8_vec,
                    process_vec=process_vec,
                    error_vec=error_vec,
                    feature_mode=feature_mode,
                )
            )
            labels.append(y)
        if len(predict_queries) >= 1:
            samples.append(
                (
                    torch.stack(predict_queries, dim=0),
                    torch.stack(write_vectors, dim=0),
                    torch.tensor(labels, dtype=torch.float32),
                )
            )
    return samples


def build_sequence_samples(
    records: list[dict],
    students: list[str],
    problem_id_to_index: dict[int, int],
    problem_q_map: dict[int, list[float]] | None = None,
    feature_mode: FeatureMode = "problem_onehot",
) -> list[SequenceSample]:
    if feature_mode in MODES_USING_Q and problem_q_map is None:
        raise ValueError(f"{feature_mode} requires problem_q_map.")
    feature_mode = normalize_feature_mode(feature_mode)  # type: ignore[assignment]
    if feature_mode not in FEATURE_MODES or feature_mode == "q_only":
        raise ValueError(f"Unsupported sequence feature_mode='{feature_mode}'.")

    num_problems = len(problem_id_to_index)
    student_set = set(students)
    by_student: dict[str, list[dict]] = {}
    for rec in records:
        sid = str(rec["subject_id"])
        if sid in student_set:
            by_student.setdefault(sid, []).append(rec)

    samples: list[SequenceSample] = []
    for rows in by_student.values():
        rows.sort(key=lambda r: int(r["trajectory"]["student_timestep"]))
        queries: list[torch.Tensor] = []
        labels: list[float] = []
        for row in rows:
            pid = int(row["problem_id"])
            if pid not in problem_id_to_index:
                continue
            q_vec = None
            if feature_mode in MODES_USING_Q:
                q_row = problem_q_map.get(pid) if problem_q_map is not None else None
                if q_row is None:
                    continue
                q_vec = torch.tensor(q_row, dtype=torch.float32)
            code_vec = (
                _code_vec_tensor(row, feature_mode) if feature_mode in MODES_USING_CODE else None
            )
            code_v8_vec = (
                _code_v8_vec_tensor(row) if feature_mode in MODES_USING_CODE_V8 else None
            )
            mechanism_v8_vec = (
                _mechanism_v8_vec_tensor(row)
                if feature_mode in MODES_USING_MECHANISM_V8
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
            y = float(row["pkt_label"] if "pkt_label" in row else row["code_issues"]["pkt_label"])
            queries.append(
                _query_vector(
                    pid,
                    problem_id_to_index,
                    num_problems,
                    q_vec=q_vec,
                    code_vec=code_vec,
                    code_v8_vec=code_v8_vec,
                    mechanism_v8_vec=mechanism_v8_vec,
                    process_vec=process_vec,
                    error_vec=error_vec,
                    feature_mode=feature_mode,
                )
            )
            labels.append(y)
        if len(queries) >= 1:
            samples.append((torch.stack(queries, dim=0), torch.tensor(labels, dtype=torch.float32)))
    return samples


class KTSequenceDataset(Dataset):
    def __init__(self, samples: list[SequenceSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_sequence_batch(batch: list[SequenceSample]):
    q_seqs, y_seqs = zip(*batch)
    max_len = max(s.shape[0] for s in q_seqs)
    q_dim = q_seqs[0].shape[-1]
    queries = torch.zeros(len(batch), max_len, q_dim, dtype=torch.float32)
    labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (q, y) in enumerate(zip(q_seqs, y_seqs)):
        length = q.shape[0]
        queries[i, :length] = q
        labels[i, :length] = y
        mask[i, :length] = True
    return queries, labels, mask


def collate_dkvmn_batch(batch: list[DKVMNSample]):
    predict_seqs, write_seqs, y_seqs = zip(*batch)
    max_len = max(s.shape[0] for s in predict_seqs)
    query_dim = predict_seqs[0].shape[-1]
    write_dim = write_seqs[0].shape[-1]
    predict_queries = torch.zeros(len(batch), max_len, query_dim, dtype=torch.float32)
    write_queries = torch.zeros(len(batch), max_len, write_dim, dtype=torch.float32)
    labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (pq, wq, y) in enumerate(zip(predict_seqs, write_seqs, y_seqs)):
        length = pq.shape[0]
        predict_queries[i, :length] = pq
        write_queries[i, :length] = wq
        labels[i, :length] = y
        mask[i, :length] = True
    return predict_queries, write_queries, labels, mask
