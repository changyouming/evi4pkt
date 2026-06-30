"""Dataset builders for IICE-lite on framework logs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .codebert_features import CODEBERT_VECTOR_DIM, code_cache_key
from .dataset import load_framework_logs
from .q_matrix import load_q_matrix

IICELiteSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _student_code(record: dict) -> str:
    return str((record.get("student_code") or {}).get("code") or "")


def _pkt_label(record: dict) -> int:
    return int(record.get("pkt_label", record.get("code_issues", {}).get("pkt_label", 0)))


def build_iice_lite_samples(
    records: List[dict],
    students: List[str],
    problem_q_map: Dict[int, List[float]],
    codebert_cache: dict[str, list[float]],
    *,
    q_dim: int | None = None,
) -> List[IICELiteSample]:
    if not codebert_cache:
        raise ValueError("codebert_cache is required.")
    student_set = set(students)
    by_student: Dict[str, List[dict]] = {}
    for rec in records:
        sid = str(rec["subject_id"])
        if sid in student_set:
            by_student.setdefault(sid, []).append(rec)

    q_dim = q_dim or len(next(iter(problem_q_map.values())))
    zero_code = [0.0] * CODEBERT_VECTOR_DIM
    samples: List[IICELiteSample] = []

    for rows in by_student.values():
        rows.sort(key=lambda r: int(r["trajectory"]["student_timestep"]))
        prob_rows: List[torch.Tensor] = []
        code_rows: List[torch.Tensor] = []
        labels: List[int] = []
        for row in rows:
            pid = int(row["problem_id"])
            q_row = problem_q_map.get(pid)
            if q_row is None:
                continue
            key = code_cache_key(_student_code(row))
            code_vec = codebert_cache.get(key, zero_code)
            prob_rows.append(torch.tensor(q_row[:q_dim], dtype=torch.float32))
            code_rows.append(torch.tensor(code_vec, dtype=torch.float32))
            labels.append(_pkt_label(row))
        if len(prob_rows) < 2:
            continue
        for t in range(1, len(prob_rows)):
            hist_len = t
            prob_hist = torch.stack(prob_rows[:t], dim=0)
            code_hist = torch.stack(code_rows[:t], dim=0)
            r_hist = torch.tensor(labels[:t], dtype=torch.float32)
            target_q = prob_rows[t]
            y = torch.tensor(float(labels[t]), dtype=torch.float32)
            samples.append(
                (
                    prob_hist,
                    code_hist,
                    r_hist,
                    target_q,
                    y,
                    torch.tensor(hist_len, dtype=torch.long),
                )
            )
    return samples


class IICELiteDataset(Dataset):
    def __init__(self, samples: List[IICELiteSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> IICELiteSample:
        return self.samples[idx]


def collate_iice_lite_batch(batch: List[IICELiteSample]):
    prob_seqs, code_seqs, r_seqs, targets, labels, lengths = zip(*batch)
    max_len = max(int(x.item()) for x in lengths)
    q_dim = prob_seqs[0].shape[-1]
    code_dim = code_seqs[0].shape[-1]
    batch_size = len(batch)
    prob_pad = torch.zeros(batch_size, max_len, q_dim, dtype=torch.float32)
    code_pad = torch.zeros(batch_size, max_len, code_dim, dtype=torch.float32)
    r_pad = torch.zeros(batch_size, max_len, dtype=torch.float32)
    for i, (p, c, r, ln) in enumerate(zip(prob_seqs, code_seqs, r_seqs, lengths)):
        t = int(ln.item())
        prob_pad[i, :t] = p
        code_pad[i, :t] = c
        r_pad[i, :t] = r
    return (
        prob_pad,
        code_pad,
        r_pad,
        torch.stack(lengths).long(),
        torch.stack(targets).float(),
        torch.stack(labels).float(),
    )


@dataclass
class IICELiteConfig:
    logs_path: str = "data/processed/framework_logs_first.jsonl"
    q_matrix_path: str = "data/metadata/q_matrix.csv"
    codebert_cache_path: str = "data/processed/codebert_cache_f19.jsonl"
    seed: int = 0
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    prob_hidden: int = 64
    code_hidden: int = 128
    gru_hidden: int = 128
    decay_lambda: float = 0.1
    dropout: float = 0.1
    train_fraction: float = 1.0
    out_dir: str = "runs/iice_lite_f19"
    model_name: str = "iice_lite"


def load_records_and_q(cfg: IICELiteConfig) -> tuple[list[dict], dict[int, list[float]]]:
    records = load_framework_logs(Path(cfg.logs_path))
    q_map = load_q_matrix(Path(cfg.q_matrix_path))
    return records, q_map
