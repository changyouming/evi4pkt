from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .code_evidence import CODE_EVIDENCE_VECTOR_DIM, code_evidence_source_for_mode
from .error_evidence import error_evidence_source_for_mode, error_evidence_vector_dim
from .evidence_v8 import evidence_v8_summary_fields
from .dataset import (
    FEATURE_MODES,
    MODES_USING_CODE,
    MODES_USING_ERROR,
    MODES_USING_PROCESS,
    MODES_USING_Q,
    load_framework_logs,
    split_students,
)
from .dkvmn import DKVMN
from .q_matrix import DEFAULT_Q_MATRIX_PATH, load_q_matrix, resolve_q_matrix
from .sequence_dataset import (
    DKVMNSample,
    build_dkvmn_samples,
    collate_dkvmn_batch,
)
from .train_sequence import run_sequence_epoch


@dataclass
class DKVMNConfig:
    logs_path: str = "data/processed/framework_logs_first.jsonl"
    q_matrix_path: str = DEFAULT_Q_MATRIX_PATH
    feature_mode: str = "problem_onehot"
    model_name: str = "dkvmn"
    seed: int = 0
    epochs: int = 8
    batch_size: int = 64
    num_memory: int = 12
    key_dim: int = 128
    value_dim: int = 128
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    train_fraction: float = 1.0
    out_dir: str = "runs/dkvmn_first"


class DKVMNDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[DKVMNSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def run_dkvmn(cfg: DKVMNConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_framework_logs(Path(cfg.logs_path))
    students = sorted({str(r["subject_id"]) for r in records})
    problem_ids = sorted({int(r["problem_id"]) for r in records})
    problem_id_to_index = {pid: i for i, pid in enumerate(problem_ids)}

    if cfg.feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported DKVMN feature_mode='{cfg.feature_mode}'.")

    problem_q_map = None
    q_kc_dim = 0
    if cfg.feature_mode in MODES_USING_Q:
        q_path = Path(cfg.q_matrix_path)
        if not q_path.is_absolute():
            q_path = Path.cwd() / q_path
        problem_q_map = load_q_matrix(q_path) if q_path.exists() else resolve_q_matrix(Path.cwd(), q_path)
        q_kc_dim = len(next(iter(problem_q_map.values())))

    split = split_students(students, seed=cfg.seed)
    if not (0.0 < cfg.train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0, 1].")
    if cfg.train_fraction < 1.0:
        n_train = max(1, int(round(len(split.train_students) * cfg.train_fraction)))
        train_students = split.train_students[:n_train]
    else:
        train_students = split.train_students

    train_samples = build_dkvmn_samples(
        records,
        train_students,
        problem_id_to_index,
        problem_q_map=problem_q_map,
        feature_mode=cfg.feature_mode,
    )
    valid_samples = build_dkvmn_samples(
        records,
        split.valid_students,
        problem_id_to_index,
        problem_q_map=problem_q_map,
        feature_mode=cfg.feature_mode,
    )
    test_samples = build_dkvmn_samples(
        records,
        split.test_students,
        problem_id_to_index,
        problem_q_map=problem_q_map,
        feature_mode=cfg.feature_mode,
    )
    if not train_samples or not valid_samples or not test_samples:
        raise RuntimeError("Empty train/valid/test sequence samples. Check logs or splits.")

    query_dim = train_samples[0][0].shape[-1]
    write_dim = train_samples[0][1].shape[-1]
    model = DKVMN(
        query_dim=query_dim,
        write_dim=write_dim,
        num_memory=cfg.num_memory,
        key_dim=cfg.key_dim,
        value_dim=cfg.value_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        DKVMNDataset(train_samples),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_dkvmn_batch,
    )
    valid_loader = DataLoader(
        DKVMNDataset(valid_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_dkvmn_batch,
    )
    test_loader = DataLoader(
        DKVMNDataset(test_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_dkvmn_batch,
    )

    best = {"epoch": -1, "valid_auc": -1.0, "state": None}
    history = []
    for ep in range(1, cfg.epochs + 1):
        tr = run_sequence_epoch(model, train_loader, optimizer, device)
        va = run_sequence_epoch(model, valid_loader, None, device)
        history.append({"epoch": ep, "train": asdict(tr), "valid": asdict(va)})
        if va.auc > best["valid_auc"]:
            best["valid_auc"] = va.auc
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best["state"])
    te = run_sequence_epoch(model, test_loader, None, device)

    summary = {
        "config": asdict(cfg),
        "model_name": cfg.model_name,
        "device": str(device),
        "query_dim": query_dim,
        "write_dim": write_dim,
        "num_problems": len(problem_ids),
        "feature_mode": cfg.feature_mode,
        "q_kc_dim": q_kc_dim,
        "task_q_source": "dataset_expert" if cfg.feature_mode in MODES_USING_Q else None,
        "code_evidence_source": code_evidence_source_for_mode(cfg.feature_mode),
        "code_evidence_dim": CODE_EVIDENCE_VECTOR_DIM if cfg.feature_mode in MODES_USING_CODE else 0,
        "process_evidence_source": "kc_history" if cfg.feature_mode in MODES_USING_PROCESS else None,
        "process_evidence_dim": (2 * q_kc_dim) if cfg.feature_mode in MODES_USING_PROCESS else 0,
        "error_evidence_source": error_evidence_source_for_mode(cfg.feature_mode),
        "error_evidence_dim": (
            error_evidence_vector_dim(q_kc_dim) if cfg.feature_mode in MODES_USING_ERROR else 0
        ),
        **evidence_v8_summary_fields(cfg.feature_mode),
        "split_sizes": {
            "train_students": len(train_students),
            "full_train_students": len(split.train_students),
            "train_fraction": cfg.train_fraction,
            "valid_students": len(split.valid_students),
            "test_students": len(split.test_students),
            "train_sequences": len(train_samples),
            "valid_sequences": len(valid_samples),
            "test_sequences": len(test_samples),
            "train_interactions": sum(s[0].shape[0] for s in train_samples),
            "valid_interactions": sum(s[0].shape[0] for s in valid_samples),
            "test_interactions": sum(s[0].shape[0] for s in test_samples),
        },
        "label_source": "code_issues.pkt_label (score_ge_1.0)",
        "evaluation_protocol": "predict_target_query_then_write_evidence",
        "best_epoch": best["epoch"],
        "best_valid_auc": best["valid_auc"],
        "test_metrics": asdict(te),
        "history": history,
    }

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"result_seed{cfg.seed}.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
