"""Shared training loop for sequence KT backbones (plug-and-play evidence concat)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .code_evidence import CODE_EVIDENCE_VECTOR_DIM, code_evidence_source_for_mode
from .dataset import (
    DKTPrefixDataset,
    FEATURE_MODES,
    MODES_USING_CODE,
    MODES_USING_ERROR,
    MODES_USING_PROCESS,
    MODES_USING_Q,
    build_dkt_samples,
    collate_dkt_batch,
    load_framework_logs,
    resolve_student_split,
    split_students,
)
from .error_evidence import error_evidence_source_for_mode, error_evidence_vector_dim
from .evidence_adapter import interaction_evidence_dim, maybe_wrap_evidence_adapter
from .evidence_v8 import evidence_v8_summary_fields
from .q_matrix import DEFAULT_Q_MATRIX_PATH, load_q_matrix, resolve_q_matrix
from .train import run_epoch


def build_kt_data(cfg, *, feature_mode: str | None = None):
    """Load logs, splits, and DKT-style samples for any backbone."""
    feature_mode = feature_mode or cfg.feature_mode
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode='{feature_mode}'.")

    records = load_framework_logs(Path(cfg.logs_path))
    students = sorted({str(r["subject_id"]) for r in records})
    problem_ids = sorted({int(r["problem_id"]) for r in records})
    problem_id_to_index = {pid: i for i, pid in enumerate(problem_ids)}

    problem_q_map = None
    q_kc_dim = 0
    if feature_mode in MODES_USING_Q:
        q_path = Path(cfg.q_matrix_path)
        if not q_path.is_absolute():
            q_path = Path.cwd() / q_path
        problem_q_map = load_q_matrix(q_path) if q_path.exists() else resolve_q_matrix(Path.cwd(), q_path)
        q_kc_dim = len(next(iter(problem_q_map.values())))

    split = resolve_student_split(students, seed=cfg.seed, logs_path=cfg.logs_path)
    if not (0.0 < cfg.train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0, 1].")
    if cfg.train_fraction < 1.0:
        n_train = max(1, int(round(len(split.train_students) * cfg.train_fraction)))
        train_students = split.train_students[:n_train]
    else:
        train_students = split.train_students

    kwargs = dict(
        problem_id_to_index=problem_id_to_index,
        problem_q_map=problem_q_map,
        feature_mode=feature_mode,
    )
    train_samples = build_dkt_samples(records, train_students, **kwargs)
    valid_samples = build_dkt_samples(records, split.valid_students, **kwargs)
    test_samples = build_dkt_samples(records, split.test_students, **kwargs)
    if not train_samples or not valid_samples or not test_samples:
        raise RuntimeError("Empty train/valid/test samples.")

    return {
        "records": records,
        "problem_ids": problem_ids,
        "problem_id_to_index": problem_id_to_index,
        "split": split,
        "train_students": train_students,
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "test_samples": test_samples,
        "q_kc_dim": q_kc_dim,
        "feature_mode": feature_mode,
    }


def train_eval_kt(
    cfg,
    *,
    build_model: Callable[..., nn.Module],
    model_kwargs: dict[str, Any] | None = None,
    feature_mode: str | None = None,
    encoding_note: str = "",
) -> dict:
    """Train/eval a backbone with standard plug-and-play evidence concat."""
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_kt_data(cfg, feature_mode=feature_mode)
    feature_mode = data["feature_mode"]
    q_kc_dim = data["q_kc_dim"]
    train_samples = data["train_samples"]
    valid_samples = data["valid_samples"]
    test_samples = data["test_samples"]
    split = data["split"]
    train_students = data["train_students"]

    raw_input_dim = train_samples[0][0].shape[-1]
    target_dim = train_samples[0][2].shape[-1]
    evidence_dim = interaction_evidence_dim(feature_mode, q_kc_dim)
    adapter_dim = int(getattr(cfg, "evidence_adapter_dim", 0) or 0)
    backbone_input_dim = (
        raw_input_dim - evidence_dim + adapter_dim
        if adapter_dim > 0 and evidence_dim > 0
        else raw_input_dim
    )

    mk = dict(model_kwargs or {})
    mk.setdefault("input_dim", backbone_input_dim)
    mk.setdefault("target_dim", target_dim)
    backbone = build_model(**mk)
    model, adapter_meta = maybe_wrap_evidence_adapter(
        backbone,
        feature_mode=feature_mode,
        num_problems=len(data["problem_ids"]),
        q_kc_dim=q_kc_dim,
        raw_input_dim=raw_input_dim,
        adapter_dim=adapter_dim,
        dropout=getattr(cfg, "dropout", 0.1),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loader = DataLoader(
        DKTPrefixDataset(train_samples),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_dkt_batch,
    )
    valid_loader = DataLoader(
        DKTPrefixDataset(valid_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_dkt_batch,
    )
    test_loader = DataLoader(
        DKTPrefixDataset(test_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_dkt_batch,
    )

    best = {"epoch": -1, "valid_auc": -1.0, "state": None}
    history = []
    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device)
        va = run_epoch(model, valid_loader, None, device)
        history.append({"epoch": ep, "train": asdict(tr), "valid": asdict(va)})
        if va.auc > best["valid_auc"]:
            best["valid_auc"] = va.auc
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best["state"])
    te = run_epoch(model, test_loader, None, device)

    summary = {
        "config": asdict(cfg),
        "model_name": cfg.model_name,
        "device": str(device),
        "encoding": encoding_note,
        "input_dim": raw_input_dim,
        "backbone_input_dim": adapter_meta["backbone_input_dim"],
        "evidence_adapter_dim": adapter_meta["evidence_adapter_dim"],
        "evidence_raw_dim": adapter_meta["evidence_raw_dim"],
        "interaction_base_dim": adapter_meta["interaction_base_dim"],
        "target_dim": target_dim,
        "num_problems": len(data["problem_ids"]),
        "feature_mode": feature_mode,
        "q_kc_dim": q_kc_dim,
        "task_q_source": "dataset_expert" if feature_mode in MODES_USING_Q else None,
        "code_evidence_source": code_evidence_source_for_mode(feature_mode),
        "code_evidence_dim": CODE_EVIDENCE_VECTOR_DIM if feature_mode in MODES_USING_CODE else 0,
        "process_evidence_source": "kc_history" if feature_mode in MODES_USING_PROCESS else None,
        "process_evidence_dim": (2 * q_kc_dim) if feature_mode in MODES_USING_PROCESS else 0,
        "error_evidence_source": error_evidence_source_for_mode(feature_mode),
        "error_evidence_dim": (
            error_evidence_vector_dim(q_kc_dim) if feature_mode in MODES_USING_ERROR else 0
        ),
        **evidence_v8_summary_fields(feature_mode),
        "split_sizes": {
            "train_students": len(train_students),
            "full_train_students": len(split.train_students),
            "train_fraction": cfg.train_fraction,
            "valid_students": len(split.valid_students),
            "test_students": len(split.test_students),
            "train_samples": len(train_samples),
            "valid_samples": len(valid_samples),
            "test_samples": len(test_samples),
        },
        "label_source": "code_issues.pkt_label (score_ge_1.0)",
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
