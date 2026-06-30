"""Train/eval IICE-lite on CSEDM F19 framework logs."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .codebert_features import CODEBERT_VECTOR_DIM, load_codebert_cache
from .iice_lite import IICELite
from .iice_lite_dataset import (
    IICELiteConfig,
    IICELiteDataset,
    build_iice_lite_samples,
    collate_iice_lite_batch,
    load_records_and_q,
)
from .dataset import split_students
from .train import EpochMetrics, _binary_metrics


def _run_iice_epoch(model, loader, optimizer, device) -> EpochMetrics:
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for prob, code, resp, lengths, target_q, y in loader:
        prob = prob.to(device)
        code = code.to(device)
        resp = resp.to(device)
        lengths = lengths.to(device)
        target_q = target_q.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(prob, code, resp, lengths, target_q)
            loss = bce(logits, y)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        all_logits.append(logits.detach())
        all_labels.append(y.detach())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    auc, acc, f1 = _binary_metrics(labels_cat, logits_cat)
    return EpochMetrics(
        loss=total_loss / max(1, n_batches),
        auc=auc,
        acc=acc,
        f1=f1,
    )


def run_iice_lite(cfg: IICELiteConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records, q_map = load_records_and_q(cfg)
    q_dim = len(next(iter(q_map.values())))
    cache_path = Path(cfg.codebert_cache_path)
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    codebert_cache = load_codebert_cache(cache_path)
    if not codebert_cache:
        raise FileNotFoundError(
            f"Missing CodeBERT cache at {cache_path}. "
            "Run: python scripts/build_codebert_cache_f19.py"
        )

    students = sorted({str(r["subject_id"]) for r in records})
    split = split_students(students, seed=cfg.seed)
    if not (0.0 < cfg.train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0, 1].")
    train_students = (
        split.train_students[: max(1, int(round(len(split.train_students) * cfg.train_fraction)))]
        if cfg.train_fraction < 1.0
        else split.train_students
    )

    train_samples = build_iice_lite_samples(records, train_students, q_map, codebert_cache, q_dim=q_dim)
    valid_samples = build_iice_lite_samples(records, split.valid_students, q_map, codebert_cache, q_dim=q_dim)
    test_samples = build_iice_lite_samples(records, split.test_students, q_map, codebert_cache, q_dim=q_dim)
    if not train_samples or not valid_samples or not test_samples:
        raise RuntimeError("Empty train/valid/test samples for IICE-lite.")

    model = IICELite(
        q_dim=q_dim,
        code_dim=CODEBERT_VECTOR_DIM,
        prob_hidden=cfg.prob_hidden,
        code_hidden=cfg.code_hidden,
        gru_hidden=cfg.gru_hidden,
        decay_lambda=cfg.decay_lambda,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        IICELiteDataset(train_samples),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_iice_lite_batch,
    )
    valid_loader = DataLoader(
        IICELiteDataset(valid_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_iice_lite_batch,
    )
    test_loader = DataLoader(
        IICELiteDataset(test_samples),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_iice_lite_batch,
    )

    best = {"epoch": -1, "valid_auc": -1.0, "state": None}
    history = []
    for ep in range(1, cfg.epochs + 1):
        tr = _run_iice_epoch(model, train_loader, optimizer, device)
        va = _run_iice_epoch(model, valid_loader, None, device)
        history.append({"epoch": ep, "train": asdict(tr), "valid": asdict(va)})
        if va.auc > best["valid_auc"]:
            best["valid_auc"] = va.auc
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best["state"])
    te = _run_iice_epoch(model, test_loader, None, device)

    summary = {
        "config": asdict(cfg),
        "model_name": cfg.model_name,
        "device": str(device),
        "q_dim": q_dim,
        "codebert_dim": CODEBERT_VECTOR_DIM,
        "split_sizes": {
            "train_students": len(train_students),
            "valid_students": len(split.valid_students),
            "test_students": len(split.test_students),
            "train_samples": len(train_samples),
            "valid_samples": len(valid_samples),
            "test_samples": len(test_samples),
        },
        "best_epoch": best["epoch"],
        "best_valid_auc": best["valid_auc"],
        "test_metrics": asdict(te),
        "history": history,
    }

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"result_seed{cfg.seed}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
