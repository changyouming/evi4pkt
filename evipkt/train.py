from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EpochMetrics:
    loss: float
    auc: float
    acc: float
    f1: float


def _binary_metrics(y_true: torch.Tensor, logits: torch.Tensor) -> tuple[float, float, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    y = y_true.float()
    acc = float((pred == y).float().mean().item()) if y.numel() else 0.0
    tp = float(((pred == 1) & (y == 1)).sum().item())
    fp = float(((pred == 1) & (y == 0)).sum().item())
    fn = float(((pred == 0) & (y == 1)).sum().item())
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    # AUC via rank statistic (no sklearn dependency)
    y_np = y.detach().cpu()
    p_np = probs.detach().cpu()
    pos = y_np == 1
    neg = y_np == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        pos_scores = p_np[pos]
        neg_scores = p_np[neg]
        auc_num = 0.0
        for ps in pos_scores:
            auc_num += float((ps > neg_scores).sum().item()) + 0.5 * float((ps == neg_scores).sum().item())
        auc = auc_num / (n_pos * n_neg)
    return auc, acc, f1


def run_epoch(model, loader, optimizer, device) -> EpochMetrics:
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, lengths, target, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        target = target.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x, lengths, target)
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
