from __future__ import annotations

from dataclasses import dataclass

import torch

from .train import _binary_metrics


@dataclass
class SequenceEpochMetrics:
    loss: float
    auc: float
    acc: float
    f1: float


def run_gkt_epoch(model, loader, optimizer, device) -> SequenceEpochMetrics:
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    total_count = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for q, r, evidence, labels, pred_mask in loader:
        q = q.to(device)
        r = r.to(device)
        labels = labels.to(device)
        pred_mask = pred_mask.to(device)
        evidence_t = evidence.to(device) if evidence is not None else None
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(q, r, evidence_t)
            loss_raw = bce(logits, labels)
            loss = loss_raw[pred_mask].mean()
            if train:
                loss.backward()
                optimizer.step()
        count = int(pred_mask.sum().item())
        total_loss += float(loss.item()) * count
        total_count += count
        all_logits.append(logits.detach()[pred_mask].cpu())
        all_labels.append(labels.detach()[pred_mask].cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    auc, acc, f1 = _binary_metrics(labels_cat, logits_cat)
    return SequenceEpochMetrics(
        loss=total_loss / max(1, total_count),
        auc=auc,
        acc=acc,
        f1=f1,
    )


def run_lpkt_epoch(model, loader, optimizer, device) -> SequenceEpochMetrics:
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    total_count = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for e, a, it, at, evidence, labels, pred_mask in loader:
        e = e.to(device)
        a = a.to(device)
        it = it.to(device)
        at = at.to(device)
        labels = labels.to(device)
        pred_mask = pred_mask.to(device)
        evidence_t = evidence.to(device) if evidence is not None else None
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(e, a, it, at, evidence_t)
            loss_raw = bce(logits, labels)
            loss = loss_raw[pred_mask].mean()
            if train:
                loss.backward()
                optimizer.step()
        count = int(pred_mask.sum().item())
        total_loss += float(loss.item()) * count
        total_count += count
        all_logits.append(logits.detach()[pred_mask].cpu())
        all_labels.append(labels.detach()[pred_mask].cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    auc, acc, f1 = _binary_metrics(labels_cat, logits_cat)
    return SequenceEpochMetrics(
        loss=total_loss / max(1, total_count),
        auc=auc,
        acc=acc,
        f1=f1,
    )


def run_sequence_epoch(model, loader, optimizer, device) -> SequenceEpochMetrics:
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    total_count = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        if len(batch) == 4:
            queries, write_queries, labels, mask = batch
            write_queries = write_queries.to(device)
        else:
            queries, labels, mask = batch
            write_queries = None
        queries = queries.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            if write_queries is not None:
                logits = model(queries, labels, write_queries)
            else:
                logits = model(queries, labels)
            loss_raw = bce(logits, labels)
            loss = loss_raw[mask].mean()
            if train:
                loss.backward()
                optimizer.step()
        count = int(mask.sum().item())
        total_loss += float(loss.item()) * count
        total_count += count
        all_logits.append(logits.detach()[mask].cpu())
        all_labels.append(labels.detach()[mask].cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    auc, acc, f1 = _binary_metrics(labels_cat, logits_cat)
    return SequenceEpochMetrics(
        loss=total_loss / max(1, total_count),
        auc=auc,
        acc=acc,
        f1=f1,
    )

