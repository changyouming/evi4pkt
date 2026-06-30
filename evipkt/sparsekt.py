"""SparseKT-style top-k sparse cross-attention readout."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseKT(nn.Module):
    """Target-conditioned attention with top-k sparsification over history positions."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        topk: int = 16,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.max_seq_len = max_seq_len
        self.topk = topk
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.history_proj = nn.Linear(input_dim, d_model)
        self.target_proj = nn.Linear(target_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def _sparse_attn(self, query: torch.Tensor, hist: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = hist.shape
        q = self.q_proj(query).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hist).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hist).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        mask = torch.arange(seq_len, device=hist.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        k_use = min(self.topk, seq_len)
        top_scores, top_idx = torch.topk(scores, k=k_use, dim=-1)
        sparse = torch.full_like(scores, float("-inf"))
        sparse.scatter_(-1, top_idx, top_scores)
        weights = F.softmax(sparse, dim=-1)
        weights = self.dropout(weights)
        ctx = torch.matmul(weights, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, 1, d_model)
        return self.out_proj(ctx).squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("target is required.")
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        hist = self.history_proj(x) + self.pos_embedding(positions)
        query = self.target_proj(target)
        h = query + self._sparse_attn(query, hist, lengths)
        h = h + self.ffn(h)
        return self.head(h).squeeze(-1)
