"""Question-aware interactive KT (QIKT-style target-conditioned attention)."""
from __future__ import annotations

import torch
import torch.nn as nn


class QIKT(nn.Module):
    """
    History encodes full interaction vectors; query fuses problem one-hot and Q-vector
    from the target step (question-aware readout).
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        num_problems: int,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_problems = num_problems
        self.max_seq_len = max_seq_len
        self.q_kc_dim = max(0, target_dim - num_problems)

        self.history_proj = nn.Linear(input_dim, d_model)
        self.problem_proj = nn.Linear(num_problems, d_model)
        self.kc_proj = (
            nn.Linear(self.q_kc_dim, d_model) if self.q_kc_dim > 0 else None
        )
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def _query(self, target: torch.Tensor) -> torch.Tensor:
        p = target[:, : self.num_problems]
        q = self.problem_proj(p)
        if self.kc_proj is not None and self.q_kc_dim > 0:
            q = q + self.kc_proj(target[:, self.num_problems :])
        return q

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
        query = self._query(target).unsqueeze(1)

        key_padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_out, _ = self.attn(
            query=query,
            key=hist,
            value=hist,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = query.squeeze(1) + attn_out.squeeze(1)
        h = h + self.ffn(h)
        return self.out(h).squeeze(-1)
