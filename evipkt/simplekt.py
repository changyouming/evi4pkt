"""SimpleKT-style Rasch interaction + self-attentive history pooling."""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleKT(nn.Module):
    """
    Self-attention over history with Rasch-like multiplicative interaction between
    pooled ability state and target problem/Q representation.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.max_seq_len = max_seq_len
        self.history_proj = nn.Linear(input_dim, d_model)
        self.target_proj = nn.Linear(target_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.self_attn = nn.MultiheadAttention(
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
        self.rasch = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

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
        key_padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_out, _ = self.self_attn(
            query=hist,
            key=hist,
            value=hist,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hist = hist + attn_out
        hist = hist + self.ffn(hist)

        idx = (lengths - 1).clamp(min=0)
        ability = hist[torch.arange(batch_size, device=x.device), idx]
        item = self.target_proj(target)
        h = ability * torch.tanh(self.rasch(item))
        return self.out(h).squeeze(-1)
