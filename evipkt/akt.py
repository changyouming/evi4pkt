from __future__ import annotations

import torch
import torch.nn as nn


class AKT(nn.Module):
    """Target-conditioned attentive KT with history self-attention and monotonic cross-attention."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        monotonic_rate: float = 0.2,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.max_seq_len = max_seq_len
        self.monotonic_rate = monotonic_rate
        self.history_proj = nn.Linear(input_dim, d_model)
        self.target_proj = nn.Linear(target_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
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
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def _key_padding_mask(self, lengths: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

    def _monotonic_cross_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Additive bias: older history positions receive higher logits (monotonic decay)."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        bias = -self.monotonic_rate * (seq_len - 1 - positions)
        return bias.unsqueeze(0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_dim] historical interactions
        lengths: [B] valid history lengths
        target: [B, target_dim] current problem/KC query
        Returns logits for the current interaction: [B]
        """
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        hist = self.history_proj(x) + self.pos_embedding(positions)
        key_padding_mask = self._key_padding_mask(lengths, seq_len, x.device)

        self_out, _ = self.self_attn(
            query=hist,
            key=hist,
            value=hist,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hist = hist + self_out

        query = self.target_proj(target).unsqueeze(1)
        cross_mask = self._monotonic_cross_mask(seq_len, x.device).expand(batch_size, -1)
        cross_mask = cross_mask.masked_fill(key_padding_mask, float("-inf"))
        # PyTorch MHA additive mask: (N * num_heads, L, S)
        cross_attn_mask = cross_mask.unsqueeze(1).repeat_interleave(
            self.cross_attn.num_heads, dim=0
        )
        cross_out, _ = self.cross_attn(
            query=query,
            key=hist,
            value=hist,
            attn_mask=cross_attn_mask,
            need_weights=False,
        )
        h = query.squeeze(1) + cross_out.squeeze(1)
        h = h + self.ffn(h)
        return self.out(h).squeeze(-1)
