from __future__ import annotations

import torch
import torch.nn as nn


class SAKT(nn.Module):
    """Target-conditioned self-attentive knowledge tracing."""

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
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        target: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, input_dim] historical interactions
        lengths: [B] valid history lengths
        target: [B, target_dim] current problem/KC query
        Returns logits for the current interaction: [B]
        If return_attention=True, also returns attn_weights [B, num_heads, 1, T].
        """
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}.")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        hist = self.history_proj(x) + self.pos_embedding(positions)
        query = self.target_proj(target).unsqueeze(1)

        key_padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_out, attn_weights = self.attn(
            query=query,
            key=hist,
            value=hist,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        h = query.squeeze(1) + attn_out.squeeze(1)
        h = h + self.ffn(h)
        logits = self.out(h).squeeze(-1)
        if return_attention:
            return logits, attn_weights
        return logits

