from __future__ import annotations

import torch
import torch.nn as nn


class DKT(nn.Module):
    """LSTM-based DKT with optional next-item conditioning."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim + target_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, T, D], lengths: [B] (number of history steps, >=1)
        target: [B, target_dim] next-problem / next-KC features
        Returns logits for the next interaction: [B]
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        if self.target_dim:
            if target is None:
                raise ValueError("target features are required when target_dim > 0.")
            h_last = torch.cat([h_last, target], dim=-1)
        return self.out(h_last).squeeze(-1)
