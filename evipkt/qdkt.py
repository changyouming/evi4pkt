"""Question-centric DKT (Sonkar et al., 2020): 2N question–response encoding + evidence concat."""
from __future__ import annotations

import torch
import torch.nn as nn


class QDKT(nn.Module):
    """
    LSTM KT with question-level interaction encoding (2N one-hot + optional evidence).
    Target is next-question one-hot (+ Q when feature_mode includes Task evidence).
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.interaction_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
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
        x = self.interaction_proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        if target is None:
            raise ValueError("target (next-question features) is required.")
        h_last = torch.cat([h_last, target], dim=-1)
        return self.out(h_last).squeeze(-1)
