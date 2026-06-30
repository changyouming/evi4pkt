"""KCGen-KT-lite: explicit KC mastery + CodeBERT (human 18-KC catalog)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KCGenKTLite(nn.Module):
    """
    Adapted KCGen-KT for CSEDM: human expert Q-matrix KCs, CodeBERT code, correctness-only.
    Maintains an explicit mastery vector updated on active KCs each step; predicts next pkt_label.
    """

    def __init__(
        self,
        num_kc: int,
        code_dim: int,
        *,
        hidden: int = 128,
        mastery_lr: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_kc = num_kc
        self.mastery_lr = mastery_lr
        self.code_proj = nn.Linear(code_dim, hidden)
        self.gru = nn.GRU(hidden + num_kc + 1, hidden, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden + num_kc + num_kc, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        code_seq: torch.Tensor,
        q_seq: torch.Tensor,
        response_seq: torch.Tensor,
        lengths: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        code_feat = self.code_proj(code_seq)
        inp = torch.cat([code_feat, q_seq, response_seq.unsqueeze(-1)], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            inp, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h_packed, _ = self.gru(packed)
        h_seq, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True)

        batch_size, max_len, _ = h_seq.shape
        mastery = torch.zeros(batch_size, self.num_kc, device=h_seq.device)
        for t in range(max_len):
            active = t < lengths
            if not active.any():
                break
            q_t = q_seq[:, t]
            r_t = response_seq[:, t].unsqueeze(-1)
            delta = self.mastery_lr * q_t * (r_t - mastery)
            mastery = mastery + delta * active.unsqueeze(-1).float()

        batch_idx = torch.arange(batch_size, device=h_seq.device)
        last_idx = (lengths - 1).clamp(min=0)
        h_last = h_seq[batch_idx, last_idx]
        return self.out(torch.cat([h_last, mastery, target_q], dim=-1)).squeeze(-1)
