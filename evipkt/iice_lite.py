"""Lightweight IICE-PKT-style dual-GRU model (Q + CodeBERT, decay attention)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def integrate_response(feat: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """IICE-style: [feat, 0] if correct else [0, feat]. feat [B,T,D], response [B,T]."""
    zeros = torch.zeros_like(feat)
    mask = (response > 0.5).float().unsqueeze(-1)
    return torch.cat([feat * mask, feat * (1.0 - mask)], dim=-1)


class IICELite(nn.Module):
    """
    Dual GRU over problem (Q) and code (CodeBERT) sequences with decay attention.
    Predicts next-step pkt_label from history + target-problem Q vector.
    """

    def __init__(
        self,
        q_dim: int,
        code_dim: int,
        *,
        prob_hidden: int = 64,
        code_hidden: int = 128,
        gru_hidden: int = 128,
        decay_lambda: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decay_lambda = decay_lambda
        self.prob_proj = nn.Linear(q_dim, prob_hidden)
        self.code_proj = nn.Linear(code_dim, code_hidden)
        self.target_proj = nn.Linear(q_dim, gru_hidden)
        self.prob_gru = nn.GRU(prob_hidden * 2, gru_hidden, batch_first=True)
        self.code_gru = nn.GRU(code_hidden * 2, gru_hidden, batch_first=True)
        self.attn_prob = nn.Linear(gru_hidden, gru_hidden, bias=False)
        self.attn_code = nn.Linear(gru_hidden, gru_hidden, bias=False)
        self.out = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1),
        )

    def _decay_attention(
        self,
        target: torch.Tensor,
        hist: torch.Tensor,
        lengths: torch.Tensor,
        attn: nn.Linear,
    ) -> torch.Tensor:
        """target [B,H], hist [B,T,H], lengths [B] history lengths."""
        batch_size, seq_len, hidden = hist.shape
        scores = torch.bmm(hist, attn(target).unsqueeze(-1)).squeeze(-1)
        positions = torch.arange(seq_len, device=hist.device).float()
        dist = (seq_len - 1) - positions
        decay = torch.exp(-self.decay_lambda * dist).unsqueeze(0)
        scores = scores * decay
        mask = torch.arange(seq_len, device=hist.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.bmm(weights.unsqueeze(1), hist).squeeze(1)

    def forward(
        self,
        prob_seq: torch.Tensor,
        code_seq: torch.Tensor,
        response_seq: torch.Tensor,
        lengths: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        prob_feat = integrate_response(self.prob_proj(prob_seq), response_seq)
        code_feat = integrate_response(self.code_proj(code_seq), response_seq)
        prob_out, _ = self.prob_gru(prob_feat)
        code_out, _ = self.code_gru(code_feat)
        target_h = self.target_proj(target_q)
        o_prob = self._decay_attention(target_h, prob_out, lengths, self.attn_prob)
        o_code = self._decay_attention(target_h, code_out, lengths, self.attn_code)
        return self.out(torch.cat([o_prob, o_code], dim=-1)).squeeze(-1)
