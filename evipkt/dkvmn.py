from __future__ import annotations

import torch
import torch.nn as nn


class DKVMN(nn.Module):
    """Dynamic Key-Value Memory Network for knowledge tracing."""

    def __init__(
        self,
        query_dim: int,
        write_dim: int | None = None,
        num_memory: int = 18,
        key_dim: int = 64,
        value_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.write_dim = write_dim if write_dim is not None else query_dim
        self.num_memory = num_memory
        self.value_dim = value_dim

        self.query_proj = nn.Linear(query_dim, key_dim)
        self.key_memory = nn.Parameter(torch.randn(num_memory, key_dim) * 0.1)
        self.init_value_memory = nn.Parameter(torch.randn(num_memory, value_dim) * 0.1)

        self.erase = nn.Sequential(nn.Linear(self.write_dim + 1, value_dim), nn.Sigmoid())
        self.add = nn.Sequential(nn.Linear(self.write_dim + 1, value_dim), nn.Tanh())
        self.predict = nn.Sequential(
            nn.Linear(query_dim + value_dim, value_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(value_dim, 1),
        )

    def forward(
        self,
        queries: torch.Tensor,
        labels: torch.Tensor | None = None,
        write_queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        queries: [B, T, query_dim] target query used for read/predict (no current-step evidence)
        write_queries: [B, T, write_dim] features used for memory write after observing y_t
        labels: [B, T], required during training/evaluation to update memory after prediction
        Returns logits before observing each label: [B, T]
        """
        if write_queries is None:
            write_queries = queries
        batch_size, seq_len, _ = queries.shape
        value_memory = self.init_value_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()
        logits: list[torch.Tensor] = []

        for t in range(seq_len):
            q_t = queries[:, t, :]
            w_t = write_queries[:, t, :]
            key = self.query_proj(q_t)
            weights = torch.softmax(key @ self.key_memory.T, dim=-1)
            read = torch.bmm(weights.unsqueeze(1), value_memory).squeeze(1)
            logits.append(self.predict(torch.cat([q_t, read], dim=-1)).squeeze(-1))

            if labels is not None:
                y_t = labels[:, t : t + 1]
                write_input = torch.cat([w_t, y_t], dim=-1)
                erase = self.erase(write_input).unsqueeze(1)
                add = self.add(write_input).unsqueeze(1)
                w = weights.unsqueeze(-1)
                value_memory = value_memory * (1.0 - w * erase) + w * add

        return torch.stack(logits, dim=1)

