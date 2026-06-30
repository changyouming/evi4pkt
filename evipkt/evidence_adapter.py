"""Optional bottleneck adapter on concatenated evidence channels (backbone unchanged)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .code_evidence import CODE_EVIDENCE_VECTOR_DIM
from .code2vec_features import CODE2VEC_VECTOR_DIM
from .dataset import (
    MODES_USING_CODE,
    MODES_USING_CODE2VEC,
    MODES_USING_ERROR,
    MODES_USING_PROCESS,
    MODES_USING_Q,
)
from .error_evidence import error_evidence_vector_dim
from .feature_modes import normalize_feature_mode
from .plugplay_evidence import (
    CODE_EVIDENCE_DIM,
    MECHANISM_EVIDENCE_DIM,
    MODES_USING_CODE as MODES_USING_PLUGPLAY_CODE,
    MODES_USING_MECHANISM as MODES_USING_PLUGPLAY_MECHANISM,
)


def interaction_base_dim(feature_mode: str, num_problems: int) -> int:
    if feature_mode == "q_only":
        return 0
    return 2 * num_problems


def interaction_evidence_dim(feature_mode: str, q_kc_dim: int) -> int:
    """Dims after problem×correctness one-hot: Q + legacy code/error + plug-and-play + process."""
    mode = normalize_feature_mode(feature_mode)
    dim = 0
    if mode in MODES_USING_Q and mode != "q_only":
        dim += q_kc_dim
    if mode in MODES_USING_CODE:
        dim += CODE_EVIDENCE_VECTOR_DIM
    if mode in MODES_USING_PLUGPLAY_CODE:
        dim += CODE_EVIDENCE_DIM
    if mode in MODES_USING_PLUGPLAY_MECHANISM:
        dim += MECHANISM_EVIDENCE_DIM
    if feature_mode in MODES_USING_PROCESS:
        dim += 2 * q_kc_dim
    if feature_mode in MODES_USING_ERROR:
        dim += error_evidence_vector_dim(q_kc_dim)
    if mode in MODES_USING_CODE2VEC:
        dim += CODE2VEC_VECTOR_DIM
    return dim


class EvidenceAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence)


class SequenceModelWithEvidenceAdapter(nn.Module):
    """Split history features into base + evidence; adapt evidence then call backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        base_dim: int,
        evidence_dim: int,
        adapter_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if evidence_dim <= 0:
            raise ValueError("evidence_dim must be positive when using an evidence adapter.")
        if adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive when using an evidence adapter.")
        self.backbone = backbone
        self.base_dim = base_dim
        self.evidence_dim = evidence_dim
        self.adapter = EvidenceAdapter(evidence_dim, adapter_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = x[..., : self.base_dim]
        evidence = x[..., self.base_dim : self.base_dim + self.evidence_dim]
        adapted = self.adapter(evidence)
        x_in = torch.cat([base, adapted], dim=-1)
        return self.backbone(x_in, lengths, target)


def maybe_wrap_evidence_adapter(
    backbone: nn.Module,
    *,
    feature_mode: str,
    num_problems: int,
    q_kc_dim: int,
    raw_input_dim: int,
    adapter_dim: int,
    dropout: float = 0.1,
) -> tuple[nn.Module, dict]:
    """Return (model, metadata). Wraps backbone when adapter_dim > 0 and evidence exists."""
    evidence_dim = interaction_evidence_dim(feature_mode, q_kc_dim)
    base_dim = interaction_base_dim(feature_mode, num_problems)
    meta = {
        "evidence_adapter_dim": 0,
        "evidence_raw_dim": evidence_dim,
        "interaction_base_dim": base_dim,
        "backbone_input_dim": raw_input_dim,
    }
    if adapter_dim <= 0 or evidence_dim <= 0:
        return backbone, meta
    expected = base_dim + evidence_dim
    if raw_input_dim != expected:
        raise ValueError(
            f"raw_input_dim={raw_input_dim} != base({base_dim}) + evidence({evidence_dim})."
        )
    wrapped = SequenceModelWithEvidenceAdapter(
        backbone,
        base_dim=base_dim,
        evidence_dim=evidence_dim,
        adapter_dim=adapter_dim,
        dropout=dropout,
    )
    meta.update(
        {
            "evidence_adapter_dim": adapter_dim,
            "backbone_input_dim": base_dim + adapter_dim,
        }
    )
    return wrapped, meta
