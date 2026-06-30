from __future__ import annotations

from dataclasses import dataclass

from .kt_runner_common import train_eval_kt
from .qdkt import QDKT

V8_LOGS = "data/processed/framework_logs_first_llm_error_process_misused_v8.jsonl"


@dataclass
class QDKTConfig:
    logs_path: str = V8_LOGS
    q_matrix_path: str = "data/metadata/q_matrix.csv"
    feature_mode: str = "problem_onehot"
    model_name: str = "qdkt"
    seed: int = 0
    epochs: int = 8
    batch_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_fraction: float = 1.0
    evidence_adapter_dim: int = 0
    out_dir: str = "runs/qdkt_v8"


def run_qdkt(cfg: QDKTConfig) -> dict:
    return train_eval_kt(
        cfg,
        build_model=QDKT,
        model_kwargs={
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
        },
        encoding_note=(
            "qDKT: 2N question-response one-hot history + evidence concat; "
            "N(+Q)-d next-question target"
        ),
    )
