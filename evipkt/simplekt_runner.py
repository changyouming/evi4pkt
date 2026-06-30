from __future__ import annotations

from dataclasses import dataclass

from .kt_runner_common import build_kt_data, train_eval_kt
from .simplekt import SimpleKT

V8_LOGS = "data/processed/framework_logs_first_llm_error_process_misused_v8.jsonl"


@dataclass
class SimpleKTConfig:
    logs_path: str = V8_LOGS
    q_matrix_path: str = "data/metadata/q_matrix.csv"
    feature_mode: str = "problem_onehot"
    model_name: str = "simplekt"
    seed: int = 0
    epochs: int = 8
    batch_size: int = 64
    d_model: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_fraction: float = 1.0
    evidence_adapter_dim: int = 0
    out_dir: str = "runs/simplekt_v8"


def run_simplekt(cfg: SimpleKTConfig) -> dict:
    data = build_kt_data(cfg)
    max_train_len = max(
        int(s[1].item())
        for s in data["train_samples"]
        + data["valid_samples"]
        + data["test_samples"]
    )
    max_seq_len = max(cfg.max_seq_len, max_train_len)
    return train_eval_kt(
        cfg,
        build_model=SimpleKT,
        model_kwargs={
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "dropout": cfg.dropout,
            "max_seq_len": max_seq_len,
        },
        encoding_note="SimpleKT: self-attentive ability state with Rasch-style target interaction",
    )
