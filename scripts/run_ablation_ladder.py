#!/usr/bin/env python3
"""Run Plan A evidence ablations across KT backbones.

Plan A (see docs/ablation_plan.md):
  v0→v1 spine; v2/v3 plugins; v4 +Process+Code; v5 Full.

Examples:
  # Full plug-and-play (8 backbones × 10 seeds, Full only):
  .venv/bin/python scripts/run_ablation_ladder.py --levels v5 --resume

  # Complete ablation ladder:
  .venv/bin/python scripts/run_ablation_ladder.py --resume

  # Main-paper four backbones only:
  .venv/bin/python scripts/run_ablation_ladder.py --levels v5 \\
      --backbones DKT DKVMN SAKT AKT --resume
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evipkt.akt_runner import AKTConfig, run_akt
from evipkt.dkvmn_runner import DKVMNConfig, run_dkvmn
from evipkt.feature_modes import (
    ABLATION_LADDER,
    ABLATION_LEVEL_ORDER,
    ABLATION_OUT_SUFFIX,
    CANONICAL_LOGS,
    LEGACY_ABLATION_OUT_SUFFIX,
    LEGACY_LOGS_V8,
    LEGACY_LEVEL_ALIASES,
    PIPELINE_ID,
    feature_modes_match,
    is_canonical_logs_path,
    normalize_ablation_level,
    normalize_feature_mode,
)
from evipkt.gkt_runner import GKTConfig, run_gkt
from evipkt.lpkt_runner import LPKTConfig, run_lpkt
from evipkt.qdkt_runner import QDKTConfig, run_qdkt
from evipkt.qikt_runner import QIKTConfig, run_qikt
from evipkt.runner import DKTConfig, run_dkt
from evipkt.sakt_runner import SAKTConfig, run_sakt
from evipkt.simplekt_runner import SimpleKTConfig, run_simplekt
from evipkt.sparsekt_runner import SparseKTConfig, run_sparsekt

METRICS = ("auc", "acc", "f1", "loss")

BACKBONE_RUNNERS: dict[str, dict] = {
    "DKT": {
        "prefix": "dkt",
        "runner": run_dkt,
        "config_cls": DKTConfig,
        "extra": {"hidden_dim": 128},
    },
    "DKVMN": {
        "prefix": "dkvmn",
        "runner": run_dkvmn,
        "config_cls": DKVMNConfig,
        "extra": {"num_memory": 12, "key_dim": 128, "value_dim": 128, "lr": 2e-3},
    },
    "SAKT": {
        "prefix": "sakt",
        "runner": run_sakt,
        "config_cls": SAKTConfig,
        "extra": {"d_model": 256, "num_heads": 8, "dropout": 0.1, "lr": 5e-4, "max_seq_len": 128},
    },
    "AKT": {
        "prefix": "akt",
        "runner": run_akt,
        "config_cls": AKTConfig,
        "extra": {"d_model": 128, "num_heads": 4, "max_seq_len": 128, "monotonic_rate": 0.2},
    },
    "GKT": {
        "prefix": "gkt",
        "runner": run_gkt,
        "config_cls": GKTConfig,
        "extra": {"hidden_dim": 64, "emb_size": 64, "dropout": 0.2, "lr": 1e-3},
    },
    "LPKT": {
        "prefix": "lpkt",
        "runner": run_lpkt,
        "config_cls": LPKTConfig,
        "extra": {"d_k": 64, "d_a": 64, "d_e": 64, "dropout": 0.2, "lr": 1e-3},
    },
    "qDKT": {
        "prefix": "qdkt",
        "runner": run_qdkt,
        "config_cls": QDKTConfig,
        "extra": {"hidden_dim": 128},
    },
    "QIKT": {
        "prefix": "qikt",
        "runner": run_qikt,
        "config_cls": QIKTConfig,
        "extra": {"d_model": 128, "num_heads": 4, "max_seq_len": 128},
    },
    "SimpleKT": {
        "prefix": "simplekt",
        "runner": run_simplekt,
        "config_cls": SimpleKTConfig,
        "extra": {"d_model": 128, "num_heads": 4, "max_seq_len": 128},
    },
    "SparseKT": {
        "prefix": "sparsekt",
        "runner": run_sparsekt,
        "config_cls": SparseKTConfig,
        "extra": {"d_model": 128, "num_heads": 4, "max_seq_len": 128, "topk": 16},
    },
}


def parse_args():
    p = argparse.ArgumentParser("Run V0–V6 evidence ablation ladder across backbones.")
    p.add_argument(
        "--backbones",
        type=str,
        nargs="+",
        default=list(BACKBONE_RUNNERS.keys()),
        choices=list(BACKBONE_RUNNERS.keys()),
    )
    p.add_argument(
        "--levels",
        type=str,
        nargs="+",
        default=list(ABLATION_LEVEL_ORDER),
        choices=list(ABLATION_LEVEL_ORDER) + list(LEGACY_LEVEL_ALIASES.keys()),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--q-matrix-path", type=str, default="data/metadata/q_matrix.csv")
    p.add_argument("--resume", action="store_true", help="Skip runs whose result JSON already exists.")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda (default), cuda:0, cpu, or auto",
    )
    p.add_argument(
        "--evidence-adapter-dim",
        type=int,
        default=0,
        help="If >0, apply bottleneck adapter to evidence channels (DKT/SAKT/AKT).",
    )
    return p.parse_args()


def _resolve_logs_path(project_root: Path) -> Path:
    canonical = project_root / CANONICAL_LOGS
    if canonical.exists():
        return canonical
    legacy = project_root / LEGACY_LOGS_V8
    if legacy.exists():
        return legacy
    return canonical


def _out_dir(prefix: str, adapter_dim: int = 0) -> str:
    adp = f"_eviadp{adapter_dim}" if adapter_dim > 0 else ""
    return f"runs/{prefix}_{ABLATION_OUT_SUFFIX}{adp}"


def _legacy_out_dir(prefix: str, adapter_dim: int = 0) -> str:
    adp = f"_eviadp{adapter_dim}" if adapter_dim > 0 else ""
    return f"runs/{prefix}_{LEGACY_ABLATION_OUT_SUFFIX}{adp}"


def _model_key(prefix: str, feature_mode: str) -> str:
    return normalize_feature_mode(feature_mode).replace("problem_", f"{prefix}_")


def _legacy_model_key(prefix: str, feature_mode: str) -> str:
    """Model dir name used before v8 suffix removal."""
    legacy = {
        "problem_plus_q_code": "problem_plus_q_code_v8",
        "problem_plus_q_mechanism": "problem_plus_q_mechanism_v8",
        "problem_plus_q_code_mechanism": "problem_plus_q_code_v8_mechanism_v8",
        "problem_plus_q_process_code_mechanism": "problem_plus_q_process_code_v8_mechanism_v8",
    }
    mode = normalize_feature_mode(feature_mode)
    legacy_mode = legacy.get(mode, mode)
    return legacy_mode.replace("problem_", f"{prefix}_")


def _result_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"result_seed{seed}.json"


def _expected_logs_path(project_root: Path, step: dict) -> Path:
    resolved = _resolve_logs_path(project_root)
    return resolved.resolve()


def _result_matches_step(
    result: dict,
    step: dict,
    project_root: Path,
    expected_epochs: int,
) -> bool:
    cfg = result.get("config", {})
    if not feature_modes_match(cfg.get("feature_mode", ""), step["feature_mode"]):
        return False
    if cfg.get("epochs") != expected_epochs:
        return False

    actual_logs = cfg.get("logs_path", "")
    if not actual_logs:
        return False
    if not is_canonical_logs_path(actual_logs):
        return False
    if Path(actual_logs).resolve() != _expected_logs_path(project_root, step):
        return False
    return True


def _find_cached_result(
    project_root: Path,
    prefix: str,
    step: dict,
    seed: int,
    args,
    adapter_dim: int,
) -> Path | None:
    canonical_dir = project_root / _out_dir(prefix, adapter_dim) / f"seed_{seed}" / _model_key(prefix, step["feature_mode"])
    legacy_dir = project_root / _legacy_out_dir(prefix, adapter_dim) / f"seed_{seed}" / _legacy_model_key(prefix, step["feature_mode"])
    for out_dir in (canonical_dir, legacy_dir):
        result_file = _result_path(out_dir, seed)
        if result_file.exists():
            cached = json.loads(result_file.read_text(encoding="utf-8"))
            if _result_matches_step(cached, step, project_root, args.epochs):
                return result_file
    return None


def _pack(summary: dict) -> dict:
    return {
        "feature_mode": summary["feature_mode"],
        "best_valid_auc": summary["best_valid_auc"],
        "best_epoch": summary["best_epoch"],
        "split_sizes": summary["split_sizes"],
        "test": summary["test_metrics"],
    }


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _summarize(rows: list[dict], model_keys: list[str]) -> dict:
    summary = {"models": {}}
    for key in model_keys:
        summary["models"][key] = {}
        for metric in METRICS:
            vals = [r[key]["test"][metric] for r in rows]
            summary["models"][key][metric] = _mean_std(vals)
    return summary


def _run_one(
    project_root: Path,
    backbone: str,
    step: dict,
    seed: int,
    args,
    out_root: Path,
    adapter_dim: int = 0,
) -> dict:
    spec = BACKBONE_RUNNERS[backbone]
    feature_mode = normalize_feature_mode(step["feature_mode"])
    model_key = _model_key(spec["prefix"], feature_mode)

    cached_path = _find_cached_result(project_root, spec["prefix"], step, seed, args, adapter_dim)
    if args.resume and cached_path is not None:
        return json.loads(cached_path.read_text(encoding="utf-8"))

    if args.resume:
        print(
            f"[{backbone}] seed={seed} {step['level']}: stale/missing result, re-running "
            f"({out_root / f'seed_{seed}' / model_key})"
        )

    logs_path = _resolve_logs_path(project_root)
    if not logs_path.exists():
        raise FileNotFoundError(
            f"Missing {logs_path} for {backbone} {step['level']}. "
            "Run enrich scripts first (see scripts/enrich_*.py)."
        )

    out_dir = out_root / f"seed_{seed}" / model_key
    cfg_kwargs = {
        "logs_path": str(logs_path.resolve()),
        "q_matrix_path": str((project_root / args.q_matrix_path).resolve()),
        "feature_mode": feature_mode,
        "model_name": model_key,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_fraction": 1.0,
        "device": args.device,
        "evidence_adapter_dim": adapter_dim,
        "out_dir": str(out_dir.resolve()),
        **spec["extra"],
    }
    allowed = {f.name for f in fields(spec["config_cls"])}
    cfg = spec["config_cls"](**{k: v for k, v in cfg_kwargs.items() if k in allowed})
    return spec["runner"](cfg)


def main():
    args = parse_args()
    project_root = PROJECT_ROOT
    steps = [s for s in ABLATION_LADDER if s["level"] in {normalize_ablation_level(l) for l in args.levels}]
    if not steps:
        raise SystemExit(f"No steps selected for levels={args.levels}")

    print(f"Levels={[s['level'] for s in steps]}  seeds={args.seeds}")
    if args.evidence_adapter_dim > 0:
        print(f"Evidence adapter dim: {args.evidence_adapter_dim} (DKT/SAKT/AKT only)")

    for backbone in args.backbones:
        spec = BACKBONE_RUNNERS[backbone]
        out_root = (project_root / _out_dir(spec["prefix"], args.evidence_adapter_dim)).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        model_keys = [_model_key(spec["prefix"], s["feature_mode"]) for s in steps]

        rows = []
        for seed in args.seeds:
            row = {"seed": seed, "epochs": args.epochs}
            for step in steps:
                model_key = _model_key(spec["prefix"], step["feature_mode"])
                summary = _run_one(
                    project_root, backbone, step, seed, args, out_root, args.evidence_adapter_dim
                )
                row[model_key] = _pack(summary)
                print(
                    f"{backbone} seed={seed} {step['level']} "
                    f"AUC={row[model_key]['test']['auc']:.4f}"
                )
            rows.append(row)

        seed_tag = "_".join(str(s) for s in args.seeds)
        prefix = spec["prefix"]
        results_file = out_root / f"{prefix}_ablation_seeds_{seed_tag}.json"
        summary_file = out_root / f"{prefix}_ablation_seeds_{seed_tag}_summary.json"
        results_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        summary = _summarize(rows, model_keys)
        summary["pipeline_id"] = PIPELINE_ID
        summary["ablation_plan"] = "A"
        summary["logs_path"] = str(_resolve_logs_path(project_root))
        summary["epochs"] = args.epochs
        summary["evidence_adapter_dim"] = args.evidence_adapter_dim
        summary["levels"] = [s["level"] for s in steps]
        summary["feature_modes"] = {s["level"]: normalize_feature_mode(s["feature_mode"]) for s in steps}
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{backbone}] Saved results: {results_file}")
        print(f"[{backbone}] Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
