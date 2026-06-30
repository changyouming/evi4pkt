#!/usr/bin/env python3
"""Run four classic PKT baselines on F19 and S19 (10 seeds each)."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evipkt.iice_lite_dataset import IICELiteConfig
from evipkt.iice_lite_runner import run_iice_lite
from evipkt.kcgen_kt_lite_runner import run_kcgen_kt_lite
from evipkt.runner import DKTConfig, run_dkt

METRICS = ("auc", "acc", "f1", "loss")

COHORTS = {
    "f19": {
        "logs_path": "data/processed/framework_logs_first.jsonl",
        "code2vec_cache": "data/processed/code2vec_cache_f19.jsonl",
        "codebert_cache": "data/processed/codebert_cache_f19.jsonl",
        "q_matrix_path": "data/metadata/q_matrix.csv",
    },
    "s19": {
        "logs_path": "data/processed_s19/framework_logs_first.jsonl",
        "code2vec_cache": "data/processed_s19/code2vec_cache_f19.jsonl",
        "codebert_cache": "data/processed_s19/codebert_cache_f19.jsonl",
        "q_matrix_path": "data/processed_s19/q_matrix.csv",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohorts", type=str, nargs="+", default=["f19", "s19"], choices=list(COHORTS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", type=str, default="runs/classic_pkt_f19_s19")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["code_dkt", "pdkt_lite", "dkt_codebert", "kcgen_kt_lite"],
        choices=["code_dkt", "pdkt_lite", "dkt_codebert", "kcgen_kt_lite"],
    )
    return p.parse_args()


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _paired_t(diff: list[float]) -> dict:
    n = len(diff)
    mean = statistics.mean(diff)
    std = statistics.stdev(diff) if n > 1 else 0.0
    t_stat = mean / (std / math.sqrt(n)) if n > 1 and std > 0 else None
    p_value = None
    if t_stat is not None:
        try:
            from scipy import stats

            p_value = float(stats.ttest_1samp(diff, 0.0).pvalue)
        except ImportError:
            pass
    return {"n": n, "mean_diff": mean, "std_diff": std, "t_stat": t_stat, "p_value": p_value}


def _run_code_dkt(root: Path, cohort: str, cfg: dict, args, seed: int) -> dict:
    out_dir = root / args.out_dir / cohort / f"seed_{seed}" / "code_dkt"
    result_file = out_dir / f"result_seed{seed}.json"
    if args.resume and result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    summary = run_dkt(
        DKTConfig(
            logs_path=str((root / cfg["logs_path"]).resolve()),
            q_matrix_path=str((root / cfg["q_matrix_path"]).resolve()),
            feature_mode="problem_onehot_code2vec",
            model_name="code_dkt",
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            code2vec_cache_path=str((root / cfg["code2vec_cache"]).resolve()),
            out_dir=str(out_dir.resolve()),
        )
    )
    return summary


def _run_dkt_codebert(root: Path, cohort: str, cfg: dict, args, seed: int) -> dict:
    out_dir = root / args.out_dir / cohort / f"seed_{seed}" / "dkt_codebert"
    result_file = out_dir / f"result_seed{seed}.json"
    if args.resume and result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    summary = run_dkt(
        DKTConfig(
            logs_path=str((root / cfg["logs_path"]).resolve()),
            q_matrix_path=str((root / cfg["q_matrix_path"]).resolve()),
            feature_mode="problem_onehot_codebert",
            model_name="dkt_codebert",
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            codebert_cache_path=str((root / cfg["codebert_cache"]).resolve()),
            out_dir=str(out_dir.resolve()),
        )
    )
    return summary


def _run_pdkt_lite(root: Path, cohort: str, cfg: dict, args, seed: int) -> dict:
    out_dir = root / args.out_dir / cohort / f"seed_{seed}" / "pdkt_lite"
    result_file = out_dir / f"result_seed{seed}.json"
    if args.resume and result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    summary = run_iice_lite(
        IICELiteConfig(
            logs_path=str((root / cfg["logs_path"]).resolve()),
            q_matrix_path=str((root / cfg["q_matrix_path"]).resolve()),
            codebert_cache_path=str((root / cfg["codebert_cache"]).resolve()),
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=str(out_dir.resolve()),
            model_name="pdkt_lite",
        )
    )
    return summary


def _run_kcgen(root: Path, cohort: str, cfg: dict, args, seed: int) -> dict:
    out_dir = root / args.out_dir / cohort / f"seed_{seed}" / "kcgen_kt_lite"
    result_file = out_dir / f"result_seed{seed}.json"
    if args.resume and result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    summary = run_kcgen_kt_lite(
        IICELiteConfig(
            logs_path=str((root / cfg["logs_path"]).resolve()),
            q_matrix_path=str((root / cfg["q_matrix_path"]).resolve()),
            codebert_cache_path=str((root / cfg["codebert_cache"]).resolve()),
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=str(out_dir.resolve()),
        )
    )
    return summary


RUNNERS = {
    "code_dkt": _run_code_dkt,
    "pdkt_lite": _run_pdkt_lite,
    "dkt_codebert": _run_dkt_codebert,
    "kcgen_kt_lite": _run_kcgen,
}


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT
    all_results: dict = {
        "protocol": "CSEDM first-attempt, 50 problems, student 80/10/10, 10 seeds",
        "methods": {
            "code_dkt": "Code-DKT (problem_onehot_code2vec + LSTM)",
            "pdkt_lite": "PDKT-lite (Q-matrix + CodeBERT dual GRU + decay attention)",
            "dkt_codebert": "DKT+CodeBERT (BePKT DKTP+PLCodeBERT line)",
            "kcgen_kt_lite": "KCGen-KT-lite (human 18-KC mastery + CodeBERT, correctness-only)",
        },
        "cohorts": {},
    }

    for cohort in args.cohorts:
        cfg = COHORTS[cohort]
        cohort_out: dict = {"seeds": args.seeds, "runs": {}}
        for method in args.methods:
            runs = []
            for seed in args.seeds:
                print(f"[{cohort}] {method} seed={seed} ...", flush=True)
                runs.append(RUNNERS[method](root, cohort, cfg, args, seed))
            cohort_out["runs"][method] = runs
            cohort_out.setdefault("summary", {})[method] = {
                "metrics": {
                    m: _mean_std([float(r["test_metrics"][m]) for r in runs]) for m in METRICS
                }
            }
        all_results["cohorts"][cohort] = cohort_out

    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    tag = "_".join(map(str, args.seeds))
    out_path = out_root / f"comparison_{'_'.join(args.cohorts)}_seeds_{tag}.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
