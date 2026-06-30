#!/usr/bin/env python3
"""Build framework-aligned CSEDM learning logs (framework.png stage 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evipkt.preprocess import PreprocessConfig, run_preprocess


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Preprocess CSEDM CodeWorkout into Evi4PKT framework logs. "
            "Each record contains programming_task, student_code, code_evidence, code_issues "
            "(with pkt_label), and trajectory metadata. Task Q from dataset expert KC; "
            "code evidence is rule-based (no LLM)."
        )
    )
    p.add_argument(
        "--csedm-root",
        type=str,
        default="data/F19_Release_All_05_23_22/All",
    )
    p.add_argument(
        "--prompts-csv",
        type=str,
        default="data/metadata/problem_prompts.csv",
    )
    p.add_argument(
        "--submission-mode",
        type=str,
        default="all",
        choices=["first", "all"],
    )
    p.add_argument(
        "--label-threshold",
        type=float,
        default=1.0,
        help="PKT correct if Score >= threshold (default 1.0 = all tests pass).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
    )
    return p.parse_args()


def main():
    args = parse_args()
    project_root = PROJECT_ROOT
    cfg = PreprocessConfig(
        csedm_root=(project_root / args.csedm_root).resolve(),
        prompts_csv=(project_root / args.prompts_csv).resolve(),
        submission_mode=args.submission_mode,  # type: ignore[arg-type]
        label_threshold=args.label_threshold,
    )
    out_dir = (project_root / args.out_dir).resolve()

    summary = run_preprocess(cfg, out_dir)
    print("=== Evi4PKT preprocess complete ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
