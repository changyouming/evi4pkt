#!/usr/bin/env python3
"""Export student train/valid/test split JSON files for F19 and S19 (seeds 0–9)."""
from __future__ import annotations

import argparse
from pathlib import Path

from evipkt.dataset import load_framework_logs, save_student_split, split_students

DEFAULT_F19_LOGS = "data/processed/framework_logs_first_process_mechanism.jsonl"
DEFAULT_S19_LOGS = "data/processed_s19/framework_logs_first_process_mechanism.jsonl"


def _students_from_logs(logs_path: Path) -> list[str]:
    records = load_framework_logs(logs_path)
    return sorted({str(r["subject_id"]) for r in records})


def export_cohort(
    *,
    cohort: str,
    logs_path: Path,
    split_dir: Path,
    seeds: range,
    source_logs: str,
) -> None:
    students = _students_from_logs(logs_path)
    for seed in seeds:
        split = split_students(students, seed=seed)
        out = save_student_split(
            split,
            cohort=cohort,
            seed=seed,
            split_dir=split_dir,
            source_logs=source_logs,
        )
        print(
            f"{cohort} seed={seed}: "
            f"train={len(split.train_students)} "
            f"valid={len(split.valid_students)} "
            f"test={len(split.test_students)} -> {out}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--f19-logs", type=Path, default=Path(DEFAULT_F19_LOGS))
    parser.add_argument("--s19-logs", type=Path, default=Path(DEFAULT_S19_LOGS))
    parser.add_argument("--cohorts", choices=("f19", "s19", "both"), default="both")
    args = parser.parse_args()
    seeds = range(min(args.seeds), max(args.seeds) + 1)

    if args.cohorts in ("f19", "both"):
        if not args.f19_logs.is_file():
            raise SystemExit(f"F19 logs not found: {args.f19_logs}")
        export_cohort(
            cohort="f19",
            logs_path=args.f19_logs,
            split_dir=args.split_dir,
            seeds=seeds,
            source_logs=args.f19_logs.as_posix(),
        )

    if args.cohorts in ("s19", "both"):
        if not args.s19_logs.is_file():
            raise SystemExit(f"S19 logs not found: {args.s19_logs}")
        export_cohort(
            cohort="s19",
            logs_path=args.s19_logs,
            split_dir=args.split_dir,
            seeds=seeds,
            source_logs=args.s19_logs.as_posix(),
        )


if __name__ == "__main__":
    main()
