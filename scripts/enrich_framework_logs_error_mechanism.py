#!/usr/bin/env python3
"""Attach v8 non-KC compile-error mechanism evidence (M1–M12, rule-based)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evipkt.error_mechanism import (
    ERROR_MECHANISM_SCHEMA_VERSION,
    attach_mechanism_to_record,
    is_compile_error,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-path",
        type=str,
        default="data/processed/framework_logs_first_process.jsonl",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default="",
        help="Default: <in_stem>_mechanism_v8.jsonl",
    )
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def main() -> None:
    args = parse_args()
    in_path = _resolve(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = _resolve(args.out_path) if args.out_path else in_path.with_name(
        in_path.stem + "_mechanism_v8.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_compile = 0
    primary_counts: Counter[str] = Counter()
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            n_in += 1
            if args.limit and n_in > args.limit:
                break

            enriched = attach_mechanism_to_record(record)
            mech = (enriched.get("error_evidence") or {}).get("mechanism_v8") or {}
            if is_compile_error(enriched):
                n_compile += 1
                pid = mech.get("primary_mechanism_id")
                if pid:
                    primary_counts[str(pid)] += 1

            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            n_out += 1

    print("=== v8 compile-error mechanism enrichment ===")
    print(f"Input:          {in_path} ({n_in} records)")
    print(f"Output:         {out_path} ({n_out} records)")
    print(f"Schema:         {ERROR_MECHANISM_SCHEMA_VERSION}")
    print(f"Compile errors: {n_compile}")
    if primary_counts:
        print("Primary mechanism distribution:")
        for mid in sorted(primary_counts, key=lambda x: (x != "M12", x)):
            c = primary_counts[mid]
            pct = 100.0 * c / max(n_compile, 1)
            print(f"  {mid}: {c} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
