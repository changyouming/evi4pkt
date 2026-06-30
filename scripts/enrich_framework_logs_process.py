#!/usr/bin/env python3
"""Add KC-conditioned process evidence to framework logs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evipkt.process_evidence import attach_process_evidence_to_records


def parse_args():
    p = argparse.ArgumentParser(
        description="Add leakage-free KC-history process_evidence to framework JSONL."
    )
    p.add_argument(
        "--in-path",
        type=str,
        default="data/processed/framework_logs_first.jsonl",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default="",
        help="Output JSONL (default: <in>_process.jsonl next to input)",
    )
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def main():
    args = parse_args()
    in_path = _resolve(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_path = (
        _resolve(args.out_path)
        if args.out_path
        else in_path.with_name(in_path.stem + "_process.jsonl")
    )

    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                records.append(json.loads(text))

    enriched = attach_process_evidence_to_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in enriched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dims = sorted(
        {
            int((r.get("process_evidence") or {}).get("vector_dim", 0))
            for r in enriched
        }
    )
    print("=== KC process evidence enrichment ===")
    print(f"Input:  {in_path} ({len(records)} records)")
    print(f"Output: {out_path} ({len(enriched)} records)")
    print(f"Vector dims: {dims}")


if __name__ == "__main__":
    main()
