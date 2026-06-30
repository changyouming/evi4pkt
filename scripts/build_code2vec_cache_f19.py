#!/usr/bin/env python3
"""Build Code-DKT code2vec cache for CSEDM F19 first-attempt logs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evipkt.code2vec_features import (
    CODE2VEC_VECTOR_DIM,
    append_code2vec_cache_row,
    build_vocab_from_codes,
    code2vec_vector,
    code_cache_key,
)
from evipkt.dataset import load_framework_logs, resolve_student_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logs-path",
        type=Path,
        default=PROJECT_ROOT / "data/processed/framework_logs_first.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
    )
    p.add_argument("--vocab-seed", type=int, default=0, help="Split seed for train-only vocab.")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--max-paths", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logs_path = args.logs_path if args.logs_path.is_absolute() else PROJECT_ROOT / args.logs_path
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_framework_logs(logs_path)
    students = sorted({str(r["subject_id"]) for r in records})
    split = resolve_student_split(students, seed=args.vocab_seed, logs_path=logs_path)
    train_set = set(split.train_students)

    train_codes: list[str] = []
    all_codes: list[tuple[str, str]] = []
    for rec in records:
        code = str((rec.get("student_code") or {}).get("code") or "")
        key = code_cache_key(code)
        all_codes.append((key, code))
        if str(rec["subject_id"]) in train_set:
            train_codes.append(code)

    vocab_path = out_dir / "code2vec_vocab_f19.json"
    cache_path = out_dir / "code2vec_cache_f19.jsonl"
    summary_path = out_dir / "code2vec_cache_f19_summary.json"

    existing: set[str] = set()
    if args.resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    existing.add(str(json.loads(text)["cache_key"]))

    if vocab_path.exists() and args.resume:
        from evipkt.code2vec_features import Code2VecVocab

        vocab = Code2VecVocab.load(vocab_path)
    else:
        vocab = build_vocab_from_codes(
            train_codes,
            vocab_size=args.vocab_size,
            max_paths=args.max_paths,
            seed=args.vocab_seed,
        )
        vocab.save(vocab_path)

    unique_codes = {key: code for key, code in all_codes}
    n_new = 0
    n_empty = 0
    for key, code in unique_codes.items():
        if key in existing:
            continue
        if not code.strip():
            vec = [0.0] * CODE2VEC_VECTOR_DIM
            n_empty += 1
        else:
            vec = code2vec_vector(code, vocab, max_paths=args.max_paths)
        append_code2vec_cache_row(cache_path, key, vec)
        n_new += 1

    summary = {
        "logs_path": str(logs_path.resolve()),
        "cache_path": str(cache_path.resolve()),
        "vocab_path": str(vocab_path.resolve()),
        "vocab_seed": args.vocab_seed,
        "train_students": len(split.train_students),
        "unique_submissions": len(unique_codes),
        "vocab_size": len(vocab.path_to_index),
        "vector_dim": CODE2VEC_VECTOR_DIM,
        "max_paths": args.max_paths,
        "new_vectors_written": n_new,
        "empty_code_vectors": n_empty,
        "resume": args.resume,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
