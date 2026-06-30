#!/usr/bin/env python3
"""Build offline CodeBERT cache for F19 first-attempt submissions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evipkt.codebert_features import (
    CODEBERT_VECTOR_DIM,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    append_codebert_cache_row,
    code_cache_key,
    embed_codes_batch,
)
from evipkt.dataset import load_framework_logs


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
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Debug: cap unique codes (0=all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logs_path = args.logs_path if args.logs_path.is_absolute() else PROJECT_ROOT / args.logs_path
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "codebert_cache_f19.jsonl"
    summary_path = out_dir / "codebert_cache_f19_summary.json"

    records = load_framework_logs(logs_path)
    unique: dict[str, str] = {}
    for rec in records:
        code = str((rec.get("student_code") or {}).get("code") or "")
        unique[code_cache_key(code)] = code

    existing: set[str] = set()
    if args.resume and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    existing.add(str(json.loads(text)["cache_key"]))

    pending = [(k, c) for k, c in unique.items() if k not in existing]
    if args.limit > 0:
        pending = pending[: args.limit]

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    n_new = 0
    for i in range(0, len(pending), args.batch_size):
        chunk = pending[i : i + args.batch_size]
        vectors = embed_codes_batch(
            [code for _, code in chunk],
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            device=device,
            tokenizer=tokenizer,
            model=model,
        )
        for (key, _code), vec in zip(chunk, vectors):
            append_codebert_cache_row(cache_path, key, vec)
            n_new += 1
        print(f"embedded {min(i + args.batch_size, len(pending))}/{len(pending)}")

    summary = {
        "logs_path": str(logs_path.resolve()),
        "cache_path": str(cache_path.resolve()),
        "model_name": args.model_name,
        "max_tokens": args.max_tokens,
        "vector_dim": CODEBERT_VECTOR_DIM,
        "unique_submissions": len(unique),
        "new_vectors_written": n_new,
        "resume": args.resume,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
