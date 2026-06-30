"""Offline CodeBERT embeddings for IICE-lite baseline on CSEDM F19."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

CODEBERT_VECTOR_DIM = 768
DEFAULT_MODEL_NAME = "microsoft/codebert-base"
DEFAULT_MAX_TOKENS = 256


def code_cache_key(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8", errors="replace")).hexdigest()


def embed_codes_batch(
    codes: Sequence[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    device: str | None = None,
    tokenizer=None,
    model=None,
) -> list[list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)

    texts = [(c or "").strip() for c in codes]
    out = [[0.0] * CODEBERT_VECTOR_DIM for _ in texts]
    nonempty_idx = [i for i, t in enumerate(texts) if t]
    if not nonempty_idx:
        return out

    with torch.no_grad():
        batch_texts = [texts[i] for i in nonempty_idx]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = model(**inputs).last_hidden_state[:, 0, :].detach().cpu().float()
        for j, i in enumerate(nonempty_idx):
            out[i] = hidden[j].tolist()
        return out


def iter_codebert_cache_rows(cache_path: Path) -> Iterator[tuple[str, list[float]]]:
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            yield str(row["cache_key"]), list(row["vector"])


def load_codebert_cache(cache_path: Path) -> dict[str, list[float]]:
    if not cache_path.exists():
        return {}
    return dict(iter_codebert_cache_rows(cache_path))


def append_codebert_cache_row(cache_path: Path, cache_key: str, vector: Sequence[float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"cache_key": cache_key, "vector": list(vector)}, ensure_ascii=False) + "\n")
