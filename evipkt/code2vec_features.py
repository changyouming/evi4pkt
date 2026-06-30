"""Code2vec-style AST path features for Code-DKT baseline on CSEDM F19."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

try:
    import javalang
except ImportError:  # pragma: no cover
    javalang = None  # type: ignore[assignment]

CODE2VEC_VECTOR_DIM = 128
DEFAULT_MAX_PATHS = 50
DEFAULT_MAX_PATH_WIDTH = 8
DEFAULT_VOCAB_SIZE = 8000


def code_cache_key(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8", errors="replace")).hexdigest()


_LEAF_NODE_TYPES: tuple[type, ...] = ()
if javalang is not None:
    _LEAF_NODE_TYPES = (javalang.tree.Literal, javalang.tree.MemberReference)


def _node_type_name(node: object) -> str:
    return type(node).__name__


def _leaf_label(node: object) -> str:
    if javalang is None:
        return "Unknown"
    if isinstance(node, javalang.tree.Literal):
        return "Literal"
    if isinstance(node, javalang.tree.MemberReference):
        return "Identifier"
    return _node_type_name(node)


def _is_leaf_node(node: object) -> bool:
    return isinstance(node, _LEAF_NODE_TYPES)


def _skip_leaf(node: object) -> bool:
    if isinstance(node, javalang.tree.MemberReference):
        member = getattr(node, "member", None) or getattr(node, "qualifier", None)
        return member in {"__Code2VecWrapper__", "__m__"}
    return False


def _attach_parents(root: object) -> None:
    stack: list[object] = [root]
    while stack:
        node = stack.pop()
        for child in getattr(node, "children", []) or []:
            if child is None:
                continue
            if isinstance(child, list):
                for item in child:
                    if item is not None and hasattr(item, "children"):
                        item.parent = node  # type: ignore[attr-defined]
                        stack.append(item)
                continue
            if isinstance(child, (str, int, float, bool, set, tuple, dict)):
                continue
            if hasattr(child, "children"):
                child.parent = node  # type: ignore[attr-defined]
                stack.append(child)


def _path_to_root(node: object) -> list[object]:
    path: list[object] = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = getattr(cur, "parent", None)
    return path


def _path_context(leaf_a: object, leaf_b: object, max_width: int) -> str | None:
    up_a = _path_to_root(leaf_a)
    up_b = _path_to_root(leaf_b)
    index_b = {id(node): idx for idx, node in enumerate(up_b)}

    lca_idx_a = None
    lca_idx_b = None
    for idx_a, node in enumerate(up_a):
        idx_b = index_b.get(id(node))
        if idx_b is not None:
            lca_idx_a = idx_a
            lca_idx_b = idx_b
            break
    if lca_idx_a is None or lca_idx_b is None:
        return None

    down_from_lca = [_node_type_name(n) for n in reversed(up_a[lca_idx_a + 1 :])]
    up_to_leaf_b = [_node_type_name(n) for n in up_b[:lca_idx_b][::-1]]
    internal = (down_from_lca + up_to_leaf_b)[:max_width]
    return f"{_leaf_label(leaf_a)},{','.join(internal)},{_leaf_label(leaf_b)}"


def _wrap_java_snippet(code: str) -> str:
    """Wrap a CodeWorkout method snippet so javalang can parse it."""
    stripped = (code or "").strip()
    if not stripped:
        return "class __Code2VecWrapper__ { void __m__() {} }"
    if stripped.startswith("class ") or stripped.startswith("public class "):
        return stripped
    return f"class __Code2VecWrapper__ {{\n{stripped}\n}}"


def extract_ast_path_strings(
    code: str,
    *,
    max_paths: int = DEFAULT_MAX_PATHS,
    max_path_width: int = DEFAULT_MAX_PATH_WIDTH,
) -> list[str]:
    """Extract code2vec-style path contexts from Java source."""
    if not (code or "").strip():
        return []
    if javalang is None:
        raise ImportError("javalang is required for code2vec features (pip install javalang).")
    wrapped = _wrap_java_snippet(code)
    try:
        tree = javalang.parse.parse(wrapped)
    except (
        javalang.parser.JavaSyntaxError,
        javalang.tokenizer.LexerError,
        RecursionError,
        ValueError,
        TypeError,
    ):
        return []

    _attach_parents(tree)

    leaves: list[object] = []
    seen: set[int] = set()
    for _path, node in tree:
        if not _is_leaf_node(node):
            continue
        if _skip_leaf(node):
            continue
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        leaves.append(node)

    if len(leaves) < 2:
        return []

    paths: list[str] = []
    seen_paths: set[str] = set()
    for i, leaf_a in enumerate(leaves):
        for leaf_b in leaves[i + 1 :]:
            ctx = _path_context(leaf_a, leaf_b, max_path_width)
            if ctx is None or ctx in seen_paths:
                continue
            seen_paths.add(ctx)
            paths.append(ctx)
            if len(paths) >= max_paths:
                return paths
    return paths


@dataclass
class Code2VecVocab:
    path_to_index: dict[str, int]
    projection: np.ndarray  # [vocab_size, CODE2VEC_VECTOR_DIM]
    dim: int = CODE2VEC_VECTOR_DIM

    def vectorize(self, path_strings: Sequence[str]) -> list[float]:
        if not path_strings:
            return [0.0] * self.dim
        counts = np.zeros(len(self.path_to_index), dtype=np.float32)
        for path in path_strings:
            idx = self.path_to_index.get(path)
            if idx is not None:
                counts[idx] += 1.0
        if counts.sum() == 0:
            return [0.0] * self.dim
        counts /= counts.sum()
        vec = counts @ self.projection
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32).tolist()

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "path_to_index": self.path_to_index,
            "projection": self.projection.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Code2VecVocab:
        return cls(
            path_to_index={str(k): int(v) for k, v in payload["path_to_index"].items()},
            projection=np.asarray(payload["projection"], dtype=np.float32),
            dim=int(payload.get("dim", CODE2VEC_VECTOR_DIM)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Code2VecVocab:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_vocab_from_codes(
    codes: Iterable[str],
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_paths: int = DEFAULT_MAX_PATHS,
    seed: int = 0,
) -> Code2VecVocab:
    """Build path vocabulary from training-split Java submissions only."""
    freq: dict[str, int] = {}
    for code in codes:
        for path in extract_ast_path_strings(code, max_paths=max_paths):
            freq[path] = freq.get(path, 0) + 1

    if not freq:
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((1, CODE2VEC_VECTOR_DIM)).astype(np.float32)
        return Code2VecVocab(path_to_index={"<unk>": 0}, projection=projection)

    top_paths = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:vocab_size]
    path_to_index = {path: idx for idx, (path, _count) in enumerate(top_paths)}

    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((len(path_to_index), CODE2VEC_VECTOR_DIM)).astype(
        np.float32
    )
    projection /= np.sqrt(CODE2VEC_VECTOR_DIM)
    return Code2VecVocab(path_to_index=path_to_index, projection=projection)


def code2vec_vector(code: str, vocab: Code2VecVocab, *, max_paths: int = DEFAULT_MAX_PATHS) -> list[float]:
    paths = extract_ast_path_strings(code, max_paths=max_paths)
    return vocab.vectorize(paths)


def iter_code2vec_cache_rows(cache_path: Path) -> Iterator[tuple[str, list[float]]]:
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            yield str(row["cache_key"]), list(row["vector"])


def load_code2vec_cache(cache_path: Path) -> dict[str, list[float]]:
    if not cache_path.exists():
        return {}
    return dict(iter_code2vec_cache_rows(cache_path))


def append_code2vec_cache_row(cache_path: Path, cache_key: str, vector: Sequence[float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"cache_key": cache_key, "vector": list(vector)}, ensure_ascii=False) + "\n")
