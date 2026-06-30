from __future__ import annotations

from typing import Sequence

from .error_evidence_filter import decide_experiment_error_filter
from .kc_catalog import DEFAULT_KC_CATALOG

ERROR_EVIDENCE_SOURCE = "rule_mechanism"
NON_KC_ERROR_DIM = 1

ERROR_FEATURE_MODES = frozenset(
    {
        "problem_plus_q_error",
        "problem_plus_q_code_llm_error",
        "problem_plus_q_process_error",
        "problem_plus_q_code_llm_process_error",
    }
)


def _catalog(record: dict) -> list[str]:
    pt = record.get("programming_task") or {}
    return list(pt.get("kc_catalog") or DEFAULT_KC_CATALOG)


def error_evidence_vector_dim(kc_dim: int) -> int:
    return int(kc_dim) + NON_KC_ERROR_DIM


def _pad_kc_error(values: Sequence[float] | None, n: int) -> list[float]:
    out = [float(v) for v in (values or [])[:n]]
    if len(out) < n:
        out.extend([0.0] * (n - len(out)))
    return out


def _has_experiment_vectors(block: dict) -> bool:
    return isinstance(block.get("kc_error"), list) and "filter_version" in block


def error_evidence_vector(
    record: dict,
    *,
    catalog: Sequence[str] | None = None,
    use_experiment_filter: bool = True,
) -> list[float]:
    """
    Fixed-size error vector for KT interaction encoding.

    Layout: [kc_error (len=catalog), non_kc_error (1)].
    Defaults to experiment-safe vectors (filtered full_pass mappable, etc.).
    """
    catalog = list(catalog or _catalog(record))
    n = len(catalog)
    block = record.get("error_evidence") or {}
    if not block:
        return [0.0] * error_evidence_vector_dim(n)

    if use_experiment_filter:
        if _has_experiment_vectors(block):
            kc_error = _pad_kc_error(block.get("kc_error"), n)
            non_kc_error = float(block.get("non_kc_error", 0) or 0)
        else:
            experiment = decide_experiment_error_filter(record)
            kc_error = _pad_kc_error(experiment.get("kc_error"), n)
            non_kc_error = float(experiment.get("non_kc_error", 0) or 0)
    else:
        llm = block.get("llm")
        source = llm if isinstance(llm, dict) else block
        kc_error = _pad_kc_error(source.get("kc_error"), n)
        non_kc_error = float(source.get("non_kc_error", 0) or 0)

    return kc_error + [non_kc_error]


def error_evidence_source_for_mode(feature_mode: str) -> str | None:
    return ERROR_EVIDENCE_SOURCE if feature_mode in ERROR_FEATURE_MODES else None
