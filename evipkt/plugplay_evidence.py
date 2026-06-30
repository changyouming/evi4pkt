"""Plug-and-play evidence vectors for KT: Code misused/missing + compile mechanism (non-KC)."""
from __future__ import annotations

from typing import Sequence

from .code_misused_rules import (
    CODE_MISUSED_LLM_SOURCE,
    MISUSED_VECTOR_DIM,
    misused_vector,
    resolve_code_misused_kc,
)
from .error_mechanism import (
    ERROR_MECHANISM_SOURCE,
    MECHANISM_VECTOR_DIM,
    mechanism_vector,
)
from .feature_modes import CANONICAL_PLUGPLAY_MODES, normalize_feature_mode
from .kc_catalog import DEFAULT_KC_CATALOG

# History interaction layout: [missing_kc (18), misused_kc (18)].
CODE_EVIDENCE_DIM = 2 * MISUSED_VECTOR_DIM
# [M1–M12 one-hot (12), non_kc_error flag (1)].
MECHANISM_EVIDENCE_DIM = MECHANISM_VECTOR_DIM + 1

# Backward-compatible aliases (removed in new code; kept for imports/tests).
CODE_V8_VECTOR_DIM = CODE_EVIDENCE_DIM
MECHANISM_V8_VECTOR_DIM = MECHANISM_EVIDENCE_DIM

MODES_USING_CODE: frozenset[str] = frozenset(
    {
        "problem_plus_q_code",
        "problem_plus_q_code_mechanism",
        "problem_plus_q_process_code",
        "problem_plus_q_process_code_mechanism",
        "problem_plus_q_process_code_mechanism_code2vec",
    }
)
MODES_USING_MECHANISM: frozenset[str] = frozenset(
    {
        "problem_plus_q_mechanism",
        "problem_plus_q_code_mechanism",
        "problem_plus_q_process_code_mechanism",
        "problem_plus_q_process_code_mechanism_code2vec",
    }
)
MODES_USING_PROCESS_FULL: frozenset[str] = frozenset(
    {
        "problem_plus_q_process_code",
        "problem_plus_q_process_code_mechanism",
        "problem_plus_q_process_code_mechanism_code2vec",
    }
)

# Deprecated names — import only.
MODES_USING_CODE_V8 = MODES_USING_CODE
MODES_USING_MECHANISM_V8 = MODES_USING_MECHANISM
MODES_USING_PROCESS_V8 = MODES_USING_PROCESS_FULL
V8_FEATURE_MODES = CANONICAL_PLUGPLAY_MODES


def _pad(values: Sequence[float] | None, n: int) -> list[float]:
    out = [float(v) for v in (values or [])[:n]]
    if len(out) < n:
        out.extend([0.0] * (n - len(out)))
    return out


def _catalog_len(record: dict, catalog: Sequence[str] | None) -> int:
    if catalog is not None:
        return len(catalog)
    pt = record.get("programming_task") or {}
    cat = pt.get("kc_catalog") or DEFAULT_KC_CATALOG
    return len(cat)


def _misused_block(record: dict) -> dict | None:
    ce = record.get("code_evidence") or {}
    block = ce.get("misused")
    if isinstance(block, dict):
        return block
    block = ce.get("misused_v8")
    return block if isinstance(block, dict) else None


def _mechanism_block(record: dict) -> dict | None:
    ee = record.get("error_evidence") or {}
    block = ee.get("mechanism")
    if isinstance(block, dict):
        return block
    block = ee.get("mechanism_v8")
    return block if isinstance(block, dict) else None


def code_evidence_vector(record: dict, *, catalog: Sequence[str] | None = None) -> list[float]:
    """
    Code diagnostic vector: missing_kc || misused_kc (each len=catalog).
    Zeros when misused evidence is absent or ineligible.
    """
    n = _catalog_len(record, catalog)
    block = _misused_block(record)
    if not isinstance(block, dict):
        return [0.0] * CODE_EVIDENCE_DIM
    missing_names, misused_names, missing, misused = resolve_code_misused_kc(
        record, catalog=catalog
    )
    if not missing_names and not misused_names and not block.get("missing_kc_vec"):
        return [0.0] * CODE_EVIDENCE_DIM
    missing = _pad(missing, n)
    misused = _pad(misused, n)
    return missing + misused


def compile_mechanism_vector(record: dict) -> list[float]:
    """Compile mechanism one-hot + non_kc_error scalar; zeros when not enriched."""
    block = _mechanism_block(record)
    mech = mechanism_vector(record)
    non_kc = float(block.get("non_kc_error", 0.0) or 0.0) if isinstance(block, dict) else 0.0
    return mech + [non_kc]


# Backward-compatible aliases.
code_v8_vector = code_evidence_vector
mechanism_v8_vector = compile_mechanism_vector


def code_evidence_source_for_mode(feature_mode: str) -> str | None:
    return CODE_MISUSED_LLM_SOURCE if normalize_feature_mode(feature_mode) in MODES_USING_CODE else None


def mechanism_evidence_source_for_mode(feature_mode: str) -> str | None:
    return (
        ERROR_MECHANISM_SOURCE
        if normalize_feature_mode(feature_mode) in MODES_USING_MECHANISM
        else None
    )


code_v8_source_for_mode = code_evidence_source_for_mode
mechanism_v8_source_for_mode = mechanism_evidence_source_for_mode


def plugplay_history_evidence_dim(feature_mode: str, q_kc_dim: int = 18) -> int:
    mode = normalize_feature_mode(feature_mode)
    dim = 0
    if mode in MODES_USING_CODE:
        dim += CODE_EVIDENCE_DIM
    if mode in MODES_USING_MECHANISM:
        dim += MECHANISM_EVIDENCE_DIM
    if mode in MODES_USING_PROCESS_FULL:
        dim += 2 * q_kc_dim
    return dim


v8_history_evidence_dim = plugplay_history_evidence_dim


def plugplay_summary_fields(feature_mode: str) -> dict[str, int | str | None]:
    mode = normalize_feature_mode(feature_mode)
    return {
        "code_evidence_source": code_evidence_source_for_mode(mode),
        "code_evidence_dim": CODE_EVIDENCE_DIM if mode in MODES_USING_CODE else 0,
        "mechanism_evidence_source": mechanism_evidence_source_for_mode(mode),
        "mechanism_evidence_dim": MECHANISM_EVIDENCE_DIM if mode in MODES_USING_MECHANISM else 0,
    }


evidence_v8_summary_fields = plugplay_summary_fields
