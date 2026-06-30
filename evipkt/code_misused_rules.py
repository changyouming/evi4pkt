"""Rule-based misused/missing KC resolution from enriched framework log records (no LLM)."""
from __future__ import annotations

import re
from typing import Any, Sequence

from .kc_catalog import DEFAULT_KC_CATALOG

CODE_MISUSED_RULES_SOURCE = "record_enriched_block"
# Alias for plugplay_evidence imports in the reproduction bundle.
CODE_MISUSED_LLM_SOURCE = CODE_MISUSED_RULES_SOURCE
MISUSED_VECTOR_DIM = len(DEFAULT_KC_CATALOG)


def _names_from_vec(values: Sequence[float] | None, catalog: Sequence[str]) -> list[str]:
    if values is None:
        return []
    return [catalog[i] for i, v in enumerate(values[: len(catalog)]) if float(v) > 0]


def _vec_from_names(names: Sequence[str], catalog: Sequence[str]) -> list[float]:
    selected = set(names)
    return [1.0 if name in selected else 0.0 for name in catalog]


def _task_required(record: dict, catalog: Sequence[str]) -> tuple[list[float], list[str]]:
    pt = record.get("programming_task") or {}
    q_row = list(pt.get("q_kc") or [0.0] * len(catalog))
    if len(q_row) < len(catalog):
        q_row = q_row + [0.0] * (len(catalog) - len(q_row))
    return q_row, _names_from_vec(q_row, catalog)


def _student_code(record: dict) -> str:
    sc = record.get("student_code") or {}
    return str(sc.get("code") or "")


def has_disjunction_range_bug(code: str) -> bool:
    return bool(re.search(r"\|\|", code)) and bool(re.search(r"[<>]=?", code))


def apply_deterministic_corrections(
    code: str,
    missing_names: Sequence[str],
    misused_names: Sequence[str],
    *,
    required_names: Sequence[str],
) -> tuple[list[str], list[str]]:
    missing = set(missing_names)
    misused = set(misused_names)
    required = set(required_names)
    if has_disjunction_range_bug(code) and "LogicAndNotOr" in required:
        misused.add("LogicAndNotOr")
        missing.discard("LogicAndNotOr")
        missing.discard("LogicCompareNum")
        misused.discard("IfElse")
    return sorted(missing & required), sorted(misused & required)


def resolve_code_misused_kc(
    record: dict,
    *,
    catalog: Sequence[str] | None = None,
) -> tuple[list[str], list[str], list[float], list[float]]:
    """Resolved missing/misused KC from pre-enriched record blocks (no LLM calls)."""
    catalog = list(catalog or (record.get("programming_task") or {}).get("kc_catalog") or DEFAULT_KC_CATALOG)
    q_row, required = _task_required(record, catalog)
    del q_row  # unused; kept for parity with enrichment validators
    ce = record.get("code_evidence") or {}
    block = ce.get("misused_v8") or ce.get("misused") or {}
    if not isinstance(block, dict) or block.get("eligible") is False:
        zeros = [0.0] * len(catalog)
        return [], [], zeros, zeros

    missing_names = list(block.get("missing_kc") or [])
    misused_names = list(block.get("misused_kc") or [])
    if not missing_names and block.get("missing_kc_vec"):
        missing_names = _names_from_vec(block.get("missing_kc_vec"), catalog)
    if not misused_names and block.get("misused_kc_vec"):
        misused_names = _names_from_vec(block.get("misused_kc_vec"), catalog)
    elif not misused_names and block.get("vector"):
        misused_names = _names_from_vec(block.get("vector"), catalog)

    missing_names, misused_names = apply_deterministic_corrections(
        _student_code(record),
        missing_names,
        misused_names,
        required_names=required,
    )
    missing_vec = _vec_from_names(missing_names, catalog)
    misused_vec = _vec_from_names(misused_names, catalog)
    return missing_names, misused_names, missing_vec, misused_vec


def misused_vector(record: dict, *, catalog: Sequence[str] | None = None) -> list[float]:
    catalog = list(catalog or DEFAULT_KC_CATALOG)
    _, _, _, misused_vec = resolve_code_misused_kc(record, catalog=catalog)
    return misused_vec
