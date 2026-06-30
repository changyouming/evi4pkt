from __future__ import annotations

from copy import deepcopy
from typing import Any

from .kc_catalog import DEFAULT_KC_CATALOG

EXPERIMENT_FILTER_VERSION = "v1_drop_full_pass_mappable"
ELIGIBLE_OUTCOMES = frozenset({"partial_pass", "test_fail"})


def _catalog(record: dict) -> list[str]:
    pt = record.get("programming_task") or {}
    return list(pt.get("kc_catalog") or DEFAULT_KC_CATALOG)


def _zero_kc_vector(catalog: list[str]) -> list[float]:
    return [0.0] * len(catalog)


def _outcome_type(record: dict) -> str:
    return str((record.get("code_issues") or {}).get("outcome_type") or "unknown")


def _llm_block(error_evidence: dict) -> dict:
    llm = error_evidence.get("llm")
    return llm if isinstance(llm, dict) else error_evidence


def decide_experiment_error_filter(
    record: dict,
    *,
    drop_full_pass_mappable: bool = True,
    mappable_outcomes_only: bool = True,
) -> dict[str, Any]:
    """
    Build experiment-safe error vectors from enriched error_evidence.

    Policy (experiment_v1):
    - Drop KC-level error vectors for full_pass + mappable (label/LLM conflict).
    - Only partial_pass / test_fail may carry non-zero kc_error when mappable.
    - review_required / unmapped always use zero kc_error; non_kc_error follows outcome.
    """
    block = dict(record.get("error_evidence") or {})
    llm = _llm_block(block)
    catalog = _catalog(record)
    zeros = _zero_kc_vector(catalog)

    raw_mappability = str(llm.get("mappability") or block.get("mappability") or "unknown")
    outcome = _outcome_type(record)
    raw_kc_error = list(llm.get("kc_error") or block.get("kc_error") or zeros)
    if len(raw_kc_error) < len(catalog):
        raw_kc_error = raw_kc_error + [0.0] * (len(catalog) - len(raw_kc_error))
    raw_mapped = list(llm.get("mapped_error_kcs") or llm.get("error_implicated_kc") or [])
    raw_non_kc = int(llm.get("non_kc_error", block.get("non_kc_error", 0)) or 0)

    eligible = False
    reason = "not_mappable"
    kc_error = zeros
    mapped_error_kcs: list[str] = []
    mappability = raw_mappability

    if raw_mappability == "mappable":
        if drop_full_pass_mappable and outcome == "full_pass":
            eligible = False
            reason = "dropped_full_pass_mappable"
            mappability = "review_required"
        elif mappable_outcomes_only and outcome not in ELIGIBLE_OUTCOMES:
            eligible = False
            reason = f"dropped_mappable_on_{outcome}"
            kc_error = zeros
            mapped_error_kcs = []
            mappability = "review_required"
        else:
            eligible = True
            reason = "kc_error_eligible"
            kc_error = raw_kc_error[: len(catalog)]
            mapped_error_kcs = list(raw_mapped)
    elif raw_mappability == "review_required":
        eligible = False
        reason = "review_required_no_kc_vector"
    elif raw_mappability == "unmapped":
        eligible = False
        reason = "unmapped_no_kc_vector"
    else:
        eligible = False
        reason = f"unknown_mappability_{raw_mappability}"

    non_kc_error = 0 if outcome == "full_pass" else 1

    return {
        "filter_version": EXPERIMENT_FILTER_VERSION,
        "eligible": eligible,
        "filter_reason": reason,
        "raw_mappability": raw_mappability,
        "mappability": mappability,
        "outcome_type": outcome,
        "mapped_error_kcs": mapped_error_kcs,
        "kc_error": kc_error,
        "non_kc_error": non_kc_error,
        "kc_error_dim": len(catalog),
    }


def apply_experiment_error_filter(
    record: dict,
    *,
    drop_full_pass_mappable: bool = True,
    mappable_outcomes_only: bool = True,
) -> dict:
    """Return a copy with error_evidence.llm preserved and experiment-safe top-level vectors."""
    out = deepcopy(record)
    block = dict(out.get("error_evidence") or {})
    llm = block.get("llm")
    if not isinstance(llm, dict):
        llm = {k: v for k, v in block.items() if k not in ("experiment", "filter")}

    experiment = decide_experiment_error_filter(
        out,
        drop_full_pass_mappable=drop_full_pass_mappable,
        mappable_outcomes_only=mappable_outcomes_only,
    )

    merged = {
        **{k: v for k, v in block.items() if k not in ("llm", "experiment")},
        "llm": deepcopy(llm) if isinstance(llm, dict) else llm,
        "experiment": experiment,
        "source": (llm or {}).get("source", block.get("source")),
        "prompt_version": (llm or {}).get("prompt_version", block.get("prompt_version")),
        "mappability": experiment["mappability"],
        "mapped_error_kcs": experiment["mapped_error_kcs"],
        "kc_error": experiment["kc_error"],
        "non_kc_error": experiment["non_kc_error"],
        "filter_version": experiment["filter_version"],
        "experiment_eligible": experiment["eligible"],
        "filter_reason": experiment["filter_reason"],
    }
    out["error_evidence"] = merged
    return out
