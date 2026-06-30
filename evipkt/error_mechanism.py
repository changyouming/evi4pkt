"""Rule-based non-KC compile-error mechanism taxonomy (v8 M1–M12)."""
from __future__ import annotations

import re
from typing import Any, Sequence

ERROR_MECHANISM_SOURCE = "rule_compile_message"
ERROR_MECHANISM_SCHEMA_VERSION = "v8_non_kc_mechanism"

# Closed novice compile-error mechanism catalog (non-KC; not aligned to Q-matrix KCs).
NON_KC_MECHANISM_CATALOG: list[tuple[str, str]] = [
    ("M1", "semicolon_missing"),
    ("M2", "delimiter_bracket_mismatch"),
    ("M3", "undefined_identifier"),
    ("M4", "missing_return_path"),
    ("M5", "type_mismatch"),
    ("M6", "control_flow_mismatch"),
    ("M7", "invalid_expression"),
    ("M8", "literal_quote_error"),
    ("M9", "operator_misuse"),
    ("M10", "method_signature_mismatch"),
    ("M11", "scope_or_duplicate"),
    ("M12", "unclassified_compile"),
]

MECHANISM_ID_BY_NAME = {name: mid for mid, name in NON_KC_MECHANISM_CATALOG}
MECHANISM_NAME_BY_ID = {mid: name for mid, name in NON_KC_MECHANISM_CATALOG}
MECHANISM_VECTOR_DIM = len(NON_KC_MECHANISM_CATALOG)

# Priority order: more specific patterns before general delimiter / catch-all.
_MECHANISM_RULES: list[tuple[str, str, str]] = [
    (
        "M6",
        "control_flow_mismatch",
        r"'else'\s+without\s+'if'|else without if|bad initializer for for-loop",
    ),
    (
        "M10",
        "method_signature_mismatch",
        r"cannot be applied to given types|method .+ cannot be applied",
    ),
    (
        "M11",
        "scope_or_duplicate",
        (
            r"already defined in|cannot be referenced from a static context"
            r"|variable .+ might not have been initialized"
            r"|cannot assign a value to final"
        ),
    ),
    ("M4", "missing_return_path", r"missing return statement"),
    ("M3", "undefined_identifier", r"cannot find symbol"),
    (
        "M9",
        "operator_misuse",
        r"bad operand types? for (?:binary|unary) operator",
    ),
    (
        "M5",
        "type_mismatch",
        r"incompatible types|incomparable types|possible lossy conversion|unexpected type",
    ),
    (
        "M8",
        "literal_quote_error",
        (
            r"illegal character|illegal line end in character literal"
            r"|unclosed character literal|unclosed string literal|empty character literal"
        ),
    ),
    ("M1", "semicolon_missing", r"';'\s*expected|empty statement after if"),
    (
        "M2",
        "delimiter_bracket_mismatch",
        (
            r"class, interface, or enum expected|class expected"
            r"|'\)'\s*expected|'\('\s*expected|'\{'\s*expected|'\}'\s*expected"
            r"|'\['\s*expected|'\]'\s*expected"
            r"|\)\s*expected|\(\s*expected|\{\s*expected|\}\s*expected"
            r"|\]\s*expected|\[\s*expected"
            r"|expected[^\n]*[\(\{\}\[\)\]]|\(\s*or\s*'\['\s*expected"
            r"|illegal start|reached end of file|return outside method"
        ),
    ),
    (
        "M7",
        "invalid_expression",
        (
            r"not a statement|<identifier>\s*expected|unreachable statement"
            r"|char cannot be dereferenced|int cannot be dereferenced"
            r"|array dimension missing|\.class expected"
        ),
    ),
]


def mechanism_vector_dim() -> int:
    return MECHANISM_VECTOR_DIM


def is_compile_error(record: dict) -> bool:
    ci = record.get("code_issues") or {}
    if ci.get("has_compile_error"):
        return True
    return str(ci.get("outcome_type") or "") == "compile_error"


def compile_messages(record: dict) -> list[str]:
    ci = record.get("code_issues") or {}
    out: list[str] = []
    for item in ci.get("issues") or []:
        if str(item.get("type") or "") != "compile_error":
            continue
        msg = str(item.get("message") or "").strip()
        if msg:
            out.append(msg)
    return out


def classify_compile_message(message: str) -> tuple[str, str]:
    text = message or ""
    for mechanism_id, mechanism_name, pattern in _MECHANISM_RULES:
        if re.search(pattern, text, re.I):
            return mechanism_id, mechanism_name
    return "M12", "unclassified_compile"


def classify_compile_messages(messages: Sequence[str]) -> dict[str, Any]:
    cleaned = [str(m).strip() for m in messages if str(m).strip()]
    if not cleaned:
        primary_id, primary_name = "M12", "unclassified_compile"
    else:
        primary_id, primary_name = classify_compile_message(cleaned[0])

    secondary: list[dict[str, str]] = []
    seen = {primary_id}
    for msg in cleaned[1:]:
        mid, name = classify_compile_message(msg)
        if mid in seen:
            continue
        seen.add(mid)
        secondary.append({"mechanism_id": mid, "mechanism_name": name, "message": msg})

    return {
        "primary_mechanism_id": primary_id,
        "primary_mechanism_name": primary_name,
        "secondary_mechanisms": secondary,
        "compile_messages": cleaned,
        "message_count": len(cleaned),
    }


def mechanism_one_hot(mechanism_id: str) -> list[float]:
    index = {mid: i for i, (mid, _) in enumerate(NON_KC_MECHANISM_CATALOG)}
    vec = [0.0] * MECHANISM_VECTOR_DIM
    idx = index.get(mechanism_id, index["M12"])
    vec[idx] = 1.0
    return vec


def build_error_mechanism_evidence(record: dict) -> dict[str, Any]:
    ci = record.get("code_issues") or {}
    eligible = is_compile_error(record)
    messages = compile_messages(record)
    classified = classify_compile_messages(messages)
    primary_id = classified["primary_mechanism_id"]
    vector = mechanism_one_hot(primary_id) if eligible else [0.0] * MECHANISM_VECTOR_DIM

    return {
        "source": ERROR_MECHANISM_SOURCE,
        "schema_version": ERROR_MECHANISM_SCHEMA_VERSION,
        "eligible": eligible,
        "outcome_type": ci.get("outcome_type"),
        "has_compile_error": bool(ci.get("has_compile_error")),
        "non_kc_error": 1.0 if eligible else 0.0,
        "primary_mechanism_id": primary_id if eligible else None,
        "primary_mechanism_name": classified["primary_mechanism_name"] if eligible else None,
        "secondary_mechanisms": classified["secondary_mechanisms"] if eligible else [],
        "compile_messages": classified["compile_messages"],
        "message_count": classified["message_count"],
        "mechanism_catalog": [
            {"id": mid, "name": name} for mid, name in NON_KC_MECHANISM_CATALOG
        ],
        "vector": vector,
        "vector_dim": len(vector),
    }


def attach_mechanism_to_record(record: dict) -> dict:
    """Attach ``error_evidence['mechanism_v8']`` for compile-error records."""
    evidence = build_error_mechanism_evidence(record)
    out = dict(record)
    block = dict(out.get("error_evidence") or {})
    block["mechanism"] = evidence
    out["error_evidence"] = block
    return out


def attach_mechanism_to_records(records: Sequence[dict]) -> list[dict]:
    return [attach_mechanism_to_record(rec) for rec in records]


def mechanism_vector(record: dict) -> list[float]:
    block = (record.get("error_evidence") or {}).get("mechanism")
    if not isinstance(block, dict):
        block = (record.get("error_evidence") or {}).get("mechanism_v8")
    if isinstance(block, dict) and block.get("vector") is not None:
        vec = [float(v) for v in block["vector"]]
        if len(vec) < MECHANISM_VECTOR_DIM:
            vec.extend([0.0] * (MECHANISM_VECTOR_DIM - len(vec)))
        return vec[:MECHANISM_VECTOR_DIM]
    return [0.0] * MECHANISM_VECTOR_DIM
