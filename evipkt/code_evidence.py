from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence

from .kc_catalog import DEFAULT_KC_CATALOG

# Rule-based code evidence (no LLM). Fixed-size vector for KT interaction encoding.
CODE_EVIDENCE_SOURCE = "rule_based"
CODE_EVIDENCE_VECTOR_DIM = len(DEFAULT_KC_CATALOG) + 5


def _count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))


def _count_if(text: str) -> int:
    return _count_matches(r"\bif\s*\(", text)


def _kc_detectors() -> Dict[str, Callable[[str], bool]]:
    return {
        "IfElse": lambda t: _count_if(t) >= 1,
        "NestedIf": lambda t: _count_if(t) >= 2,
        "While": lambda t: bool(re.search(r"\bwhile\s*\(", t, re.I)),
        "For": lambda t: bool(re.search(r"\bfor\s*\(", t, re.I)),
        "NestedFor": lambda t: len(re.findall(r"\bfor\s*\(", t, re.I)) >= 2,
        "MathBasic": lambda t: bool(re.search(r"[+\-*/]", t)),
        "MathMod": lambda t: bool(re.search(r"%", t)),
        "LogicAndNotOr": lambda t: bool(
            re.search(r"&&|\|\||\b!\s*\(|\bnot\b", t, re.I)
        ),
        "LogicCompareNum": lambda t: bool(re.search(r"[<>=!]=?|!=", t)),
        "LogicBoolean": lambda t: bool(re.search(r"\b(true|false)\b", t, re.I)),
        "StringFormat": lambda t: bool(
            re.search(r"System\.out\.(print|format)|String\.format|printf\s*\(", t, re.I)
        ),
        "StringConcat": lambda t: bool(re.search(r"\+.*\"|\"\s*\+", t)),
        "StringIndex": lambda t: bool(re.search(r"\.charAt\s*\(|\.substring\s*\(", t, re.I)),
        "StringLen": lambda t: bool(re.search(r"\.length\s*\(", t, re.I)),
        "StringEqual": lambda t: bool(re.search(r"\.equals\s*\(", t, re.I)),
        "CharEqual": lambda t: bool(re.search(r"['\"].?['\"]\s*==|==\s*['\"].?['\"]", t)),
        "ArrayIndex": lambda t: bool(re.search(r"\[\s*\w+\s*\]", t)),
        "DefFunction": lambda t: bool(
            re.search(
                r"\b(?:public|private|protected|static|\w+)\s+[\w<>\[\]]+\s+\w+\s*\(",
                t,
                re.I,
            )
            or re.search(r"\bvoid\s+\w+\s*\(", t, re.I)
        ),
    }


def detect_kc_in_code(code: str, catalog: Sequence[str] | None = None) -> List[float]:
    """Binary indicators: KC construct appears in submitted Java code."""
    catalog = list(catalog or DEFAULT_KC_CATALOG)
    detectors = _kc_detectors()
    text = code or ""
    return [1.0 if detectors[name](text) else 0.0 for name in catalog]


def _code_metrics(code: str) -> dict:
    text = code or ""
    lines = text.splitlines() if text else []
    non_empty = [ln for ln in lines if ln.strip()]
    loc = len(non_empty)
    chars = len(text)
    return {
        "line_count": loc,
        "char_count": chars,
        "non_empty_line_count": loc,
        "has_brace": bool("{" in text and "}" in text),
    }


def _alignment_stats(
    kc_in_code: List[float],
    q_kc: Sequence[float],
    *,
    catalog: Sequence[str] | None = None,
) -> dict:
    catalog = list(catalog or DEFAULT_KC_CATALOG)
    n = min(len(kc_in_code), len(q_kc), len(catalog))
    if n == 0:
        return {
            "kc_coverage": 0.0,
            "missing_kc_ratio": 0.0,
            "extra_kc_ratio": 0.0,
            "missing_kc": [],
            "extra_kc": [],
        }
    code = kc_in_code[:n]
    req = list(q_kc[:n])
    n_req = sum(1 for x in req if x > 0)
    covered = sum(1 for c, r in zip(code, req) if r > 0 and c > 0)
    missing = [catalog[i] for i, (c, r) in enumerate(zip(code, req)) if r > 0 and c <= 0]
    extra = [catalog[i] for i, (c, r) in enumerate(zip(code, req)) if c > 0 and r <= 0]
    n_code = sum(1 for c in code if c > 0)
    return {
        "kc_coverage": covered / n_req if n_req else 0.0,
        "missing_kc_ratio": len(missing) / n_req if n_req else 0.0,
        "extra_kc_ratio": len(extra) / n_code if n_code else 0.0,
        "missing_kc": missing,
        "extra_kc": extra,
    }


def pack_code_evidence(
    kc_in_code: List[float],
    q_kc: Sequence[float] | None,
    *,
    catalog: Sequence[str] | None = None,
    metrics: dict | None = None,
    source: str = CODE_EVIDENCE_SOURCE,
) -> dict:
    """Build code-evidence dict + fixed vector from KC-in-code and task Q."""
    import math

    catalog = list(catalog or DEFAULT_KC_CATALOG)
    q_row = list(q_kc) if q_kc is not None else [0.0] * len(catalog)
    if len(q_row) < len(catalog):
        q_row = q_row + [0.0] * (len(catalog) - len(q_row))
    align = _alignment_stats(kc_in_code, q_row, catalog=catalog)
    m = metrics if metrics is not None else {"line_count": 0, "char_count": 0}
    norm_loc = min(1.0, math.log1p(m.get("line_count", 0)) / math.log1p(50.0))
    norm_chars = min(1.0, math.log1p(m.get("char_count", 0)) / math.log1p(2000.0))
    vector = list(kc_in_code) + [
        align["kc_coverage"],
        align["missing_kc_ratio"],
        align["extra_kc_ratio"],
        norm_loc,
        norm_chars,
    ]
    return {
        "source": source,
        "kc_catalog": catalog,
        "kc_in_code": list(kc_in_code),
        "kc_coverage": align["kc_coverage"],
        "missing_kc": align["missing_kc"],
        "extra_kc": align["extra_kc"],
        "metrics": m,
        "vector": vector,
        "vector_dim": len(vector),
    }


def build_code_evidence(
    code: str,
    q_kc: Sequence[float] | None = None,
    *,
    catalog: Sequence[str] | None = None,
) -> dict:
    """
    Structured code evidence for one submission.

    Answers: which KC constructs appear in code, and how they align with task Q.
    """
    catalog = list(catalog or DEFAULT_KC_CATALOG)
    kc_in_code = detect_kc_in_code(code, catalog)
    metrics = _code_metrics(code)
    return pack_code_evidence(
        kc_in_code,
        q_kc,
        catalog=catalog,
        metrics=metrics,
        source=CODE_EVIDENCE_SOURCE,
    )


LLM_CODE_FEATURE_MODES = frozenset(
    {
        "problem_plus_code_llm",
        "problem_plus_q_code_llm",
        "problem_plus_q_code_llm_process",
        "problem_plus_q_code_llm_error",
        "problem_plus_q_code_llm_process_error",
    }
)


def code_evidence_backend_for_mode(feature_mode: str) -> str:
    if feature_mode in LLM_CODE_FEATURE_MODES:
        return "llm"
    return "rule"


def code_evidence_source_for_mode(feature_mode: str) -> str | None:
    if feature_mode in LLM_CODE_FEATURE_MODES:
        return "llm_not_available_in_repro"
    if feature_mode in ("problem_plus_code", "problem_plus_q_code_rule"):
        return CODE_EVIDENCE_SOURCE
    return None


def code_evidence_vector(
    record: dict,
    *,
    catalog: Sequence[str] | None = None,
    backend: str = "rule",
) -> List[float]:
    """Load vector from framework log; backend='llm' uses LLM KC alignment when present."""
    block = record.get("code_evidence")
    if isinstance(block, dict):
        if backend == "llm":
            llm_block = block.get("llm")
            if isinstance(llm_block, dict) and llm_block.get("vector"):
                return list(llm_block["vector"])
            if block.get("source") == "llm_kc_alignment" and block.get("vector"):
                return list(block["vector"])
        elif block.get("vector"):
            rule = block.get("rule")
            if isinstance(rule, dict) and rule.get("vector"):
                return list(rule["vector"])
            return list(block["vector"])
    code = ""
    sc = record.get("student_code")
    if isinstance(sc, dict):
        code = str(sc.get("code") or "")
    q_kc = None
    pt = record.get("programming_task")
    if isinstance(pt, dict):
        q_kc = pt.get("q_kc")
    if backend == "llm":
        raise RuntimeError("LLM code evidence is not available in the reproduction bundle.")
    return list(build_code_evidence(code, q_kc, catalog=catalog)["vector"])
