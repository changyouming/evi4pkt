from __future__ import annotations

from typing import Literal

OutcomeType = Literal["full_pass", "partial_pass", "test_fail", "compile_error"]


def pkt_label_from_score(score: float, *, threshold: float = 1.0) -> int:
    """PKT binary correctness: 1 if all tests pass, else 0."""
    return 1 if float(score) >= threshold else 0


def classify_outcome(
    score: float,
    *,
    has_compile_error: bool,
    threshold: float = 1.0,
) -> OutcomeType:
    if has_compile_error:
        return "compile_error"
    if float(score) >= threshold:
        return "full_pass"
    if float(score) <= 0.0:
        return "test_fail"
    return "partial_pass"


def build_code_issues(
    score: float,
    *,
    has_compile_error: bool,
    compile_messages: list[str],
    threshold: float = 1.0,
) -> dict:
    """
    Framework log component (3): Code Issues + PKT-aligned correctness label.

    pkt_label is the canonical PKT target used downstream for next-step prediction
    and for interaction encoding (problem x correctness one-hot).
    """
    label = pkt_label_from_score(score, threshold=threshold)
    outcome = classify_outcome(score, has_compile_error=has_compile_error, threshold=threshold)

    issues: list[dict] = []
    for msg in compile_messages:
        text = (msg or "").strip()
        if text:
            issues.append({"type": "compile_error", "message": text})

    if outcome == "partial_pass":
        issues.append(
            {
                "type": "partial_pass",
                "message": f"Partial test pass ratio={float(score):.4f} (< {threshold})",
            }
        )
    elif outcome == "test_fail" and not has_compile_error:
        issues.append({"type": "test_failure", "message": "All tests failed (Score=0)."})
    elif outcome == "full_pass":
        issues.append({"type": "full_pass", "message": "All tests passed."})

    return {
        "pkt_label": label,
        "pkt_correct": bool(label),
        "label_rule": f"score_ge_{threshold}".replace(".", "_"),
        "score": float(score),
        "outcome_type": outcome,
        "has_compile_error": bool(has_compile_error),
        "issues": issues,
    }
