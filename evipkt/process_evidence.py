from __future__ import annotations

import math
from typing import Sequence

from .kc_catalog import DEFAULT_KC_CATALOG

PROCESS_EVIDENCE_SOURCE = "kc_history"


def _active_kc_indices(q_kc: Sequence[float] | None, n: int) -> list[int]:
    if q_kc is None:
        return []
    return [i for i, v in enumerate(q_kc[:n]) if float(v) > 0]


def pack_process_evidence(
    exposure_count: Sequence[int],
    success_count: Sequence[int],
    *,
    catalog: Sequence[str] | None = None,
) -> dict:
    """Build process evidence from KC history before the current interaction."""
    catalog = list(catalog or DEFAULT_KC_CATALOG)
    n = len(catalog)
    exposure = list(exposure_count[:n])
    success = list(success_count[:n])
    if len(exposure) < n:
        exposure += [0] * (n - len(exposure))
    if len(success) < n:
        success += [0] * (n - len(success))

    exposure_norm = [min(1.0, math.log1p(c) / math.log1p(20.0)) for c in exposure]
    success_rate = [
        (float(s) / float(c)) if c > 0 else 0.0 for s, c in zip(success, exposure)
    ]
    vector = exposure_norm + success_rate
    return {
        "source": PROCESS_EVIDENCE_SOURCE,
        "timing": "before_current_interaction",
        "kc_catalog": catalog,
        "kc_exposure_count": exposure,
        "kc_success_count": success,
        "kc_exposure_norm": exposure_norm,
        "kc_success_rate": success_rate,
        "vector": vector,
        "vector_dim": len(vector),
    }


def attach_process_evidence_to_records(
    records: list[dict],
    *,
    catalog: Sequence[str] | None = None,
) -> list[dict]:
    """
    Add process_evidence to each record using only earlier interactions of the same student.

    The current record's outcome is added to the running KC history only after its
    process_evidence has been created, preventing target leakage.
    """
    catalog = list(catalog or DEFAULT_KC_CATALOG)
    n = len(catalog)
    by_student: dict[str, list[tuple[int, int, dict]]] = {}
    for pos, rec in enumerate(records):
        sid = str(rec["subject_id"])
        timestep = int((rec.get("trajectory") or {}).get("student_timestep", pos))
        by_student.setdefault(sid, []).append((timestep, pos, rec))

    out = [dict(rec) for rec in records]
    for rows in by_student.values():
        rows.sort(key=lambda x: (x[0], x[1]))
        exposure = [0] * n
        success = [0] * n
        for _, pos, rec in rows:
            out[pos]["process_evidence"] = pack_process_evidence(
                exposure,
                success,
                catalog=catalog,
            )
            q_kc = (rec.get("programming_task") or {}).get("q_kc")
            y = int(rec.get("pkt_label", (rec.get("code_issues") or {}).get("pkt_label", 0)))
            for idx in _active_kc_indices(q_kc, n):
                exposure[idx] += 1
                success[idx] += y
    return out


def process_evidence_vector(record: dict, *, catalog: Sequence[str] | None = None) -> list[float]:
    block = record.get("process_evidence")
    if isinstance(block, dict) and block.get("vector"):
        return list(block["vector"])
    catalog = list(catalog or DEFAULT_KC_CATALOG)
    return [0.0] * (2 * len(catalog))
