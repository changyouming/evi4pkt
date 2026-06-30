#!/usr/bin/env python3
"""Build Plan A ablation table from measured ladder summaries."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evipkt.feature_modes import (
    ABLATION_LEVEL_ORDER,
    ABLATION_LEVEL_SPECS,
    ABLATION_OUT_SUFFIX,
    LEGACY_ABLATION_OUT_SUFFIX,
    LEGACY_MODEL_KEY_SUFFIX,
    PIPELINE_ID,
    ablation_model_key,
    delta_baseline_for_level,
    is_canonical_logs_path,
    normalize_ablation_level,
    normalize_feature_mode,
)

OUT_DIR = PROJECT_ROOT / "runs/final_report/tables"
METRICS = ("auc", "acc", "f1")

BACKBONES: tuple[tuple[str, str], ...] = (
    ("DKT", "dkt"),
    ("DKVMN", "dkvmn"),
    ("SAKT", "sakt"),
    ("AKT", "akt"),
    ("qDKT", "qdkt"),
    ("QIKT", "qikt"),
    ("SimpleKT", "simplekt"),
    ("SparseKT", "sparsekt"),
)


def _find_summary(prefix: str) -> Path | None:
    """Prefer ablation_ladder_first over legacy v8 (longer filename ≠ newer)."""
    canonical_dir = PROJECT_ROOT / f"runs/{prefix}_{ABLATION_OUT_SUFFIX}"
    if canonical_dir.is_dir():
        matches = list(canonical_dir.glob(f"{prefix}_ablation_seeds_*_summary.json"))
        if matches:
            return max(matches, key=lambda p: len(p.name))
    legacy_dir = PROJECT_ROOT / f"runs/{prefix}_{LEGACY_ABLATION_OUT_SUFFIX}"
    if legacy_dir.is_dir():
        matches = list(legacy_dir.glob(f"{prefix}_ablation_v8_seeds_*_summary.json"))
        if matches:
            return max(matches, key=lambda p: len(p.name))
    return None


def _find_results(summary: Path) -> Path | None:
    p = summary.with_name(summary.name.replace("_summary.json", ".json"))
    return p if p.exists() else None


def _resolve_key(summary: dict, prefix: str, level: str) -> str | None:
    level = normalize_ablation_level(level)
    key = ablation_model_key(prefix, level)
    if key in summary.get("models", {}):
        return key
    legacy = LEGACY_MODEL_KEY_SUFFIX.get(level)
    if legacy:
        lk = f"{prefix}_{legacy}"
        if lk in summary.get("models", {}):
            return lk
    return None


def _load_rows(prefix: str) -> tuple[dict | None, list[dict] | None]:
    summary_path = _find_summary(prefix)
    if summary_path is None:
        return None, None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("pipeline_id") not in (PIPELINE_ID, "v8_misused_v8") and not is_canonical_logs_path(
        summary.get("logs_path") or ""
    ):
        return None, None
    results_path = _find_results(summary_path)
    rows = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path
        else None
    )
    return summary, rows


def _metric_mean_std(summary: dict, key: str, metric: str) -> tuple[float | None, float | None]:
    block = summary.get("models", {}).get(key, {}).get(metric)
    if not block or block.get("mean") is None:
        return None, None
    return float(block["mean"]), float(block.get("std", 0.0))


def _paired_delta(rows: list[dict], key_a: str, key_b: str, metric: str) -> tuple[float | None, float | None]:
    diffs: list[float] = []
    for row in rows:
        if key_a not in row or key_b not in row:
            continue
        a = row[key_a].get("test", {}).get(metric)
        b = row[key_b].get("test", {}).get(metric)
        if a is not None and b is not None:
            diffs.append(float(b) - float(a))
    if not diffs:
        return None, None
    return statistics.mean(diffs), statistics.stdev(diffs) if len(diffs) > 1 else 0.0


def _fmt(m: float | None, s: float | None) -> str:
    if m is None:
        return "—"
    if s is None:
        return f"{m:.4f}"
    return f"{m:.4f} ± {s:.4f}"


def _fmt_delta(m: float | None, s: float | None) -> str:
    if m is None:
        return "—"
    pp = m * 100.0
    pp_s = (s * 100.0) if s is not None else None
    if pp_s is None:
        return f"{pp:+.2f}"
    return f"{pp:+.2f} ± {pp_s:.2f}"


def build_payload() -> dict:
    payload: dict = {
        "plan": "A",
        "pipeline_id": PIPELINE_ID,
        "levels": list(ABLATION_LEVEL_ORDER),
        "backbones": {},
    }
    for bb_name, prefix in BACKBONES:
        summary, rows = _load_rows(prefix)
        bb: dict = {"prefix": prefix, "levels": {}, "status": "missing"}
        if summary is None:
            payload["backbones"][bb_name] = bb
            continue
        bb["status"] = "ok"
        keys = {lv: _resolve_key(summary, prefix, lv) for lv in ABLATION_LEVEL_ORDER}
        for spec in ABLATION_LEVEL_SPECS:
            lv = spec["level"]
            key = keys[lv]
            entry: dict = {
                "tier": spec["tier"],
                "feature_mode": normalize_feature_mode(spec["feature_mode"]),
                "delta_baseline": spec["delta_baseline"],
            }
            for metric in METRICS:
                mean, std = _metric_mean_std(summary, key, metric) if key else (None, None)
                entry[metric] = {"mean": mean, "std": std}
            base_lv = spec["delta_baseline"]
            base_key = keys.get(base_lv)
            if rows and key and base_key and lv != base_lv:
                for metric in METRICS:
                    d_mean, d_std = _paired_delta(rows, base_key, key, metric)
                    entry.setdefault("delta", {})[metric] = {
                        "vs": base_lv,
                        "mean": d_mean,
                        "std": d_std,
                        "pp_mean": (d_mean * 100.0) if d_mean is not None else None,
                    }
            bb["levels"][lv] = entry
        payload["backbones"][bb_name] = bb
    return payload


def _markdown_table(payload: dict, metric: str) -> str:
    lines = [
        f"# Plan A ablation ({metric.upper()})",
        "",
        f"v2–v4: Δ vs **v1**; v5 = v4 + Mechanism, Δ vs **v4**.",
        "",
        "| Backbone | " + " | ".join(ABLATION_LEVEL_ORDER) + " |",
        "| --- | " + " | ".join(["---:"] * len(ABLATION_LEVEL_ORDER)) + " |",
    ]
    for bb_name, _prefix in BACKBONES:
        bb = payload["backbones"].get(bb_name, {})
        cells = [bb_name]
        for lv in ABLATION_LEVEL_ORDER:
            block = bb.get("levels", {}).get(lv, {})
            m = block.get(metric, {})
            cells.append(_fmt(m.get("mean"), m.get("std")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Δ (pp)", ""])
    header = "| Backbone | " + " | ".join(
        f"{lv} (vs {delta_baseline_for_level(lv)})" for lv in ABLATION_LEVEL_ORDER if lv != "v0"
    ) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join(["---:"] * (len(ABLATION_LEVEL_ORDER) - 1)) + " |")
    for bb_name, _prefix in BACKBONES:
        bb = payload["backbones"].get(bb_name, {})
        cells = [bb_name]
        for lv in ABLATION_LEVEL_ORDER:
            if lv == "v0":
                continue
            d = bb.get("levels", {}).get(lv, {}).get("delta", {}).get(metric, {})
            cells.append(_fmt_delta(d.get("mean"), d.get("std")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Generate: `python scripts/build_ablation_plan_table.py`", ""])
    return "\n".join(lines)


def main() -> None:
    payload = build_payload()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "ablation_plan_a_measured.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for metric in METRICS:
        md_path = OUT_DIR / f"table_ablation_plan_a_{metric}.md"
        md_path.write_text(_markdown_table(payload, metric), encoding="utf-8")
        print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
