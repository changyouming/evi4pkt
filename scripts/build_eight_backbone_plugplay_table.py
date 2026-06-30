#!/usr/bin/env python3
"""Build Table VIII: eight-backbone plug-and-play (AUC / ACC / F1)."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = PROJECT_ROOT / "runs/final_report/figures/fig_backbone_ladder_v8_auc_manifest.json"
OUT_DIR = PROJECT_ROOT / "runs/final_report/tables"

METRICS = ("auc", "acc", "f1")
METRIC_TITLES = {
    "auc": "AUC",
    "acc": "ACC",
    "f1": "F1 (F-score)",
}

EIGHT_BACKBONES: tuple[dict[str, str], ...] = (
    {"name": "DKT", "family": "Sequential", "prefix": "dkt"},
    {"name": "DKVMN", "family": "Sequential", "prefix": "dkvmn"},
    {"name": "SAKT", "family": "Sequential", "prefix": "sakt"},
    {"name": "AKT", "family": "Sequential", "prefix": "akt"},
    {"name": "qDKT", "family": "Q-matrix–aware", "prefix": "qdkt"},
    {"name": "QIKT", "family": "Q-matrix–aware", "prefix": "qikt"},
    {"name": "SimpleKT", "family": "Attention-based", "prefix": "simplekt"},
    {"name": "SparseKT", "family": "Attention-based", "prefix": "sparsekt"},
)

from evipkt.feature_modes import (
    ABLATION_OUT_SUFFIX,
    LEGACY_ABLATION_OUT_SUFFIX,
    PIPELINE_ID,
    is_canonical_logs_path,
    normalize_feature_mode,
)

V0_MODE = "problem_onehot"
V6_MODE = "problem_plus_q_process_code_mechanism"
FULL_LEVEL = "v5"
LEGACY_PIPELINE_IDS = {PIPELINE_ID, "v8_misused_v8"}


def _fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _paired_t(diff: list[float]) -> float | None:
    if len(diff) < 2:
        return None
    try:
        from scipy import stats

        return float(stats.ttest_1samp(diff, 0.0).pvalue)
    except ImportError:
        return None


def _model_key(prefix: str, feature_mode: str) -> str:
    return normalize_feature_mode(feature_mode).replace("problem_", f"{prefix}_")


def _legacy_v6_key(prefix: str) -> str:
    return f"{prefix}_plus_q_process_code_v8_mechanism_v8"


def _full_level_keys(summary: dict, prefix: str) -> list[str]:
    keys = [
        _model_key(prefix, V6_MODE),
        _legacy_v6_key(prefix),
    ]
    return [k for k in keys if k in summary.get("models", {})]


def _find_ablation_summary(prefix: str) -> Path | None:
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


def _summary_is_canonical_pipeline(summary: dict) -> bool:
    if summary.get("pipeline_id") in LEGACY_PIPELINE_IDS:
        return True
    logs = summary.get("logs_path") or ""
    return is_canonical_logs_path(logs)


def _resolve_model_key(summary: dict, prefix: str, feature_mode: str) -> str:
    key = _model_key(prefix, feature_mode)
    if key in summary.get("models", {}):
        return key
    legacy_keys = {
        _model_key(prefix, V6_MODE): _legacy_v6_key(prefix),
    }
    return legacy_keys.get(key, key)


def _row_from_ablation(bb: dict, metric: str, seeds: list[int]) -> dict:
    prefix = bb["prefix"]
    summary_path = _find_ablation_summary(prefix)
    if summary_path is None:
        return {
            "backbone": bb["name"],
            "family": bb["family"],
            "v0": None,
            "full": None,
            "delta": None,
            "p": None,
            "status": "pending",
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not _summary_is_canonical_pipeline(summary):
        return {
            "backbone": bb["name"],
            "family": bb["family"],
            "v0": None,
            "full": None,
            "delta": None,
            "p": None,
            "status": "legacy_pipeline",
        }

    v0_key = _model_key(prefix, V0_MODE)
    v6_key = _resolve_model_key(summary, prefix, V6_MODE)
    v0 = _metric_block(summary, v0_key, metric)
    v6 = _metric_block(summary, v6_key, metric)
    results_path = summary_path.with_name(summary_path.name.replace("_summary.json", ".json"))
    delta, p = _paired_delta(results_path, v0_key, v6_key, metric, seeds)
    if delta is None and v0 and v6:
        delta = {"mean": v6["mean"] - v0["mean"], "std": None}
    return {
        "backbone": bb["name"],
        "family": bb["family"],
        "v0": v0,
        "full": v6,
        "delta": delta,
        "p": p,
        "status": "complete" if v0 and v6 else "partial",
    }


def _metric_block(summary: dict, model_key: str, metric: str) -> dict | None:
    block = summary.get("models", {}).get(model_key, {}).get(metric)
    if not block or block.get("mean") is None:
        return None
    return {"mean": block["mean"], "std": block.get("std", 0.0)}


def _paired_delta(
    results_path: Path | None,
    v0_key: str,
    v6_key: str,
    metric: str,
    seeds: list[int],
    v0_results_path: Path | None = None,
) -> tuple[dict | None, float | None]:
    v0_path = v0_results_path or results_path
    v6_path = results_path
    if v0_path is None or v6_path is None or not v0_path.exists() or not v6_path.exists():
        return None, None
    v0_rows = json.loads(v0_path.read_text(encoding="utf-8"))
    v6_rows = v0_rows if v0_path == v6_path else json.loads(v6_path.read_text(encoding="utf-8"))
    by_v0 = {
        r["seed"]: r[v0_key]["test"][metric]
        for r in v0_rows
        if v0_key in r and metric in r[v0_key].get("test", {})
    }
    by_v6 = {
        r["seed"]: r[v6_key]["test"][metric]
        for r in v6_rows
        if v6_key in r and metric in r[v6_key].get("test", {})
    }
    diffs = [by_v6[s] - by_v0[s] for s in seeds if s in by_v0 and s in by_v6]
    if not diffs:
        return None, None
    return {
        "mean": statistics.mean(diffs),
        "std": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
    }, _paired_t(diffs)


def _metric_table_md(metric: str, rows: list[dict]) -> list[str]:
    title = METRIC_TITLES[metric]
    lines = [
        f"## {title}",
        "",
        f"| Backbone | Family | Backbone (V0) | Full Evi4PKT | Δ (Full − V0) | *p* |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        v0_s = _fmt(r["v0"]["mean"], r["v0"]["std"]) if r["v0"] else "—"
        full_s = _fmt(r["full"]["mean"], r["full"]["std"]) if r["full"] else "—"
        d = r.get("delta") or {}
        if d.get("mean") is not None and d.get("std") is not None:
            d_s = f"{d['mean']:+.4f} ± {d['std']:.4f}"
        elif d.get("mean") is not None:
            d_s = f"{d['mean']:+.4f}"
        else:
            d_s = "—"
        p = r.get("p")
        p_s = f"{p:.4f}" if p is not None else "—"
        lines.append(f"| {r['backbone']} | {r['family']} | {v0_s} | {full_s} | {d_s} | {p_s} |")
    return lines


def _combined_markdown(all_tables: dict[str, list[dict]], seeds: list[int]) -> str:
    n_complete = sum(1 for r in all_tables["auc"] if r["status"] == "complete")
    lines = [
        "# TABLE VIII — Plug-and-Play Evi4PKT on Eight KT Backbones",
        "",
        "Setting: CSEDM CodeWorkout F19, first-attempt, student-level 80/10/10, "
        f"{len(seeds)} seeds.",
        "V0: `problem_onehot`. "
        "Full: `problem_plus_q_process_code_mechanism`.",
        "Evidence concatenated to history; backbone internals unchanged.",
        "",
    ]
    for metric in METRICS:
        lines.extend(_metric_table_md(metric, all_tables[metric]))
        lines.append("")

    lines += [
        f"*Table note:* {n_complete}/8 backbones complete. "
        "Δ and *p*: paired *t*-test on per-seed differences.",
        "",
        "Generate: `python scripts/build_eight_backbone_plugplay_table.py`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    seeds = manifest.get("seeds", list(range(10)))
    all_tables: dict[str, list[dict]] = {}
    for metric in METRICS:
        rows = [_row_from_ablation(bb, metric, seeds) for bb in EIGHT_BACKBONES]
        all_tables[metric] = rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        path = OUT_DIR / f"table_viii_eight_backbone_{metric}.md"
        title = METRIC_TITLES[metric]
        body = [
            f"# TABLE VIII — Plug-and-Play Evi4PKT ({title})",
            "",
            *_metric_table_md(metric, all_tables[metric]),
            "",
            "Generate: `python scripts/build_eight_backbone_plugplay_table.py`",
            "",
        ]
        path.write_text("\n".join(body), encoding="utf-8")

    combined = OUT_DIR / "table_viii_eight_backbone_plugplay.md"
    combined.write_text(_combined_markdown(all_tables, seeds), encoding="utf-8")

    payload = {
        "table": "TABLE VIII",
        "metrics": list(METRICS),
        "seeds": seeds,
        "by_metric": all_tables,
        "n_complete": sum(1 for r in all_tables["auc"] if r["status"] == "complete"),
    }
    (OUT_DIR / "table_viii_eight_backbone_plugplay.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(combined.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
