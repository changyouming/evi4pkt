"""Canonical Evi4PKT feature modes and dependency-aware ablation plan (Plan A)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

CANONICAL_LOGS = "data/processed/framework_logs_first_process_mechanism.jsonl"
LEGACY_LOGS_V8 = "data/processed/framework_logs_first_process_mechanism.jsonl"
PIPELINE_ID = "evipkt_rule_process_mechanism"

Tier = Literal["spine", "plugin", "pair", "full"]
DeltaBaseline = Literal["v0", "v1", "v4"]


class AblationLevelSpec(TypedDict):
    level: str
    feature_mode: str
    logs_path: str
    label: str
    tier: Tier
    parent: str | None
    enables: tuple[str, ...]
    delta_baseline: DeltaBaseline
    depends_on: tuple[str, ...]
    note: str


CANONICAL_PLUGPLAY_MODES: frozenset[str] = frozenset(
    {
        "problem_plus_q_code",
        "problem_plus_q_mechanism",
        "problem_plus_q_code_mechanism",
        "problem_plus_q_process_code",
        "problem_plus_q_process_code_mechanism",
    }
)

FEATURE_MODE_ALIASES: dict[str, str] = {
    "problem_plus_q_code_v8": "problem_plus_q_code",
    "problem_plus_q_mechanism_v8": "problem_plus_q_mechanism",
    "problem_plus_q_code_v8_mechanism_v8": "problem_plus_q_code_mechanism",
    "problem_plus_q_process_code_v8_mechanism_v8": "problem_plus_q_process_code_mechanism",
}

# Plan A v0–v5: v5 = v4 + Mechanism (strict cumulative on the v4 stack).
ABLATION_LEVEL_SPECS: tuple[AblationLevelSpec, ...] = (
    {
        "level": "v0",
        "feature_mode": "problem_onehot",
        "logs_path": CANONICAL_LOGS,
        "label": "Backbone (V0)",
        "tier": "spine",
        "parent": None,
        "enables": ("problem",),
        "delta_baseline": "v0",
        "depends_on": (),
        "note": "Problem one-hot only.",
    },
    {
        "level": "v1",
        "feature_mode": "problem_plus_q",
        "logs_path": CANONICAL_LOGS,
        "label": "+ Task Q",
        "tier": "spine",
        "parent": "v0",
        "enables": ("problem", "Q"),
        "delta_baseline": "v0",
        "depends_on": ("v0",),
        "note": "Expert Q-matrix on target/history.",
    },
    {
        "level": "v2",
        "feature_mode": "problem_plus_q_process",
        "logs_path": CANONICAL_LOGS,
        "label": "+ Process",
        "tier": "plugin",
        "parent": "v1",
        "enables": ("problem", "Q", "Process"),
        "delta_baseline": "v1",
        "depends_on": ("v1",),
        "note": "Pre-attempt KC history (requires Q).",
    },
    {
        "level": "v3",
        "feature_mode": "problem_plus_q_code",
        "logs_path": CANONICAL_LOGS,
        "label": "+ Code",
        "tier": "plugin",
        "parent": "v1",
        "enables": ("problem", "Q", "Code"),
        "delta_baseline": "v1",
        "depends_on": ("v1",),
        "note": "Post-attempt misused/missing KC; does not require Process.",
    },
    {
        "level": "v4",
        "feature_mode": "problem_plus_q_process_code",
        "logs_path": CANONICAL_LOGS,
        "label": "+ Process + Code",
        "tier": "pair",
        "parent": "v1",
        "enables": ("problem", "Q", "Process", "Code"),
        "delta_baseline": "v1",
        "depends_on": ("v1", "v2", "v3"),
        "note": "Process and Code together on Q base.",
    },
    {
        "level": "v5",
        "feature_mode": "problem_plus_q_process_code_mechanism",
        "logs_path": CANONICAL_LOGS,
        "label": "+ Mechanism (Full)",
        "tier": "full",
        "parent": "v4",
        "enables": ("problem", "Q", "Process", "Code", "Mechanism"),
        "delta_baseline": "v4",
        "depends_on": ("v4",),
        "note": "v5 = v4 + Mechanism (compile-error channel in history).",
    },
)

ABLATION_LADDER: tuple[dict[str, str], ...] = tuple(
    {"level": s["level"], "feature_mode": s["feature_mode"], "logs_path": s["logs_path"]}
    for s in ABLATION_LEVEL_SPECS
)

ABLATION_LEVEL_ORDER: tuple[str, ...] = tuple(s["level"] for s in ABLATION_LEVEL_SPECS)
DEFAULT_ABLATION_LEVELS: tuple[str, ...] = ABLATION_LEVEL_ORDER

ABLATION_COLUMN_GROUPS: tuple[tuple[str, ...], ...] = (
    ("v0", "v1"),
    ("v2", "v3"),
    ("v4",),
    ("v5",),
)

LEGACY_LEVEL_ALIASES: dict[str, str] = {
    "v3": "v2",
    "v4": "v3",
    "v6": "v5",
}

LEGACY_MODEL_KEY_SUFFIX: dict[str, str] = {
    "v3": "plus_q_code_v8",
    "v5": "plus_q_process_code_v8_mechanism_v8",
}

ABLATION_OUT_SUFFIX = "ablation_ladder_first"
LEGACY_ABLATION_OUT_SUFFIX = "ablation_ladder_v8_first"

PLAN_A_CAPTION = (
    "Plan A: v0→v1 (+Q); v2/v3 plugins @ v1; v4 +Process+Code; "
    "v5 = v4 + Mechanism. Δ: v2–v4 vs v1; v5 vs v4."
)

# Paper figures: no internal level IDs (V0/V1/…).
FIGURE_LEVEL_LABEL: dict[str, str] = {
    "v0": "Backbone",
    "v1": "+ Task Q",
    "v2": "+ Process\n| Q",
    "v3": "+ Code\n| Q",
    "v4": "+ Code+Process\n| Q",
    "v5": "+ Compile Error\n(Full)",
}

FIGURE_DELTA_BASELINE_LABEL: dict[str, str] = {
    "v0": "Backbone",
    "v1": "Task Q",
    "v4": "Code+Proc.|Q",
}

FIGURE_PLAN_A_CAPTION = (
    "Spine: Backbone → +Task Q; single plugins (+Process|Q, +Code|Q); "
    "+Code+Process|Q; +Compile Error (Full) = complete Evi4PKT (slight noise vs peak). "
    "Δ: plugins vs Task Q; Full vs +Code+Process|Q."
)

FIGURE_COLUMN_GROUPS = ("Spine", "Single plugins", "Combined", "Full")

FIGURE_DELTA_SHORT_LABEL: dict[str, str] = {
    "v1": "+ Task Q",
    "v2": "+ Process",
    "v3": "+ Code",
    "v4": "+ Process + Code",
    "v5": "+ Compile Error (Full)",
}

# Δ heatmap x-axis: compact symbols (full names in caption).
# Matches backbone-ladder notation: +Q +P +C +P+C +E.
FIGURE_DELTA_COLUMN_SYMBOL: dict[str, str] = {
    "v1": "+Q",
    "v2": "+P",
    "v3": "+C",
    "v4": "+P+C",
    "v5": "+E",
}

FIGURE_DELTA_SYMBOL_LEGEND: tuple[tuple[str, str], ...] = (
    ("+Q", "+ Task Q (expert Q-matrix)"),
    ("+P", "+ Process evidence"),
    ("+C", "+ Code evidence (missing/misused KC)"),
    ("+P+C", "+ Process + Code jointly"),
    ("+E", "+ Compile Error (Full Evi4PKT)"),
)

# Δ heatmap x-axis: +prefix labels (default plot mode).
FIGURE_DELTA_COLUMN_FOOTNOTE: dict[str, str] = {
    "v1": "+Task Q",
    "v2": "+Process|Q",
    "v3": "+Code|Q",
    "v4": "+Process\n+Code|Q",
    "v5": "+E",
}

# Default axis labels for Δ heatmap plot.
FIGURE_DELTA_COLUMN_LABEL: dict[str, str] = dict(FIGURE_DELTA_COLUMN_SYMBOL)

# Other ablation heatmaps: compact single-column ticks.
FIGURE_LEVEL_LABEL_COMPACT: dict[str, str] = {
    "v0": "Base",
    "v1": "+Task Q",
    "v2": "+Proc.|Q",
    "v3": "+Code|Q",
    "v4": "+P+C|Q",
    "v5": "Full",
}

# Shared abbreviation expansions for ablation figure captions (IEEE paste-ready).
FIGURE_ABBREVIATION_GLOSSARY: tuple[tuple[str, str], ...] = (
    ("Base", "backbone-only baseline (problem one-hot); no Evi4PKT evidence"),
    ("+Task Q", "expert Q-matrix: task-required KC indicators on history steps"),
    (
        "+Proc.|Q",
        "process evidence (pre-attempt KC exposure/success under Q), "
        "single plugin on the Task-Q spine",
    ),
    (
        "+Code|Q",
        "code evidence (missing/misused KC labels from executable failures), "
        "single plugin on the Task-Q spine",
    ),
    (
        "+P+C|Q",
        "combined process and code evidence on the Task-Q spine "
        "(+ Process + Code | Q)",
    ),
    (
        "Full",
        "complete Evi4PKT: +P+C|Q plus compile-error mechanism evidence "
        "(+ Compile Error | Q)",
    ),
    ("| Q", "evidence vectors are Q-anchored / concatenated on the Task-Q base"),
    ("Proc.", "process (pre-attempt KC history)"),
    ("P+C", "process + code jointly"),
    ("Err", "compile-error mechanism channel (non-KC auxiliary tags)"),
    ("Δ (pp)", "seed-paired change in percentage points vs the dependency-aligned baseline"),
    ("w/o", "variant with the listed component removed from Full"),
)

FIGURE_CAPTION_ABLATION_LADDER = (
    "Plan A evidence ablation on eight KT backbones (test AUC). "
    "Columns follow the dependency-aligned ladder: Base → +Task Q; single plugins "
    "+Proc.|Q and +Code|Q branch from +Task Q; +P+C|Q combines both; Full adds "
    "compile-error mechanism evidence. "
    "Abbreviations: Base = backbone-only; +Task Q = expert Q-matrix; +Proc.|Q = "
    "process evidence | Q; +Code|Q = code (missing/misused KC) evidence | Q; "
    "+P+C|Q = process + code | Q; Full = + Compile Error (complete Evi4PKT). "
    "| Q denotes Q-anchored evidence concatenated to the backbone input."
)

FIGURE_CAPTION_ABLATION_DELTA = (
    "Plan A incremental ablation on eight KT backbones (rows): test AUC change "
    "(Δ, percentage points) when adding each evidence block (columns: +Task Q, "
    "+Process|Q, +Code|Q, +Process+Code|Q, +E). "
    "Baselines: +Task Q vs backbone-only; +Process|Q, +Code|Q, and +Process+Code|Q "
    "each vs +Task Q; +E (+Compile Error, Full Evi4PKT) vs +Process+Code|Q. "
    "Red/blue: seed-paired gain or drop. All channels are Q-anchored on the expert Q-matrix spine."
)

FIGURE_CAPTION_ABLATION_WITHOUT = (
    "Leave-one-out ablation from Full Evi4PKT: each row removes one evidence "
    "channel while keeping Task Q as the anchor; cell values are Full − w/o variant "
    "(positive ⇒ the removed component helps Full). "
    "Rows: w/o Task Q (backbone-only); w/o Code|Q (Full without code evidence); "
    "w/o Proc.|Q (without process); w/o P+C|Q (without process+code pair); "
    "w/o Err|Q (Full without compile-error mechanism). "
    "Abbreviations: Proc. = process; P+C = process + code; Err = compile-error "
    "mechanism; | Q = Q-anchored evidence channel."
)

FIGURE_CAPTION_ABLATION_LADDER_LINES = (
    "Plan A ablation trajectories (test AUC) across evidence levels for representative "
    "KT backbones. Same ladder as the heatmaps: Base → +Task Q → single plugins "
    "(+Proc.|Q, +Code|Q) → +P+C|Q → Full (+ Compile Error). "
    "Abbreviations: Proc. = process evidence; P+C = process + code; Full = complete "
    "Evi4PKT including compile-error mechanism evidence on the Q spine."
)


def figure_abbreviation_footnote() -> str:
    """One-line parenthetical for inline caption use."""
    return (
        "Abbrev.: Base = backbone-only; +Task Q = expert Q-matrix; "
        "+Proc.|Q = process | Q; +Code|Q = code (missing/misused KC) | Q; "
        "+P+C|Q = process+code | Q; Full = + compile-error mechanism; "
        "Δ in pp; | Q = Q-anchored evidence."
    )


def write_figure_caption(out_stem: Path, caption: str, *, fig_label: str = "") -> None:
    """Write LaTeX-ready caption text alongside a figure export stem."""
    header = f"{fig_label}\n\n" if fig_label else ""
    out_stem.with_suffix(".caption.txt").write_text(
        f"{header}{caption}\n", encoding="utf-8"
    )


def normalize_feature_mode(feature_mode: str) -> str:
    return FEATURE_MODE_ALIASES.get(feature_mode, feature_mode)


def normalize_ablation_level(level: str) -> str:
    """Map legacy level IDs (old ladder) to canonical v0–v5; keep canonical IDs as-is."""
    if level in ABLATION_LEVEL_ORDER:
        return level
    return LEGACY_LEVEL_ALIASES.get(level, level)


def feature_modes_match(a: str, b: str) -> bool:
    return normalize_feature_mode(a) == normalize_feature_mode(b)


def is_canonical_logs_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in {
        CANONICAL_LOGS.rsplit("/", 1)[-1],
        LEGACY_LOGS_V8.rsplit("/", 1)[-1],
    }


def ablation_level_spec(level: str) -> AblationLevelSpec:
    level = normalize_ablation_level(level)
    for spec in ABLATION_LEVEL_SPECS:
        if spec["level"] == level:
            return spec
    raise KeyError(f"Unknown ablation level '{level}'")


def ablation_feature_mode(level: str) -> str:
    return ablation_level_spec(level)["feature_mode"]


def ablation_model_key(prefix: str, level: str) -> str:
    mode = normalize_feature_mode(ablation_feature_mode(level))
    return mode.replace("problem_", f"{prefix}_")


def legacy_ablation_model_key(prefix: str, level: str) -> str | None:
    level = normalize_ablation_level(level)
    suffix = LEGACY_MODEL_KEY_SUFFIX.get(level)
    return f"{prefix}_{suffix}" if suffix else None


def delta_baseline_for_level(level: str) -> str:
    return ablation_level_spec(level)["delta_baseline"]
