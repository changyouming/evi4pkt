#!/usr/bin/env python3
"""Build a minimal public reproduction bundle (no LLM artifacts, no private data)."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Manifest: core evipkt (8-backbone + rule-based evidence) ---
CORE_EVIPKT = [
    "__init__.py",
    "dataset.py",
    "feature_modes.py",
    "train.py",
    "train_sequence.py",
    "sequence_dataset.py",
    "kt_runner_common.py",
    "labels.py",
    "device.py",
    "plugplay_evidence.py",
    "evidence_v8.py",
    "evidence_adapter.py",
    "code_evidence.py",
    "error_evidence.py",
    "error_evidence_filter.py",
    "process_evidence.py",
    "error_mechanism.py",
    "code_misused_rules.py",
    "preprocess.py",
    "csedm_io.py",
    "kc_catalog.py",
    "q_matrix.py",
    "dkt.py",
    "runner.py",
    "dkvmn.py",
    "dkvmn_runner.py",
    "sakt.py",
    "sakt_runner.py",
    "akt.py",
    "akt_runner.py",
    "qdkt.py",
    "qdkt_runner.py",
    "qikt.py",
    "qikt_runner.py",
    "simplekt.py",
    "simplekt_runner.py",
    "sparsekt.py",
    "sparsekt_runner.py",
]

FULL_EXTRA_EVIPKT = [
    "code2vec_features.py",
    "codebert_features.py",
    "iice_lite.py",
    "iice_lite_runner.py",
    "iice_lite_dataset.py",
    "kcgen_kt_lite.py",
    "kcgen_kt_lite_runner.py",
]

CORE_SCRIPTS = [
    "preprocess_csedm.py",
    "enrich_framework_logs_process.py",
    "enrich_framework_logs_error_mechanism.py",
    "run_ablation_ladder.py",
    "build_ablation_plan_table.py",
    "build_eight_backbone_plugplay_table.py",
    "build_repro_bundle.py",
]

FULL_EXTRA_SCRIPTS = [
    "build_code2vec_cache_f19.py",
    "build_codebert_cache_f19.py",
    "run_classic_pkt_f19_s19.py",
    "build_classic_pkt_summary.py",
]

ROOT_FILES = ["pyproject.toml"]

REPRO_CANONICAL_LOGS = "data/processed/framework_logs_first_process_mechanism.jsonl"
REPRO_PIPELINE_ID = "evipkt_rule_process_mechanism"

PRIVACY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"/home/Jerry-deepin-codex/Desktop/Evi4PKT"),
    re.compile(r"EVI4PKT_LLM_API_KEY\s*="),
    re.compile(r"OPENAI_API_KEY\s*="),
]

LLM_PATH_DENY = re.compile(
    r"(llm_client|code_misused_llm|code_evidence_llm|error_evidence_llm|misused_llm|enrich_framework_logs_.*llm)",
    re.I,
)

LLM_JSON_KEYS = frozenset({"llm", "misused_v8", "code_evidence_llm", "error_evidence_llm"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "Evi4PKT-repro",
    )
    p.add_argument("--tier", choices=("core", "full"), default="full")
    p.add_argument("--force", action="store_true", help="Remove existing output directory first.")
    return p.parse_args()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_manifest(names: list[str], src_dir: Path, dst_dir: Path) -> None:
    for name in names:
        src = src_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        _copy_file(src, dst_dir / name)


def _patch_plugplay(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from .code_misused_llm import (",
        "from .code_misused_rules import (",
    )
    path.write_text(text, encoding="utf-8")


def _patch_error_evidence(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from .error_evidence_llm import ERROR_EVIDENCE_LLM_SOURCE\n",
        "",
    )
    text = text.replace(
        "ERROR_EVIDENCE_SOURCE = ERROR_EVIDENCE_LLM_SOURCE",
        'ERROR_EVIDENCE_SOURCE = "rule_mechanism"',
    )
    path.write_text(text, encoding="utf-8")


def _patch_code_evidence(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "        from .code_evidence_llm import CODE_EVIDENCE_LLM_SOURCE\n\n        return CODE_EVIDENCE_LLM_SOURCE",
        '        return "llm_not_available_in_repro"',
    )
    old = """    if backend == "llm":
        from .code_evidence_llm import build_code_evidence_llm

        return list(
            build_code_evidence_llm(
                code,
                q_kc=q_kc,
                problem_prompt=str(pt.get("prompt", "") if isinstance(pt, dict) else ""),
                catalog=catalog,
            )["vector"]
        )
"""
    if old in text:
        text = text.replace(
            old,
            '    if backend == "llm":\n        raise RuntimeError("LLM code evidence is not available in the reproduction bundle.")\n',
        )
    path.write_text(text, encoding="utf-8")


def _patch_feature_modes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'CANONICAL_LOGS = "data/processed/framework_logs_first_llm_error_process_misused.jsonl"',
        f'CANONICAL_LOGS = "{REPRO_CANONICAL_LOGS}"',
    )
    text = text.replace(
        'LEGACY_LOGS_V8 = "data/processed/framework_logs_first_llm_error_process_misused_v8.jsonl"',
        f'LEGACY_LOGS_V8 = "{REPRO_CANONICAL_LOGS}"',
    )
    text = text.replace(
        'PIPELINE_ID = "evipkt_misused_mechanism"',
        f'PIPELINE_ID = "{REPRO_PIPELINE_ID}"',
    )
    path.write_text(text, encoding="utf-8")


def _patch_enrich_defaults(path: Path, *, in_default: str, out_suffix: str | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    if 'default="data/processed/framework_logs_first_llm' in text:
        text = re.sub(
            r'default="data/processed/framework_logs_first[^"]*\.jsonl"',
            f'default="{in_default}"',
            text,
            count=1,
        )
    if out_suffix and "default=" in text:
        pass  # scripts derive out path from input stem
    path.write_text(text, encoding="utf-8")


def _patch_repro_sources(out_root: Path) -> None:
    evi = out_root / "evipkt"
    _patch_plugplay(evi / "plugplay_evidence.py")
    _patch_error_evidence(evi / "error_evidence.py")
    _patch_code_evidence(evi / "code_evidence.py")
    _patch_feature_modes(evi / "feature_modes.py")
    _patch_enrich_defaults(
        out_root / "scripts/enrich_framework_logs_process.py",
        in_default="data/processed/framework_logs_first.jsonl",
    )
    _patch_enrich_defaults(
        out_root / "scripts/enrich_framework_logs_error_mechanism.py",
        in_default="data/processed/framework_logs_first_process.jsonl",
    )


def _strip_llm_from_sample(record: dict) -> dict:
    out = json.loads(json.dumps(record))
    ce = out.get("code_evidence")
    if isinstance(ce, dict):
        ce.pop("llm", None)
        ce.pop("misused_v8", None)
        block = ce.get("misused")
        if isinstance(block, dict) and block.get("source", "").startswith("llm"):
            ce.pop("misused", None)
    ee = out.get("error_evidence")
    if isinstance(ee, dict):
        ee.pop("llm", None)
    return out


def _write_kc_catalog_md(path: Path) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from evipkt.kc_catalog import DEFAULT_KC_CATALOG, PROMPT_CSV_KC_COLUMNS

    lines = [
        "# KC Catalog (18 concepts)",
        "",
        "Closed catalog aligned with expert Q-matrix columns in `problem_prompts.csv`.",
        "",
        "| # | KC name | Prompt CSV column |",
        "|---|---------|-------------------|",
    ]
    for i, name in enumerate(DEFAULT_KC_CATALOG, start=1):
        col = next((k for k, v in PROMPT_CSV_KC_COLUMNS.items() if v == name), "—")
        lines.append(f"| {i} | {name} | {col} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_evidence_schema_md(path: Path) -> None:
    path.write_text(
        """# Framework Log Evidence Schema (rule-based channels)

One JSON object per line (`Run.Program` interaction). Fields used by the public reproduction bundle:

| Field | Role |
|-------|------|
| `subject_id` | Hashed student identifier from CSEDM |
| `problem_id` | CodeWorkout problem ID |
| `pkt_label` | Binary correctness (1 iff all unit tests pass) |
| `programming_task` | Task prompt + expert `q_kc` + `kc_catalog` |
| `student_code` | Submitted Java source |
| `code_issues` | Outcome typing (`compile_error`, `partial_pass`, …) |
| `process_evidence` | Pre-attempt KC exposure/success (rule-derived) |
| `error_evidence.mechanism_v8` | Compile-error mechanism tags M1–M12 (rule-derived) |

**Fair protocol:** at step *t*, prediction uses history through *t−1* plus Task/Process evidence available before the attempt; compile mechanism attaches to the current step for state updates at *t+1*.

**Not included in this bundle:** LLM-generated Code/Error alignment artifacts (`code_evidence.llm`, `misused_v8`, `error_evidence.llm`). The paper's Full Evi4PKT uses additional LLM channels that are not redistributed.
""",
        encoding="utf-8",
    )


def _write_filtering_rules_md(path: Path) -> None:
    path.write_text(
        """# Filtering and Split Rules

- **Submission mode:** `first` — earliest `Run.Program` per (student, problem) for main experiments.
- **Label:** `pkt_label = 1` iff unit-test score ≥ 1.0 (all tests pass).
- **Split:** student-level 80% train / 10% validation / 10% test within each semester cohort.
- **Seeds:** 10 independent splits (seeds 0–9).
- **Training:** 8 epochs, batch size 64, checkpoint selected by validation AUC.
""",
        encoding="utf-8",
    )


def _write_data_access_md(path: Path) -> None:
    path.write_text(
        """# CSEDM / CodeWorkout Data Access

The raw CSEDM/CodeWorkout F19 and S19 programming-learning datasets are available from the original dataset provider, subject to its access conditions.

After obtaining the releases:

1. Place F19 ProgSnap2 under `data/raw/F19/` (e.g. `MainTable.csv`, `CodeStates.csv` under `All/`).
2. Place S19 under `data/raw/S19/` (full tier only).
3. Run `bash scripts/enrich_rule_based.sh f19` (and `s19` for full tier).

This repository does **not** ship raw CSEDM files, processed framework logs, LLM caches, or model checkpoints.
""",
        encoding="utf-8",
    )


def _write_reproduce_md(out_root: Path, tier: str) -> None:
    pkt_section = ""
    if tier == "full":
        pkt_section = """
## 6. Specialized PKT baselines (optional)

After building F19/S19 rule-based logs and code2vec/codebert caches:

```bash
python scripts/run_classic_pkt_f19_s19.py --cohorts f19 s19 --resume
python scripts/build_classic_pkt_summary.py
```
"""

    text = f"""# Evi4PKT — Minimal Reproduction Guide

Public bundle tier: **{tier}**. This package contains code and supplementary metadata only.

## Scope

- **Included:** rule-based preprocessing (Task Q, Process, compile Mechanism), eight KT backbones, plug-and-play training scripts, aggregation tables.
- **Excluded:** raw CSEDM data, processed logs, LLM-generated evidence, API keys, paper drafts, trained checkpoints.

The paper's Full Evi4PKT includes LLM Code/Error channels that are **not redistributed**. Here, **Full (repro)** = `problem_plus_q_process_code_mechanism` trained on logs enriched through Process + Mechanism only (Code channel reads zeros without LLM enrichment).

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[repro]"
pytest tests/ -q
```

## 2. Obtain CSEDM data

See [supplementary/DATA_ACCESS.md](supplementary/DATA_ACCESS.md).

## 3. Build framework logs (rule-based)

```bash
# F19
bash scripts/enrich_rule_based.sh f19

# S19 (full tier)
bash scripts/enrich_rule_based.sh s19
```

Output: `{REPRO_CANONICAL_LOGS}` (F19) or `data/processed_s19/framework_logs_first_process_mechanism.jsonl` (S19).

## 4. Train eight backbones (TABLE II backbone vs Full repro)

```bash
bash scripts/run_repro_eight_backbone.sh
```

Or manually:

```bash
python scripts/run_ablation_ladder.py \\
  --levels v0 v5 \\
  --backbones DKT DKVMN SAKT AKT qDKT QIKT SimpleKT SparseKT \\
  --seeds 0 1 2 3 4 5 6 7 8 9 \\
  --epochs 8 --batch-size 64 --device cuda --resume
```

- **v0 (Backbone):** `problem_onehot`
- **v5 (Full repro):** `problem_plus_q_process_code_mechanism` on rule-enriched logs

## 5. Summarize results

```bash
python scripts/build_eight_backbone_plugplay_table.py
python scripts/build_ablation_plan_table.py
```
{pkt_section}
## References

- [docs/experiment_protocol.md](docs/experiment_protocol.md)
- [docs/ablation_ladder.md](docs/ablation_ladder.md)
- [supplementary/evidence_schema.md](supplementary/evidence_schema.md)
"""
    (out_root / "REPRODUCE.md").write_text(text, encoding="utf-8")


def _write_docs(out_root: Path, tier: str) -> None:
    docs = out_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    (docs / "experiment_protocol.md").write_text(
        """# Experiment Protocol (reproduction bundle)

- **Cohorts:** F19 (primary); S19 (full tier).
- **Split:** student 80/10/10, seeds 0–9.
- **Backbones:** DKT, DKVMN, SAKT, AKT, qDKT, QIKT, SimpleKT, SparseKT.
- **Backbone variant:** `problem_onehot` (v0).
- **Full variant (repro):** v5 feature mode on rule-enriched logs (Q + Process + Mechanism; no LLM Code/Error redistribution).
- **Metrics:** test AUC, ACC, F1 (mean ± std over seeds).
""",
        encoding="utf-8",
    )

    (docs / "ablation_ladder.md").write_text(
        """# Evidence Ladder (public bundle)

| Level | Feature mode | Evidence |
|-------|--------------|----------|
| v0 | problem_onehot | Problem ID only |
| v1 | problem_plus_q | + expert Q-matrix |
| v2 | problem_plus_q_process | + Process |
| v5 | problem_plus_q_process_code_mechanism | + Mechanism (Full repro) |

Levels v3–v4 rely on Code evidence enriched with LLM labels and are omitted from this bundle.
""",
        encoding="utf-8",
    )

    if tier == "full":
        s19_readme = PROJECT_ROOT / "data/processed_s19/README.md"
        if s19_readme.is_file():
            _copy_file(s19_readme, docs / "s19_pipeline.md")


def _write_supplementary(out_root: Path) -> None:
    sup = out_root / "supplementary"
    sup.mkdir(parents=True, exist_ok=True)
    meta = PROJECT_ROOT / "data/metadata"
    _copy_file(meta / "q_matrix.csv", sup / "q_matrix.csv")
    _copy_file(meta / "problem_prompts.csv", sup / "problem_prompts.csv")
    _write_kc_catalog_md(sup / "kc_catalog.md")
    _write_evidence_schema_md(sup / "evidence_schema.md")
    _write_filtering_rules_md(sup / "filtering_rules.md")
    _write_data_access_md(sup / "DATA_ACCESS.md")

    sample_src = PROJECT_ROOT / "docs/paper_framework_log_sample.json"
    if sample_src.is_file():
        raw = json.loads(sample_src.read_text(encoding="utf-8"))
        cleaned = _strip_llm_from_sample(raw)
        (sup / "framework_log_sample.json").write_text(
            json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _write_gitignore(out_root: Path) -> None:
    (out_root / ".gitignore").write_text(
        """.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.env
.env.*
runs/
data/raw/
data/processed/
data/processed_s19/
data/processed_all/
*.pt
*.ckpt
llm_*cache*
""",
        encoding="utf-8",
    )


def _write_pyproject(out_root: Path) -> None:
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if "[project.optional-dependencies]" not in text:
        text = text.rstrip() + """

[project.optional-dependencies]
repro = ["scipy>=1.11"]
pkt = ["transformers>=4.40", "torch>=2.0"]
"""
    (out_root / "pyproject.toml").write_text(text, encoding="utf-8")


def _write_repro_tests(out_root: Path, tier: str) -> None:
    tests = out_root / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test_repro_smoke.py").write_text(
        '''"""Smoke tests for the public reproduction bundle (no LLM / EviDiag)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from evipkt.code_evidence import build_code_evidence
from evipkt.code_misused_rules import resolve_code_misused_kc
from evipkt.dataset import build_dkt_samples, collate_dkt_batch
from evipkt.dkt import DKT
from evipkt.error_mechanism import attach_mechanism_to_record, build_error_mechanism_evidence
from evipkt.feature_modes import CANONICAL_LOGS
from evipkt.plugplay_evidence import code_evidence_vector, compile_mechanism_vector
from evipkt.preprocess import default_preprocess_config, run_preprocess
from evipkt.process_evidence import attach_process_evidence_to_records


class ReproSmokeTests(unittest.TestCase):
    def test_canonical_logs_path_is_rule_based(self):
        self.assertIn("process_mechanism", CANONICAL_LOGS)
        self.assertNotIn("llm", CANONICAL_LOGS)

    def test_preprocess_first_writes_rule_based_records(self):
        root = Path(__file__).resolve().parents[1]
        cfg = default_preprocess_config(root, submission_mode="first")
        if not (cfg.csedm_root / "Data" / "MainTable.csv").exists():
            self.skipTest("CSEDM raw data not installed (see supplementary/DATA_ACCESS.md)")
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_preprocess(cfg, Path(tmp))
            self.assertGreater(summary["total_interactions"], 0)
            log_path = Path(tmp) / "framework_logs_first.jsonl"
            first = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["code_evidence"]["source"], "rule_based")

    def test_mechanism_and_process_pipeline(self):
        records = attach_process_evidence_to_records(
            [
                {
                    "subject_id": "s1",
                    "problem_id": 1,
                    "pkt_label": 1,
                    "trajectory": {"student_timestep": 0},
                    "programming_task": {"q_kc": [1.0, 0.0], "kc_catalog": ["IfElse", "For"]},
                }
            ]
        )
        compile_rec = {
            **records[0],
            "code_issues": {
                "outcome_type": "compile_error",
                "has_compile_error": True,
                "score": 0.0,
                "issues": [{"type": "compile_error", "message": "error: ';' expected"}],
            },
        }
        out = attach_mechanism_to_record(compile_rec)
        mech = build_error_mechanism_evidence(out)
        self.assertTrue(mech["eligible"])
        self.assertGreater(sum(compile_mechanism_vector(out)), 0.0)

    def test_plugplay_reads_pre_enriched_misused_block(self):
        catalog = ["IfElse", "LogicAndNotOr", "LogicCompareNum"]
        record = {
            "student_code": {"code": "if (a+b >= 10 || a+b <= 19) return 20;"},
            "programming_task": {"q_kc": [1.0, 1.0, 1.0], "kc_catalog": catalog},
            "code_evidence": {
                "misused": {
                    "eligible": True,
                    "missing_kc": ["LogicCompareNum"],
                    "misused_kc": ["IfElse"],
                }
            },
        }
        _, misused_names, _, _ = resolve_code_misused_kc(record, catalog=catalog)
        self.assertIn("LogicAndNotOr", misused_names)
        vec = code_evidence_vector(record, catalog=catalog)
        self.assertEqual(len(vec), 6)

    def test_dkt_forward(self):
        model = DKT(input_dim=6, target_dim=4, hidden_dim=8)
        batch = [
            (
                torch.zeros(1, 6),
                torch.tensor(1),
                torch.tensor([1.0, 0.0, 1.0, 0.0]),
                torch.tensor(1.0),
            )
        ]
        x, lengths, target, _ = collate_dkt_batch(batch)
        logits = model(x, lengths, target)
        self.assertEqual(logits.shape, (1,))

    def test_problem_plus_q_samples(self):
        q = [1.0, 0.0]
        records = [
            {
                "subject_id": "s1",
                "problem_id": 1,
                "pkt_label": 0,
                "trajectory": {"student_timestep": 0},
                "programming_task": {"q_kc": q},
            },
            {
                "subject_id": "s1",
                "problem_id": 2,
                "pkt_label": 1,
                "trajectory": {"student_timestep": 1},
                "programming_task": {"q_kc": [0.0, 1.0]},
            },
        ]
        samples = build_dkt_samples(
            records,
            ["s1"],
            {1: 0, 2: 1},
            problem_q_map={1: q, 2: [0.0, 1.0]},
            feature_mode="problem_plus_q",
        )
        self.assertEqual(len(samples), 1)


if __name__ == "__main__":
    unittest.main()
''',
        encoding="utf-8",
    )
    if tier == "full":
        _copy_file(
            PROJECT_ROOT / "tests/test_code2vec_features.py",
            tests / "test_code2vec_features.py",
        )


def _write_shell_scripts(out_root: Path) -> None:
    (out_root / "scripts/enrich_rule_based.sh").write_text(
        """#!/usr/bin/env bash
# Rule-based enrichment: preprocess -> +Process -> +Mechanism (no LLM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
COHORT="${1:-f19}"

if [[ "$COHORT" == "f19" ]]; then
  RAW="data/raw/F19/All"
  OUT="data/processed"
elif [[ "$COHORT" == "s19" ]]; then
  RAW="data/raw/S19/All"
  OUT="data/processed_s19"
else
  echo "Usage: $0 {f19|s19}" >&2
  exit 1
fi

mkdir -p "$OUT"
$PY scripts/preprocess_csedm.py \\
  --csedm-root "$RAW" \\
  --prompts-csv data/metadata/problem_prompts.csv \\
  --submission-mode first \\
  --out-dir "$OUT"

$PY scripts/enrich_framework_logs_process.py \\
  --in-path "$OUT/framework_logs_first.jsonl"

$PY scripts/enrich_framework_logs_error_mechanism.py \\
  --in-path "$OUT/framework_logs_first_process.jsonl" \\
  --out-path "$OUT/framework_logs_first_process_mechanism.jsonl"

echo "Wrote: $OUT/framework_logs_first_process_mechanism.jsonl"
""",
        encoding="utf-8",
    )
    (out_root / "scripts/enrich_rule_based.sh").chmod(0o755)

    (out_root / "scripts/run_repro_eight_backbone.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
SEEDS="0 1 2 3 4 5 6 7 8 9"
BACKBONES="DKT DKVMN SAKT AKT qDKT QIKT SimpleKT SparseKT"

mkdir -p runs
$PY scripts/run_ablation_ladder.py \\
  --levels v0 v5 \\
  --backbones $BACKBONES \\
  --seeds $SEEDS \\
  --epochs 8 \\
  --batch-size 64 \\
  --device cuda \\
  --resume
""",
        encoding="utf-8",
    )
    (out_root / "scripts/run_repro_eight_backbone.sh").chmod(0o755)

    if (out_root / "scripts/run_classic_pkt_f19_s19.py").exists():
        (out_root / "scripts/run_repro_s19_preprocess.sh").write_text(
            """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/enrich_rule_based.sh f19
bash scripts/enrich_rule_based.sh s19
""",
            encoding="utf-8",
        )
        (out_root / "scripts/run_repro_s19_preprocess.sh").chmod(0o755)


def privacy_scan(out_root: Path) -> list[str]:
    issues: list[str] = []
    for path in out_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "build_repro_bundle.py":
            continue
        if path.suffix.lower() in {".pt", ".ckpt", ".docx", ".docm"}:
            issues.append(f"forbidden file type: {path.relative_to(out_root)}")
            continue
        if LLM_PATH_DENY.search(str(path.relative_to(out_root))):
            issues.append(f"LLM-related path in bundle: {path.relative_to(out_root)}")
            continue
        if path.suffix.lower() in {".jsonl", ".env"}:
            issues.append(f"data/secret file should not be bundled: {path.relative_to(out_root)}")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in PRIVACY_PATTERNS:
            if pat.search(text):
                issues.append(f"privacy pattern {pat.pattern!r} in {path.relative_to(out_root)}")
    return issues


def build_bundle(out_root: Path, tier: str, force: bool) -> None:
    if out_root.exists():
        if force:
            shutil.rmtree(out_root)
        else:
            raise SystemExit(f"Output exists: {out_root} (use --force)")
    out_root.mkdir(parents=True)

    evipkt_names = list(CORE_EVIPKT)
    script_names = list(CORE_SCRIPTS)
    if tier == "full":
        evipkt_names.extend(FULL_EXTRA_EVIPKT)
        script_names.extend(FULL_EXTRA_SCRIPTS)

    _copy_manifest(evipkt_names, PROJECT_ROOT / "evipkt", out_root / "evipkt")
    _copy_manifest(script_names, PROJECT_ROOT / "scripts", out_root / "scripts")
    for name in ROOT_FILES:
        _copy_file(PROJECT_ROOT / name, out_root / name)
    _copy_file(PROJECT_ROOT / "data/metadata/q_matrix.csv", out_root / "data/metadata/q_matrix.csv")
    _copy_file(
        PROJECT_ROOT / "data/metadata/problem_prompts.csv",
        out_root / "data/metadata/problem_prompts.csv",
    )
    _write_repro_tests(out_root, tier)

    _write_pyproject(out_root)
    _write_gitignore(out_root)
    _patch_repro_sources(out_root)
    _write_shell_scripts(out_root)
    _write_supplementary(out_root)
    _write_docs(out_root, tier)
    _write_reproduce_md(out_root, tier)

    issues = privacy_scan(out_root)
    if issues:
        report = out_root / "PRIVACY_SCAN_ISSUES.txt"
        report.write_text("\n".join(issues) + "\n", encoding="utf-8")
        raise SystemExit(f"Privacy scan failed ({len(issues)} issues). See {report}")

    print(f"Built reproduction bundle: {out_root} (tier={tier})")
    print(f"  evipkt modules: {len(evipkt_names)}")
    print(f"  scripts: {len(script_names)}")
    print("  Privacy scan: OK")


def main() -> None:
    args = parse_args()
    build_bundle(args.out.resolve(), args.tier, args.force)


if __name__ == "__main__":
    main()
