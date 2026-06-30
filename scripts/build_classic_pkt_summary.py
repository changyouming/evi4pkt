#!/usr/bin/env python3
"""Merge F19 + S19 classic PKT comparison JSON into one summary table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/classic_pkt_f19_s19"
METHODS = ("code_dkt", "pdkt_lite", "dkt_codebert", "kcgen_kt_lite")
LABELS = {
    "code_dkt": "Code-DKT",
    "pdkt_lite": "PDKT-lite",
    "dkt_codebert": "DKT+CodeBERT",
    "kcgen_kt_lite": "KCGen-KT-lite",
}


def main() -> None:
    f19 = json.loads((OUT / "comparison_f19_seeds_0_1_2_3_4_5_6_7_8_9.json").read_text())
    s19 = json.loads((OUT / "comparison_s19_seeds_0_1_2_3_4_5_6_7_8_9.json").read_text())
    merged = {
        "protocol": f19["protocol"],
        "methods": f19["methods"],
        "f19": f19["cohorts"]["f19"]["summary"],
        "s19": s19["cohorts"]["s19"]["summary"],
    }
    lines = [
        "# Classic PKT Baselines — F19 & S19 (10 seeds, first-attempt, 80/10/10)",
        "",
        "| Method | F19 AUC | S19 AUC | F19 ACC | S19 ACC | F19 F1 | S19 F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in METHODS:
        f = merged["f19"][m]["metrics"]
        s = merged["s19"][m]["metrics"]
        lines.append(
            f"| {LABELS[m]} | {f['auc']['mean']:.4f} ± {f['auc']['std']:.4f} "
            f"| {s['auc']['mean']:.4f} ± {s['auc']['std']:.4f} "
            f"| {f['acc']['mean']:.4f} ± {f['acc']['std']:.4f} "
            f"| {s['acc']['mean']:.4f} ± {s['acc']['std']:.4f} "
            f"| {f['f1']['mean']:.4f} ± {f['f1']['std']:.4f} "
            f"| {s['f1']['mean']:.4f} ± {s['f1']['std']:.4f} |"
        )
    md = "\n".join(lines) + "\n"
    (OUT / "comparison_f19_s19_summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    (OUT / "comparison_f19_s19_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved: {OUT / 'comparison_f19_s19_summary.json'}")


if __name__ == "__main__":
    main()
