"""Smoke tests for the public reproduction bundle (no LLM / EviDiag)."""
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
