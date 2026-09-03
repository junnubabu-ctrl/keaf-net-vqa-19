from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evitrust_vqa.experiments import run_corruption_suite
from evitrust_vqa.manifests import validate_manifest
from evitrust_vqa.pipeline import InferenceRequest
from evitrust_vqa.statistics import holm_adjust, paired_bootstrap_delta, summarize

from helpers import ROOT, load_evidence


class ManifestStatisticsExperimentTests(unittest.TestCase):
    def test_example_manifest_fails_until_frozen(self) -> None:
        result = validate_manifest(ROOT / "data" / "manifests" / "datasets.example.json")
        self.assertFalse(result.valid)
        self.assertTrue(any("unresolved" in item.message for item in result.findings))

    def test_complete_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps({"schema_version": "1.0", "datasets": [{"name": "fixture"}]}),
                encoding="utf-8",
            )
            self.assertTrue(validate_manifest(path).valid)

    def test_manifest_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": "2.0"}), encoding="utf-8")
            result = validate_manifest(path)
            self.assertFalse(result.valid)

    def test_summary_known_values(self) -> None:
        result = summarize([1.0, 2.0, 3.0])
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.standard_deviation, 1.0)
        self.assertEqual(result.count, 3)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_delta([1, 1, 1], [0, 0, 0], replicates=200, seed=7)
        second = paired_bootstrap_delta([1, 1, 1], [0, 0, 0], replicates=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first.observed_delta, 1.0)

    def test_holm_adjust_preserves_original_order(self) -> None:
        adjusted = holm_adjust([0.04, 0.01, 0.03])
        self.assertEqual(adjusted, [0.06, 0.03, 0.06])

    def test_corruption_suite_has_five_by_three_trials(self) -> None:
        trials = run_corruption_suite(
            request=InferenceRequest("tiny-001", "In which city is the tower shown?"),
            evidence=load_evidence(),
            target_answer="London",
        )
        self.assertEqual(len(trials), 15)
        self.assertEqual({trial.rate for trial in trials}, {0.10, 0.25, 0.50})
        self.assertEqual(len({trial.corruption for trial in trials}), 5)


if __name__ == "__main__":
    unittest.main()

