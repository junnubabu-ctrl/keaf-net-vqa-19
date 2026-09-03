from __future__ import annotations

import unittest

from evitrust_vqa.corruption import EvidenceCorruptor
from evitrust_vqa.metrics import (
    area_under_risk_coverage,
    binary_classification,
    brier_score,
    exact_match,
    expected_calibration_error,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    risk_coverage_curve,
    robustness_degradation,
    vqa_soft_accuracy,
)
from evitrust_vqa.schemas import EvidenceStance, SourceType

from helpers import load_evidence


class CorruptionMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = load_evidence()
        self.corruptor = EvidenceCorruptor()

    def test_each_corruption_is_deterministic(self) -> None:
        for kind in sorted(EvidenceCorruptor.SUPPORTED):
            first = self.corruptor.apply(self.evidence, kind=kind, rate=0.5, seed=7)
            second = self.corruptor.apply(self.evidence, kind=kind, rate=0.5, seed=7)
            self.assertEqual(first, second, kind)

    def test_duplicate_adds_unique_ids(self) -> None:
        result = self.corruptor.apply(self.evidence, kind="duplicate", rate=0.5, seed=7)
        ids = [item.evidence_id for item in result.evidence]
        self.assertGreater(len(ids), len(self.evidence))
        self.assertEqual(len(ids), len(set(ids)))

    def test_missing_removes_items(self) -> None:
        result = self.corruptor.apply(self.evidence, kind="missing", rate=0.5, seed=7)
        self.assertLess(len(result.evidence), len(self.evidence))

    def test_contradiction_flips_stance(self) -> None:
        result = self.corruptor.apply(self.evidence, kind="contradictory", rate=1.0, seed=7)
        mapping = {item.evidence_id: item for item in result.evidence}
        self.assertEqual(mapping["ev-001"].stance, EvidenceStance.REFUTE)

    def test_source_swap_reduces_provenance(self) -> None:
        result = self.corruptor.apply(self.evidence, kind="stale_source_swapped", rate=1.0, seed=7)
        self.assertTrue(all(item.source_type == SourceType.UNVERIFIED_WEB for item in result.evidence))
        self.assertTrue(all(item.snapshot == "stale-or-unknown" for item in result.evidence))

    def test_irrelevant_adds_neutral_items(self) -> None:
        result = self.corruptor.apply(self.evidence, kind="irrelevant", rate=0.5, seed=7)
        added = result.evidence[len(self.evidence) :]
        self.assertTrue(added)
        self.assertTrue(all(item.stance == EvidenceStance.NEUTRAL for item in added))

    def test_retrieval_metrics(self) -> None:
        ranked = ["a", "b", "c"]
        self.assertEqual(recall_at_k(ranked, {"b", "x"}, 2), 0.5)
        self.assertEqual(reciprocal_rank(ranked, {"b"}), 0.5)
        self.assertAlmostEqual(ndcg_at_k(ranked, {"b": 2.0, "a": 1.0}, 2), 0.79670758)

    def test_vqa_metrics(self) -> None:
        self.assertEqual(exact_match("New  York", "new york"), 1.0)
        self.assertEqual(vqa_soft_accuracy("cat", ["cat", "cat", "dog"]), 2 / 3)
        self.assertEqual(vqa_soft_accuracy("cat", ["cat"] * 10), 1.0)

    def test_binary_metrics_perfect(self) -> None:
        result = binary_classification([True, False, True, False], [True, False, True, False])
        self.assertEqual((result.precision, result.recall, result.f1, result.mcc), (1.0, 1.0, 1.0, 1.0))

    def test_calibration_metrics(self) -> None:
        confidences = [1.0, 0.0]
        correct = [True, False]
        self.assertEqual(brier_score(confidences, correct), 0.0)
        self.assertEqual(expected_calibration_error(confidences, correct, bins=2), 0.0)

    def test_risk_coverage_and_aurc(self) -> None:
        curve = risk_coverage_curve([0.9, 0.8, 0.7], [True, False, True])
        self.assertEqual(len(curve), 4)
        area = area_under_risk_coverage([0.9, 0.8, 0.7], [True, False, True])
        self.assertGreater(area, 0.0)
        self.assertLess(area, 1.0)

    def test_risk_coverage_groups_equal_confidences(self) -> None:
        curve = risk_coverage_curve([0.9, 0.9, 0.7], [True, False, True])
        self.assertEqual(len(curve), 3)
        self.assertAlmostEqual(curve[1][0], 2 / 3)
        self.assertEqual(curve[1][1], 0.5)

    def test_robustness_degradation(self) -> None:
        result = robustness_degradation(0.8, 0.6)
        self.assertAlmostEqual(result["absolute"], 0.2)
        self.assertAlmostEqual(result["relative"], 0.25)

    def test_invalid_corruption_fails(self) -> None:
        with self.assertRaises(ValueError):
            self.corruptor.apply(self.evidence, kind="unknown", rate=0.5, seed=7)


if __name__ == "__main__":
    unittest.main()
