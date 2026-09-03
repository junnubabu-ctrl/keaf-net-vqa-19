from __future__ import annotations

import unittest

from evitrust_vqa.graph import SignedEvidenceGraphBuilder
from evitrust_vqa.provenance import CalibrationExample, ProvenanceCalibrator
from evitrust_vqa.schemas import EvidenceStance, RelationType, SourceType
from evitrust_vqa.verifier import EvidenceVerifier

from helpers import load_evidence


class GraphVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = load_evidence()
        self.graph = SignedEvidenceGraphBuilder().build(self.evidence)

    def test_graph_nodes_are_sorted_and_unique(self) -> None:
        self.assertEqual(self.graph.node_ids, tuple(sorted(self.graph.node_ids)))
        self.assertEqual(len(self.graph.node_ids), len(set(self.graph.node_ids)))

    def test_conflict_edge_is_detected(self) -> None:
        conflicts = [edge for edge in self.graph.edges if edge.relation == RelationType.CONTRADICT]
        pairs = [{edge.source_id, edge.target_id} for edge in conflicts]
        self.assertIn({"ev-001", "ev-003"}, pairs)

    def test_support_edge_is_detected(self) -> None:
        supports = [edge for edge in self.graph.edges if edge.relation == RelationType.SUPPORT]
        self.assertTrue(any({edge.source_id, edge.target_id} == {"ev-001", "ev-002"} for edge in supports))

    def test_graph_is_deterministic_under_input_permutation(self) -> None:
        reversed_graph = SignedEvidenceGraphBuilder().build(list(reversed(self.evidence)))
        self.assertEqual(self.graph, reversed_graph)

    def test_duplicate_ids_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            SignedEvidenceGraphBuilder().build([self.evidence[0], self.evidence[0]])

    def test_high_quality_support_is_accepted(self) -> None:
        decisions = EvidenceVerifier().verify(self.evidence, self.graph)
        indexed = {item.evidence_id: item for item in decisions}
        self.assertTrue(indexed["ev-001"].accepted)
        self.assertEqual(indexed["ev-001"].label, EvidenceStance.SUPPORT)

    def test_low_reliability_conflict_is_not_accepted(self) -> None:
        decisions = EvidenceVerifier().verify(self.evidence, self.graph)
        indexed = {item.evidence_id: item for item in decisions}
        self.assertFalse(indexed["ev-003"].accepted)
        self.assertEqual(indexed["ev-003"].label, EvidenceStance.REFUTE)

    def test_provenance_prior_orders_known_and_unverified_sources(self) -> None:
        calibrator = ProvenanceCalibrator()
        self.assertGreater(calibrator.prior(self.evidence[0]), calibrator.prior(self.evidence[2]))

    def test_calibrator_fit_changes_parameters(self) -> None:
        calibrator = ProvenanceCalibrator()
        before = (calibrator.bias, calibrator.weight_reliability, calibrator.weight_prior)
        calibrator.fit(
            [
                CalibrationExample(0.9, SourceType.KNOWLEDGE_GRAPH, True),
                CalibrationExample(0.2, SourceType.UNVERIFIED_WEB, False),
            ],
            steps=20,
        )
        after = (calibrator.bias, calibrator.weight_reliability, calibrator.weight_prior)
        self.assertNotEqual(before, after)
        self.assertTrue(calibrator.fitted)


if __name__ == "__main__":
    unittest.main()

