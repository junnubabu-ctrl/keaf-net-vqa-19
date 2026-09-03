from __future__ import annotations

import inspect
import unittest

from evitrust_vqa.answering import EvidenceVoteAnswerer, InMemoryRetriever
from evitrust_vqa.pipeline import EviTrustPipeline, InferenceRequest
from evitrust_vqa.selective import SelectiveCalibrator

from helpers import load_evidence


class PipelineSelectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = load_evidence()

    def test_pipeline_returns_grounded_london_answer(self) -> None:
        pipeline = EviTrustPipeline(InMemoryRetriever(self.evidence))
        result = pipeline.infer(InferenceRequest("tiny-001", "In which city is the tower shown?"))
        self.assertFalse(result.abstained)
        self.assertEqual(result.answer, "London")
        self.assertEqual(result.candidate.supporting_evidence_ids, ("ev-001", "ev-002"))

    def test_pipeline_is_deterministic_except_timestamp(self) -> None:
        pipeline = EviTrustPipeline(InMemoryRetriever(self.evidence))
        request = InferenceRequest("tiny-001", "In which city is the tower shown?")
        first, second = pipeline.infer(request).to_dict(), pipeline.infer(request).to_dict()
        first["audit"].pop("timestamp_utc")
        second["audit"].pop("timestamp_utc")
        self.assertEqual(first, second)

    def test_pipeline_signature_cannot_receive_gold_answer(self) -> None:
        signature = inspect.signature(EviTrustPipeline.infer)
        self.assertNotIn("gold_answer", signature.parameters)
        self.assertEqual(tuple(signature.parameters), ("self", "request"))

    def test_answerer_uses_only_accepted_evidence(self) -> None:
        pipeline = EviTrustPipeline(InMemoryRetriever(self.evidence))
        result = pipeline.infer(InferenceRequest("tiny-001", "In which city is the tower shown?"))
        self.assertNotIn("ev-003", result.candidate.supporting_evidence_ids)

    def test_empty_evidence_abstains(self) -> None:
        pipeline = EviTrustPipeline(InMemoryRetriever([]))
        result = pipeline.infer(InferenceRequest("empty", "What is shown?"))
        self.assertTrue(result.abstained)
        self.assertIsNone(result.answer)

    def test_selective_fit_chooses_highest_coverage_safe_threshold(self) -> None:
        calibrator = SelectiveCalibrator(target_risk=0.25)
        calibrator.fit([0.9, 0.8, 0.7, 0.6], [True, True, False, True])
        self.assertEqual(calibrator.threshold, 0.6)

    def test_selective_gate_abstains_when_insufficient(self) -> None:
        decision = SelectiveCalibrator(default_threshold=0.5).decide("answer", 0.9, False)
        self.assertTrue(decision.abstained)
        self.assertEqual(decision.reason, "insufficient evidence")

    def test_selective_gate_abstains_below_threshold(self) -> None:
        decision = SelectiveCalibrator(default_threshold=0.8).decide("answer", 0.7, True)
        self.assertTrue(decision.abstained)

    def test_no_safe_validation_prefix_forces_abstention(self) -> None:
        calibrator = SelectiveCalibrator(target_risk=0.0).fit([1.0], [False])
        self.assertGreater(calibrator.threshold, 1.0)
        self.assertTrue(calibrator.decide("answer", 1.0, True).abstained)

    def test_equal_confidence_examples_are_calibrated_as_a_group(self) -> None:
        calibrator = SelectiveCalibrator(target_risk=0.0).fit(
            [0.9, 0.9],
            [True, False],
        )
        self.assertGreater(calibrator.threshold, 1.0)

    def test_vote_answerer_returns_unknown_without_accepted_items(self) -> None:
        answer = EvidenceVoteAnswerer().answer("question", self.evidence, ())
        self.assertEqual(answer.text, "unknown")
        self.assertEqual(answer.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
