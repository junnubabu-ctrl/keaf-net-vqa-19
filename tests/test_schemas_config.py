from __future__ import annotations

import unittest

from evitrust_vqa.config import EviTrustConfig
from evitrust_vqa.pipeline import InferenceRequest, request_schema_fields
from evitrust_vqa.schemas import EvidenceRecord, EvidenceStance, SourceType


class SchemaConfigTests(unittest.TestCase):
    def test_evidence_requires_provenance(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceRecord(
                evidence_id="x",
                claim="claim",
                source_id="",
                source_type=SourceType.SYNTHETIC,
                snapshot="fixture",
                retrieval_score=0.5,
                content_score=0.5,
                source_reliability=0.5,
            )

    def test_scores_are_bounded(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceRecord(
                evidence_id="x",
                claim="claim",
                source_id="source",
                source_type=SourceType.SYNTHETIC,
                snapshot="fixture",
                retrieval_score=1.1,
                content_score=0.5,
                source_reliability=0.5,
            )

    def test_from_dict_retains_unknown_metadata(self) -> None:
        item = EvidenceRecord.from_dict(
            {
                "evidence_id": "x",
                "claim": "claim",
                "source_id": "source",
                "source_type": "synthetic",
                "snapshot": "fixture",
                "retrieval_score": 0.5,
                "content_score": 0.5,
                "source_reliability": 0.5,
                "stance": "support",
                "custom": "retained",
            }
        )
        self.assertEqual(item.metadata["custom"], "retained")
        self.assertEqual(item.stance, EvidenceStance.SUPPORT)

    def test_inference_request_has_no_gold_label_field(self) -> None:
        fields = request_schema_fields()
        self.assertNotIn("gold_answer", fields)
        self.assertNotIn("label", fields)
        self.assertEqual(fields, ("question_id", "question", "image_id"))

    def test_request_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            InferenceRequest("id", "")

    def test_config_requires_three_seeds(self) -> None:
        with self.assertRaises(ValueError):
            EviTrustConfig(minimum_training_seeds=2)

    def test_config_digest_is_stable(self) -> None:
        self.assertEqual(EviTrustConfig().digest(), EviTrustConfig().digest())


if __name__ == "__main__":
    unittest.main()

