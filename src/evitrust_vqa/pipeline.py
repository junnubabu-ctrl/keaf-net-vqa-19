"""End-to-end inference pipeline with audit trace and no ground-truth input."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .answering import Answerer, EvidenceVoteAnswerer, Retriever
from .config import EviTrustConfig
from .graph import EvidenceGraph, SignedEvidenceGraphBuilder
from .schemas import AnswerCandidate, EvidenceDecision, EvidenceRecord, SufficiencyResult
from .selective import SelectiveCalibrator, SelectiveDecision
from .sufficiency import SufficiencyEvaluator
from .verifier import EvidenceVerifier


@dataclass(frozen=True)
class InferenceRequest:
    question_id: str
    question: str
    image_id: str | None = None

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.question.strip():
            raise ValueError("question_id and question are required")


@dataclass(frozen=True)
class InferenceResult:
    question_id: str
    answer: str | None
    abstained: bool
    confidence: float
    graph: EvidenceGraph
    evidence: tuple[EvidenceRecord, ...]
    decisions: tuple[EvidenceDecision, ...]
    sufficiency: SufficiencyResult
    candidate: AnswerCandidate
    selective: SelectiveDecision
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "abstained": self.abstained,
            "confidence": self.confidence,
            "graph": self.graph.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "decisions": [item.to_dict() for item in self.decisions],
            "sufficiency": self.sufficiency.to_dict(),
            "candidate": self.candidate.to_dict(),
            "selective": self.selective.to_dict(),
            "audit": self.audit,
        }


class EviTrustPipeline:
    def __init__(
        self,
        retriever: Retriever,
        *,
        config: EviTrustConfig | None = None,
        graph_builder: SignedEvidenceGraphBuilder | None = None,
        verifier: EvidenceVerifier | None = None,
        sufficiency: SufficiencyEvaluator | None = None,
        answerer: Answerer | None = None,
        selective: SelectiveCalibrator | None = None,
    ):
        self.config = config or EviTrustConfig()
        self.retriever = retriever
        self.graph_builder = graph_builder or SignedEvidenceGraphBuilder()
        self.verifier = verifier or EvidenceVerifier(
            accept_threshold=self.config.evidence_accept_threshold,
            support_threshold=self.config.support_threshold,
            refute_threshold=self.config.refute_threshold,
        )
        self.sufficiency = sufficiency or SufficiencyEvaluator()
        self.answerer = answerer or EvidenceVoteAnswerer()
        self.selective = selective or SelectiveCalibrator(
            target_risk=self.config.target_selective_risk,
            default_threshold=0.60,
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        evidence = self.retriever.retrieve(
            request.question,
            request.image_id,
            self.config.top_k,
        )
        graph = self.graph_builder.build(evidence)
        decisions = self.verifier.verify(evidence, graph)
        sufficiency = self.sufficiency.evaluate(request.question, evidence, decisions)
        candidate = self.answerer.answer(request.question, evidence, decisions)

        accepted = set(sufficiency.accepted_evidence_ids)
        grounded = set(candidate.supporting_evidence_ids)
        grounding = len(accepted & grounded) / len(grounded) if grounded else 0.0
        confidence = (
            0.45 * candidate.confidence
            + 0.35 * sufficiency.score
            + 0.20 * grounding
        )
        confidence = round(max(0.0, min(1.0, confidence)), 8)
        selective = self.selective.decide(candidate.text, confidence, sufficiency.sufficient)
        audit = {
            "schema_version": self.config.schema_version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": self.config.digest(),
            "seed": self.config.seed,
            "grounding_score": round(grounding, 8),
            "retrieved_evidence_ids": [item.evidence_id for item in evidence],
            "accepted_evidence_ids": list(sufficiency.accepted_evidence_ids),
            "note": "No ground-truth answer is accepted by the inference interface.",
        }
        return InferenceResult(
            question_id=request.question_id,
            answer=selective.answer,
            abstained=selective.abstained,
            confidence=confidence,
            graph=graph,
            evidence=tuple(evidence),
            decisions=decisions,
            sufficiency=sufficiency,
            candidate=candidate,
            selective=selective,
            audit=audit,
        )


def request_schema_fields() -> tuple[str, ...]:
    """Exposed for leakage tests and downstream adapter validation."""
    return tuple(InferenceRequest.__dataclass_fields__.keys())
