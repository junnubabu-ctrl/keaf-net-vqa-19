"""Set-level evidence sufficiency and contradiction accounting."""

from __future__ import annotations

from .schemas import EvidenceDecision, EvidenceRecord, EvidenceStance, SufficiencyResult
from .text import tokens


class SufficiencyEvaluator:
    def __init__(self, threshold: float = 0.52):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        decisions: tuple[EvidenceDecision, ...],
    ) -> SufficiencyResult:
        by_id = {item.evidence_id: item for item in evidence}
        accepted = [decision for decision in decisions if decision.accepted]
        support_mass = min(1.0, sum(item.trust for item in accepted) / 2.0)
        refutations = [
            item for item in decisions if item.label == EvidenceStance.REFUTE
        ]
        contradiction_mass = min(1.0, sum(item.trust for item in refutations) / 2.0)
        unique_sources = {by_id[item.evidence_id].source_id for item in accepted}
        source_diversity = min(1.0, len(unique_sources) / 2.0)

        question_tokens = set(tokens(question))
        accepted_tokens: set[str] = set()
        for item in accepted:
            accepted_tokens.update(tokens(by_id[item.evidence_id].claim))
        query_coverage = (
            len(question_tokens & accepted_tokens) / len(question_tokens)
            if question_tokens
            else 0.0
        )
        score = (
            0.45 * support_mass
            + 0.25 * source_diversity
            + 0.20 * query_coverage
            + 0.10 * (1.0 - contradiction_mass)
        )
        score = max(0.0, min(1.0, score))
        return SufficiencyResult(
            score=round(score, 8),
            sufficient=score >= self.threshold and bool(accepted),
            support_mass=round(support_mass, 8),
            contradiction_mass=round(contradiction_mass, 8),
            source_diversity=round(source_diversity, 8),
            query_coverage=round(query_coverage, 8),
            accepted_evidence_ids=tuple(item.evidence_id for item in accepted),
        )

