"""Evidence-weighted reference answerer and in-memory retriever."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from .schemas import AnswerCandidate, EvidenceDecision, EvidenceRecord
from .text import normalize_text, tokens


class Retriever(Protocol):
    def retrieve(self, question: str, image_id: str | None, top_k: int) -> list[EvidenceRecord]: ...


class Answerer(Protocol):
    def answer(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        decisions: tuple[EvidenceDecision, ...],
    ) -> AnswerCandidate: ...


class InMemoryRetriever:
    """Deterministic fixture/research adapter; production retrievers implement the same protocol."""

    def __init__(self, evidence: list[EvidenceRecord]):
        self.evidence = list(evidence)

    def retrieve(self, question: str, image_id: str | None, top_k: int) -> list[EvidenceRecord]:
        del question, image_id
        return sorted(
            self.evidence,
            key=lambda item: (-item.retrieval_score, item.evidence_id),
        )[:top_k]


class EvidenceVoteAnswerer:
    """Auditable reference answerer that votes only with accepted supportive evidence."""

    def answer(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        decisions: tuple[EvidenceDecision, ...],
    ) -> AnswerCandidate:
        del question
        by_id = {item.evidence_id: item for item in evidence}
        votes: dict[str, float] = defaultdict(float)
        display: dict[str, str] = {}
        support_ids: dict[str, list[str]] = defaultdict(list)
        for decision in decisions:
            item = by_id[decision.evidence_id]
            if not decision.accepted or not item.answer:
                continue
            key = normalize_text(item.answer)
            if not key:
                continue
            votes[key] += decision.trust
            display.setdefault(key, item.answer.strip())
            support_ids[key].append(item.evidence_id)
        if not votes:
            return AnswerCandidate("unknown", 0.0, (), 0.0)
        ranked = sorted(votes.items(), key=lambda pair: (-pair[1], pair[0]))
        winner, winner_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(votes.values())
        confidence = winner_score / total if total else 0.0
        margin = (winner_score - runner_up) / total if total else 0.0
        return AnswerCandidate(
            text=display[winner],
            confidence=round(confidence, 8),
            supporting_evidence_ids=tuple(sorted(support_ids[winner])),
            vote_margin=round(margin, 8),
        )


class RelevanceOnlyAnswerer:
    """Ablation baseline that ignores provenance and graph decisions."""

    def answer_from_retrieval(self, evidence: list[EvidenceRecord]) -> AnswerCandidate:
        candidates = [item for item in evidence if item.answer]
        if not candidates:
            return AnswerCandidate("unknown", 0.0, (), 0.0)
        best = max(candidates, key=lambda item: (item.retrieval_score, item.evidence_id))
        confidence = best.retrieval_score
        return AnswerCandidate(best.answer or "unknown", confidence, (best.evidence_id,), confidence)


def query_overlap(question: str, claim: str) -> float:
    q = set(tokens(question))
    c = set(tokens(claim))
    return len(q & c) / len(q) if q else 0.0

