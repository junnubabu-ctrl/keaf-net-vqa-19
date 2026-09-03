"""Construction of a deterministic signed evidence graph."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .schemas import EvidenceRecord, RelationType, SignedEdge
from .text import jaccard, normalize_text


@dataclass(frozen=True)
class EvidenceGraph:
    node_ids: tuple[str, ...]
    edges: tuple[SignedEdge, ...]

    def neighbors(self, evidence_id: str) -> tuple[SignedEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.source_id == evidence_id or edge.target_id == evidence_id
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "node_ids": list(self.node_ids),
            "edges": [edge.to_dict() for edge in self.edges],
        }


class SignedEvidenceGraphBuilder:
    """Transparent graph builder; replaceable by a frozen NLI model in full runs."""

    def __init__(self, support_overlap: float = 0.35, redundancy_overlap: float = 0.92):
        if not 0.0 <= support_overlap <= redundancy_overlap <= 1.0:
            raise ValueError("overlap thresholds are inconsistent")
        self.support_overlap = support_overlap
        self.redundancy_overlap = redundancy_overlap

    def build(self, evidence: list[EvidenceRecord]) -> EvidenceGraph:
        ids = [item.evidence_id for item in evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique")
        edges: list[SignedEdge] = []
        ordered = sorted(evidence, key=lambda item: item.evidence_id)
        for left, right in combinations(ordered, 2):
            relation = self._classify(left, right)
            if relation is not None:
                edges.append(relation)
        return EvidenceGraph(tuple(item.evidence_id for item in ordered), tuple(edges))

    def _classify(self, left: EvidenceRecord, right: EvidenceRecord) -> SignedEdge | None:
        overlap = jaccard(left.claim, right.claim)
        same_sp = (
            left.subject is not None
            and left.predicate is not None
            and normalize_text(left.subject) == normalize_text(right.subject)
            and normalize_text(left.predicate) == normalize_text(right.predicate)
        )
        different_object = (
            left.object is not None
            and right.object is not None
            and normalize_text(left.object) != normalize_text(right.object)
        )
        if same_sp and different_object:
            return SignedEdge(
                left.evidence_id,
                right.evidence_id,
                RelationType.CONTRADICT,
                max(0.65, overlap),
                "same subject/predicate with different object",
            )
        if normalize_text(left.claim) == normalize_text(right.claim) or overlap >= self.redundancy_overlap:
            return SignedEdge(
                left.evidence_id,
                right.evidence_id,
                RelationType.REDUNDANT,
                max(0.5, overlap),
                "near-duplicate claim",
            )
        same_answer = (
            left.answer is not None
            and right.answer is not None
            and normalize_text(left.answer) == normalize_text(right.answer)
        )
        if same_answer and (overlap >= self.support_overlap or left.stance == right.stance):
            return SignedEdge(
                left.evidence_id,
                right.evidence_id,
                RelationType.SUPPORT,
                max(0.45, overlap),
                "independent claims converge on the same answer",
            )
        return None

