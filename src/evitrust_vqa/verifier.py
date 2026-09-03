"""Provenance-aware evidence verification over a signed graph."""

from __future__ import annotations

from .graph import EvidenceGraph
from .provenance import ProvenanceCalibrator
from .schemas import EvidenceDecision, EvidenceRecord, EvidenceStance, RelationType


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class EvidenceVerifier:
    def __init__(
        self,
        calibrator: ProvenanceCalibrator | None = None,
        *,
        accept_threshold: float = 0.55,
        support_threshold: float = 0.15,
        refute_threshold: float = -0.15,
        propagation_strength: float = 0.18,
    ):
        self.calibrator = calibrator or ProvenanceCalibrator()
        self.accept_threshold = accept_threshold
        self.support_threshold = support_threshold
        self.refute_threshold = refute_threshold
        self.propagation_strength = propagation_strength

    def verify(
        self, evidence: list[EvidenceRecord], graph: EvidenceGraph
    ) -> tuple[EvidenceDecision, ...]:
        by_id = {item.evidence_id: item for item in evidence}
        base: dict[str, float] = {}
        priors: dict[str, float] = {}
        for item in evidence:
            prior = self.calibrator.prior(item)
            priors[item.evidence_id] = prior
            trust = 0.40 * item.retrieval_score + 0.35 * item.content_score + 0.25 * prior
            base[item.evidence_id] = trust * item.stance.sign

        decisions: list[EvidenceDecision] = []
        for evidence_id in sorted(by_id):
            item = by_id[evidence_id]
            adjustment = 0.0
            reasons: list[str] = []
            for edge in graph.neighbors(evidence_id):
                other_id = edge.target_id if edge.source_id == evidence_id else edge.source_id
                other = by_id[other_id]
                if edge.relation == RelationType.REDUNDANT:
                    adjustment -= self.propagation_strength * edge.weight * 0.15
                    reasons.append(f"redundancy:{other_id}")
                    continue
                direction = edge.relation.sign
                adjustment += (
                    self.propagation_strength
                    * edge.weight
                    * direction
                    * base[other_id]
                    * max(0.25, other.source_reliability)
                )
                reasons.append(f"{edge.relation.value}:{other_id}")
            signed_score = max(-1.0, min(1.0, base[evidence_id] + adjustment))
            if signed_score >= self.support_threshold:
                label = EvidenceStance.SUPPORT
            elif signed_score <= self.refute_threshold:
                label = EvidenceStance.REFUTE
            else:
                label = EvidenceStance.NEUTRAL
            trust = abs(signed_score)
            accepted = label == EvidenceStance.SUPPORT and trust >= self.accept_threshold
            decisions.append(
                EvidenceDecision(
                    evidence_id=evidence_id,
                    signed_score=round(signed_score, 8),
                    trust=round(_clamp(trust), 8),
                    label=label,
                    accepted=accepted,
                    provenance_prior=round(priors[evidence_id], 8),
                    graph_adjustment=round(adjustment, 8),
                    reasons=tuple(reasons),
                )
            )
        return tuple(decisions)

