"""Canonical evidence, graph, answer, and audit schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping


class SourceType(str, Enum):
    KNOWLEDGE_GRAPH = "knowledge_graph"
    ENCYCLOPEDIA = "encyclopedia"
    RETRIEVED_PASSAGE = "retrieved_passage"
    UNVERIFIED_WEB = "unverified_web"
    SYNTHETIC = "synthetic"


class EvidenceStance(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> int:
        return {
            EvidenceStance.SUPPORT: 1,
            EvidenceStance.REFUTE: -1,
            EvidenceStance.NEUTRAL: 0,
        }[self]


class RelationType(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    REDUNDANT = "redundant"

    @property
    def sign(self) -> int:
        return {
            RelationType.SUPPORT: 1,
            RelationType.CONTRADICT: -1,
            RelationType.REDUNDANT: 0,
        }[self]


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    claim: str
    source_id: str
    source_type: SourceType
    snapshot: str
    retrieval_score: float
    content_score: float
    source_reliability: float
    stance: EvidenceStance = EvidenceStance.NEUTRAL
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    answer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id is required")
        if not self.claim.strip():
            raise ValueError("claim is required")
        if not self.source_id.strip() or not self.snapshot.strip():
            raise ValueError("source_id and snapshot are required provenance fields")
        for name in ("retrieval_score", "content_score", "source_reliability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        known = {
            "evidence_id",
            "claim",
            "source_id",
            "source_type",
            "snapshot",
            "retrieval_score",
            "content_score",
            "source_reliability",
            "stance",
            "subject",
            "predicate",
            "object",
            "answer",
            "metadata",
        }
        data = dict(payload)
        extras = {key: value for key, value in data.items() if key not in known}
        metadata = dict(data.get("metadata", {}))
        metadata.update(extras)
        data = {key: value for key, value in data.items() if key in known}
        data["source_type"] = SourceType(data["source_type"])
        data["stance"] = EvidenceStance(data.get("stance", "neutral"))
        data["metadata"] = metadata
        return cls(**data)

    def with_updates(self, **changes: Any) -> "EvidenceRecord":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_type"] = self.source_type.value
        payload["stance"] = self.stance.value
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class SignedEdge:
    source_id: str
    target_id: str
    relation: RelationType
    weight: float
    reason: str

    def __post_init__(self) -> None:
        if self.source_id == self.target_id:
            raise ValueError("self edges are not allowed")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("edge weight must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relation"] = self.relation.value
        payload["sign"] = self.relation.sign
        return payload


@dataclass(frozen=True)
class EvidenceDecision:
    evidence_id: str
    signed_score: float
    trust: float
    label: EvidenceStance
    accepted: bool
    provenance_prior: float
    graph_adjustment: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label"] = self.label.value
        return payload


@dataclass(frozen=True)
class SufficiencyResult:
    score: float
    sufficient: bool
    support_mass: float
    contradiction_mass: float
    source_diversity: float
    query_coverage: float
    accepted_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerCandidate:
    text: str
    confidence: float
    supporting_evidence_ids: tuple[str, ...]
    vote_margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

