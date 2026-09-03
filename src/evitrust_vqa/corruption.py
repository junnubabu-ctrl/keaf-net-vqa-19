"""Deterministic controlled evidence corruptions for robustness experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schemas import EvidenceRecord, EvidenceStance, SourceType


@dataclass(frozen=True)
class CorruptionResult:
    kind: str
    rate: float
    seed: int
    evidence: tuple[EvidenceRecord, ...]
    affected_ids: tuple[str, ...]


class EvidenceCorruptor:
    SUPPORTED = {"irrelevant", "duplicate", "contradictory", "missing", "stale_source_swapped"}

    def apply(
        self,
        evidence: list[EvidenceRecord],
        *,
        kind: str,
        rate: float,
        seed: int,
        irrelevant_pool: list[EvidenceRecord] | None = None,
    ) -> CorruptionResult:
        if kind not in self.SUPPORTED:
            raise ValueError(f"unsupported corruption: {kind}")
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
        if not evidence:
            return CorruptionResult(kind, rate, seed, (), ())
        rng = random.Random(seed)
        count = max(1, round(len(evidence) * rate))
        selected = sorted(rng.sample(range(len(evidence)), min(count, len(evidence))))
        affected = tuple(evidence[index].evidence_id for index in selected)
        output = list(evidence)

        if kind == "missing":
            output = [item for index, item in enumerate(evidence) if index not in selected]
        elif kind == "duplicate":
            for index in selected:
                item = evidence[index]
                output.append(
                    item.with_updates(
                        evidence_id=f"{item.evidence_id}__duplicate",
                        metadata={**dict(item.metadata), "corruption": "duplicate"},
                    )
                )
        elif kind == "contradictory":
            for index in selected:
                item = output[index]
                flipped = (
                    EvidenceStance.REFUTE
                    if item.stance == EvidenceStance.SUPPORT
                    else EvidenceStance.SUPPORT
                )
                output[index] = item.with_updates(
                    stance=flipped,
                    content_score=max(0.0, 1.0 - item.content_score),
                    metadata={**dict(item.metadata), "corruption": "contradictory"},
                )
        elif kind == "stale_source_swapped":
            for index in selected:
                item = output[index]
                output[index] = item.with_updates(
                    source_id=f"source-swapped::{item.source_id}",
                    source_type=SourceType.UNVERIFIED_WEB,
                    snapshot="stale-or-unknown",
                    source_reliability=min(item.source_reliability, 0.20),
                    metadata={**dict(item.metadata), "corruption": "stale_source_swapped"},
                )
        elif kind == "irrelevant":
            pool = list(irrelevant_pool or [])
            if not pool:
                for position, index in enumerate(selected):
                    output.append(
                        EvidenceRecord(
                            evidence_id=f"irrelevant-{seed}-{position}",
                            claim="This evidence concerns an unrelated synthetic topic.",
                            source_id="synthetic-irrelevant-pool",
                            source_type=SourceType.SYNTHETIC,
                            snapshot="fixture",
                            retrieval_score=evidence[index].retrieval_score,
                            content_score=0.05,
                            source_reliability=0.10,
                            stance=EvidenceStance.NEUTRAL,
                            metadata={"corruption": "irrelevant"},
                        )
                    )
            else:
                for position in range(len(selected)):
                    item = pool[position % len(pool)]
                    output.append(
                        item.with_updates(
                            evidence_id=f"{item.evidence_id}__irrelevant_{position}",
                            metadata={**dict(item.metadata), "corruption": "irrelevant"},
                        )
                    )
        return CorruptionResult(kind, rate, seed, tuple(output), affected)

