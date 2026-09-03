"""Controlled robustness experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .answering import InMemoryRetriever
from .config import EviTrustConfig
from .corruption import EvidenceCorruptor
from .metrics import exact_match, robustness_degradation
from .pipeline import EviTrustPipeline, InferenceRequest
from .schemas import EvidenceRecord


@dataclass(frozen=True)
class RobustnessTrial:
    corruption: str
    rate: float
    seed: int
    answer: str | None
    abstained: bool
    score: float
    absolute_degradation: float
    relative_degradation: float
    affected_ids: tuple[str, ...]


def run_corruption_suite(
    *,
    request: InferenceRequest,
    evidence: list[EvidenceRecord],
    target_answer: str,
    config: EviTrustConfig | None = None,
    corruption_rates: tuple[float, ...] | None = None,
    seed: int | None = None,
) -> tuple[RobustnessTrial, ...]:
    active = config or EviTrustConfig()
    trial_seed = active.seed if seed is None else seed
    rates = corruption_rates or active.corruption_rates
    clean_result = EviTrustPipeline(InMemoryRetriever(evidence), config=active).infer(request)
    clean_score = exact_match(clean_result.answer or "", target_answer)
    corruptor = EvidenceCorruptor()
    trials: list[RobustnessTrial] = []
    for kind in sorted(EvidenceCorruptor.SUPPORTED):
        for rate in rates:
            corrupted = corruptor.apply(
                evidence,
                kind=kind,
                rate=rate,
                seed=trial_seed,
            )
            result = EviTrustPipeline(
                InMemoryRetriever(list(corrupted.evidence)), config=active
            ).infer(request)
            score = exact_match(result.answer or "", target_answer)
            degradation = robustness_degradation(clean_score, score)
            trials.append(
                RobustnessTrial(
                    corruption=kind,
                    rate=rate,
                    seed=trial_seed,
                    answer=result.answer,
                    abstained=result.abstained,
                    score=score,
                    absolute_degradation=degradation["absolute"],
                    relative_degradation=degradation["relative"],
                    affected_ids=corrupted.affected_ids,
                )
            )
    return tuple(trials)

