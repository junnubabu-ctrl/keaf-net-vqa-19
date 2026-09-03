"""Validation-fit provenance calibration with no test-label access."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import EvidenceRecord, SourceType


DEFAULT_SOURCE_PRIORS: dict[SourceType, float] = {
    SourceType.KNOWLEDGE_GRAPH: 0.85,
    SourceType.ENCYCLOPEDIA: 0.80,
    SourceType.RETRIEVED_PASSAGE: 0.65,
    SourceType.UNVERIFIED_WEB: 0.35,
    SourceType.SYNTHETIC: 0.50,
}


@dataclass(frozen=True)
class CalibrationExample:
    source_reliability: float
    source_type: SourceType
    correct: bool


class ProvenanceCalibrator:
    """A small interpretable logistic calibrator fitted only on validation examples."""

    def __init__(self, source_priors: dict[SourceType, float] | None = None):
        self.source_priors = dict(source_priors or DEFAULT_SOURCE_PRIORS)
        self.bias = -0.2
        self.weight_reliability = 1.4
        self.weight_prior = 1.0
        self.fitted = False

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def prior(self, evidence: EvidenceRecord) -> float:
        type_prior = self.source_priors.get(evidence.source_type, 0.5)
        logit = (
            self.bias
            + self.weight_reliability * (2.0 * evidence.source_reliability - 1.0)
            + self.weight_prior * (2.0 * type_prior - 1.0)
        )
        return self._sigmoid(logit)

    def fit(
        self,
        examples: list[CalibrationExample],
        *,
        learning_rate: float = 0.15,
        steps: int = 250,
    ) -> "ProvenanceCalibrator":
        if not examples:
            raise ValueError("validation examples are required for calibration")
        bias, w_rel, w_prior = self.bias, self.weight_reliability, self.weight_prior
        for _ in range(steps):
            grad_b = grad_rel = grad_prior = 0.0
            for item in examples:
                rel = 2.0 * item.source_reliability - 1.0
                prior = 2.0 * self.source_priors.get(item.source_type, 0.5) - 1.0
                probability = self._sigmoid(bias + w_rel * rel + w_prior * prior)
                error = probability - float(item.correct)
                grad_b += error
                grad_rel += error * rel
                grad_prior += error * prior
            scale = learning_rate / len(examples)
            bias -= scale * grad_b
            w_rel -= scale * grad_rel
            w_prior -= scale * grad_prior
        self.bias, self.weight_reliability, self.weight_prior = bias, w_rel, w_prior
        self.fitted = True
        return self

