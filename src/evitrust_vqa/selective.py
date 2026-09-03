"""Validation-calibrated selective prediction and risk-coverage utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectiveDecision:
    answer: str | None
    confidence: float
    threshold: float
    abstained: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "abstained": self.abstained,
            "reason": self.reason,
        }


class SelectiveCalibrator:
    """Choose the highest-coverage validation threshold within a target risk."""

    def __init__(self, target_risk: float = 0.10, default_threshold: float = 0.60):
        if not 0.0 <= target_risk <= 1.0:
            raise ValueError("target_risk must be in [0, 1]")
        self.target_risk = target_risk
        self.threshold = default_threshold
        self.fitted = False

    def fit(self, confidences: list[float], correct: list[bool]) -> "SelectiveCalibrator":
        if not confidences or len(confidences) != len(correct):
            raise ValueError("aligned non-empty validation confidences and labels are required")
        pairs = sorted(
            zip(confidences, correct, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )
        errors = 0
        best_count = 0
        best_threshold = 1.000000001
        index = 0
        while index < len(pairs):
            confidence = pairs[index][0]
            group_end = index
            while group_end < len(pairs) and pairs[group_end][0] == confidence:
                errors += int(not pairs[group_end][1])
                group_end += 1
            risk = errors / group_end
            if risk <= self.target_risk:
                best_count = group_end
                best_threshold = confidence
            index = group_end
        self.threshold = best_threshold if best_count else 1.000000001
        self.fitted = True
        return self

    def decide(self, answer: str, confidence: float, sufficient: bool) -> SelectiveDecision:
        if not sufficient:
            return SelectiveDecision(None, confidence, self.threshold, True, "insufficient evidence")
        if confidence < self.threshold:
            return SelectiveDecision(None, confidence, self.threshold, True, "below calibrated threshold")
        return SelectiveDecision(answer, confidence, self.threshold, False, "accepted")
