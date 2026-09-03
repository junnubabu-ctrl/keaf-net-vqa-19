"""Dependency-free metrics required by the Paper 4 evaluation protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass


def exact_match(prediction: str, target: str) -> float:
    return float(" ".join(prediction.lower().split()) == " ".join(target.lower().split()))


def vqa_soft_accuracy(prediction: str, human_answers: list[str]) -> float:
    normalized = " ".join(prediction.lower().split())
    matches = sum(" ".join(answer.lower().split()) == normalized for answer in human_answers)
    return min(1.0, matches / 3.0)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, evidence_id in enumerate(ranked_ids, start=1):
        if evidence_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    def dcg(values: list[float]) -> float:
        return sum((2.0**value - 1.0) / math.log2(index + 2.0) for index, value in enumerate(values))

    observed = [relevance.get(item, 0.0) for item in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(observed) / denominator if denominator else 0.0


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    mcc: float


def binary_classification(predicted: list[bool], actual: list[bool]) -> ClassificationMetrics:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual lengths differ")
    tp = sum(p and a for p, a in zip(predicted, actual, strict=True))
    fp = sum(p and not a for p, a in zip(predicted, actual, strict=True))
    fn = sum(not p and a for p, a in zip(predicted, actual, strict=True))
    tn = sum(not p and not a for p, a in zip(predicted, actual, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return ClassificationMetrics(precision, recall, f1, mcc)


def brier_score(confidences: list[float], correct: list[bool]) -> float:
    if not confidences or len(confidences) != len(correct):
        raise ValueError("aligned non-empty values are required")
    return sum((confidence - float(label)) ** 2 for confidence, label in zip(confidences, correct, strict=True)) / len(confidences)


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = 10
) -> float:
    if bins < 1 or not confidences or len(confidences) != len(correct):
        raise ValueError("valid bins and aligned non-empty values are required")
    total = len(confidences)
    ece = 0.0
    for bin_index in range(bins):
        low, high = bin_index / bins, (bin_index + 1) / bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if low <= confidence < high or (bin_index == bins - 1 and confidence == 1.0)
        ]
        if not indices:
            continue
        bin_confidence = sum(confidences[index] for index in indices) / len(indices)
        bin_accuracy = sum(correct[index] for index in indices) / len(indices)
        ece += (len(indices) / total) * abs(bin_accuracy - bin_confidence)
    return ece


def risk_coverage_curve(
    confidences: list[float], correct: list[bool]
) -> list[tuple[float, float, float]]:
    if not confidences or len(confidences) != len(correct):
        raise ValueError("aligned non-empty values are required")
    pairs = sorted(
        zip(confidences, correct, strict=True),
        key=lambda pair: pair[0],
        reverse=True,
    )
    curve: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    errors = 0
    index = 0
    while index < len(pairs):
        confidence = pairs[index][0]
        group_end = index
        while group_end < len(pairs) and pairs[group_end][0] == confidence:
            errors += int(not pairs[group_end][1])
            group_end += 1
        curve.append((group_end / len(pairs), errors / group_end, confidence))
        index = group_end
    return curve


def area_under_risk_coverage(confidences: list[float], correct: list[bool]) -> float:
    curve = risk_coverage_curve(confidences, correct)
    area = 0.0
    for previous, current in zip(curve[:-1], curve[1:], strict=True):
        width = current[0] - previous[0]
        area += width * (previous[1] + current[1]) / 2.0
    return area


def robustness_degradation(clean_score: float, corrupted_score: float) -> dict[str, float]:
    absolute = clean_score - corrupted_score
    relative = absolute / clean_score if clean_score else 0.0
    return {"absolute": absolute, "relative": relative}
