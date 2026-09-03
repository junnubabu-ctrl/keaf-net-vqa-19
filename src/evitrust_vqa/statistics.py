"""Deterministic statistical utilities for locked experiment analysis."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Summary:
    mean: float
    standard_deviation: float
    ci95_low: float
    ci95_high: float
    count: int


def summarize(values: list[float]) -> Summary:
    if not values:
        raise ValueError("at least one value is required")
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    standard_deviation = math.sqrt(variance)
    half_width = 1.96 * standard_deviation / math.sqrt(len(values))
    return Summary(mean, standard_deviation, mean - half_width, mean + half_width, len(values))


@dataclass(frozen=True)
class BootstrapResult:
    observed_delta: float
    ci95_low: float
    ci95_high: float
    p_two_sided: float
    replicates: int


def paired_bootstrap_delta(
    system_a: list[float],
    system_b: list[float],
    *,
    replicates: int = 10_000,
    seed: int = 2_300_106_037,
) -> BootstrapResult:
    if not system_a or len(system_a) != len(system_b):
        raise ValueError("aligned non-empty paired samples are required")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    deltas = [a - b for a, b in zip(system_a, system_b, strict=True)]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    sampled: list[float] = []
    for _ in range(replicates):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        sampled.append(sum(draw) / len(draw))
    sampled.sort()
    low = sampled[int(0.025 * (replicates - 1))]
    high = sampled[int(0.975 * (replicates - 1))]
    non_positive = sum(value <= 0.0 for value in sampled) / replicates
    non_negative = sum(value >= 0.0 for value in sampled) / replicates
    p_value = min(1.0, 2.0 * min(non_positive, non_negative))
    return BootstrapResult(observed, low, high, p_value, replicates)


def holm_adjust(p_values: list[float]) -> list[float]:
    if any(value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, (original_index, value) in enumerate(indexed):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted

