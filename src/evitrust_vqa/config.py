"""Typed, validation-first configuration for EviTrust-VQA."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EviTrustConfig:
    """Configuration values that affect inference or evaluation."""

    schema_version: str = "1.0"
    seed: int = 2_300_106_037
    top_k: int = 10
    evidence_accept_threshold: float = 0.55
    support_threshold: float = 0.15
    refute_threshold: float = -0.15
    target_selective_risk: float = 0.10
    corruption_rates: tuple[float, ...] = (0.10, 0.25, 0.50)
    minimum_training_seeds: int = 3
    bootstrap_replicates: int = 10_000
    knowledge_snapshots: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        for name in (
            "evidence_accept_threshold",
            "target_selective_risk",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.refute_threshold >= self.support_threshold:
            raise ValueError("refute_threshold must be lower than support_threshold")
        if self.minimum_training_seeds < 3:
            raise ValueError("publication protocol requires at least three seeds")
        if any(rate <= 0.0 or rate > 1.0 for rate in self.corruption_rates):
            raise ValueError("corruption rates must be in (0, 1]")

    @classmethod
    def from_json(cls, path: str | Path) -> "EviTrustConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "corruption_rates" in payload:
            payload["corruption_rates"] = tuple(payload["corruption_rates"])
        return cls(**payload)

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

