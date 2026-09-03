"""EviTrust-VQA research implementation."""

from .config import EviTrustConfig
from .pipeline import EviTrustPipeline, InferenceRequest, InferenceResult
from .schemas import EvidenceRecord, EvidenceStance, SourceType

__all__ = [
    "EvidenceRecord",
    "EvidenceStance",
    "EviTrustConfig",
    "EviTrustPipeline",
    "InferenceRequest",
    "InferenceResult",
    "SourceType",
]

__version__ = "0.1.0"

