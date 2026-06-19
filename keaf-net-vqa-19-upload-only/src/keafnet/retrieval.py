"""Knowledge retrieval interfaces.

Expected retrieval pipeline:
1. extract top object labels from Faster R-CNN detections;
2. extract noun phrases from question text;
3. query ConceptNet 5.5 and CSKG;
4. retrieve one-hop and two-hop triplets;
5. deduplicate, rank by edge weight, cap at P=50;
6. encode triplets using Sentence-BERT all-MiniLM-L6-v2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Triplet = Tuple[str, str, str]


@dataclass
class RetrievalConfig:
    max_triplets: int = 50
    detection_threshold: float = 0.3
    top_object_labels: int = 10


class KnowledgeRetriever:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg

    def retrieve(self, object_labels: List[str], noun_phrases: List[str]) -> List[Triplet]:
        raise NotImplementedError("Connect to local ConceptNet/CSKG index. Do not call online APIs during benchmark evaluation.")
