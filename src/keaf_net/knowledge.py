"""Knowledge retrieval contracts and deterministic offline implementation."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    score: float = 1.0
    source: str = "unknown"

    @property
    def text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"


class Retriever(Protocol):
    def retrieve(self, query: str, limit: int = 100) -> list[Fact]: ...


class JsonlKnowledgeStore:
    """Reproducible lexical retriever over a versioned JSONL knowledge snapshot."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.facts = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                self.facts.append(Fact(**row))

    def retrieve(self, query: str, limit: int = 100) -> list[Fact]:
        terms = set(query.lower().split())
        ranked = []
        for fact in self.facts:
            overlap = len(terms.intersection(fact.text.lower().split()))
            if overlap:
                ranked.append((overlap * fact.score, fact.text, fact))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked[:limit]]

    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
