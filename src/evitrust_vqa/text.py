"""Small deterministic text helpers used by the transparent reference system."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
}


def normalize_text(value: str | None) -> str:
    return " ".join(tokens(value or "", drop_stopwords=False))


def tokens(value: str, *, drop_stopwords: bool = True) -> tuple[str, ...]:
    found = tuple(TOKEN_PATTERN.findall(value.lower()))
    if drop_stopwords:
        return tuple(token for token in found if token not in STOPWORDS)
    return found


def jaccard(left: str, right: str) -> float:
    a, b = set(tokens(left)), set(tokens(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

