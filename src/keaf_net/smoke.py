"""Dependency-light installation smoke test."""

import tempfile
from pathlib import Path

from .data import pad_2d
from .knowledge import JsonlKnowledgeStore


def main():
    assert pad_2d([[1], [2, 3]]) == [[1, 0], [2, 3]]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "facts.jsonl"
        path.write_text('{"subject":"bird","relation":"can","object":"fly","source":"test"}\n', encoding="utf-8")
        store = JsonlKnowledgeStore(path)
        assert store.retrieve("which bird can fly", 1)[0].object == "fly"
        assert len(store.sha256()) == 64
    print("KEAF-Net smoke test passed")


if __name__ == "__main__":
    main()
