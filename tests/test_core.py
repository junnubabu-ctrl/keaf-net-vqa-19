import json
from pathlib import Path
import tempfile
import unittest

from keaf_net.data import pad_2d, read_jsonl
from keaf_net.knowledge import JsonlKnowledgeStore


class CoreTests(unittest.TestCase):
    def test_padding(self):
        self.assertEqual(pad_2d([[1, 2], [3]]), [[1, 2], [3, 0]])

    def test_retrieval_is_deterministic(self):
        rows = [
            {"subject": "zebra", "relation": "is", "object": "striped", "source": "test"},
            {"subject": "zebra", "relation": "lives in", "object": "Africa", "source": "test"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            store = JsonlKnowledgeStore(path)
            first = store.retrieve("where zebra lives", 2)
            second = store.retrieve("where zebra lives", 2)
            self.assertEqual(first, second)
            self.assertEqual(first[0].object, "Africa")

    def test_jsonl_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"question_id": 1}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                list(read_jsonl(path))


if __name__ == "__main__":
    unittest.main()
