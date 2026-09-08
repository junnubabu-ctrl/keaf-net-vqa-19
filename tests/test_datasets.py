import json
import tempfile
import unittest
from pathlib import Path

from evitrust_vqa.datasets import AOKVQAAdapter, KRVQAAdapter, OKVQAAdapter


class DatasetAdapterTests(unittest.TestCase):
    def test_okvqa_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q = root / "q.json"
            a = root / "a.json"
            q.write_text(json.dumps({"questions": [{"question_id": 1, "image_id": 7, "question": "What is shown?"}]}))
            a.write_text(json.dumps({"annotations": [{"question_id": 1, "answers": [{"answer": "cat"}] }]}))
            rows = OKVQAAdapter({"val": q}, {"val": a}).load("val")
            self.assertEqual(rows[0].question_id, "1")
            self.assertEqual(rows[0].answers, ("cat",))

    def test_aokvqa_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aok.json"
            path.write_text(json.dumps([{
                "question_id": "q1",
                "image_id": 10,
                "question": "Why?",
                "direct_answers": ["because"],
                "choices": ["because", "never"],
                "correct_choice_idx": 0,
            }]))
            row = AOKVQAAdapter({"val": path}).load("val")[0]
            self.assertEqual(row.direct_answer, "because")
            self.assertEqual(row.correct_choice_idx, 0)

    def test_krvqa_configurable_field_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kr.json"
            path.write_text(json.dumps([{"qid": 2, "img": "x", "query": "Who?", "gold": "alice"}]))
            adapter = KRVQAAdapter(
                {"test": path},
                field_map={"question_id": "qid", "image_id": "img", "question": "query", "answer": "gold"},
            )
            row = adapter.load("test")[0]
            self.assertEqual(row.answers, ("alice",))


if __name__ == "__main__":
    unittest.main()
