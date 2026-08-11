import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class CollationTests(unittest.TestCase):
    def test_masks_and_shapes(self):
        from keaf_net.data import torch_collate

        rows = [
            {"question_id": 1, "visual": [[1.0, 2.0]], "question": [1, 2], "facts": [[3.0]], "answer": 0},
            {"question_id": 2, "visual": [[1.0, 2.0], [2.0, 3.0]], "question": [3], "facts": [[4.0], [5.0]], "answer": 1},
        ]
        batch = torch_collate(rows)
        self.assertEqual(tuple(batch["inputs"]["visual"].shape), (2, 2, 2))
        self.assertEqual(batch["inputs"]["visual_mask"].sum().item(), 3)
        self.assertEqual(batch["inputs"]["fact_mask"].sum().item(), 3)
