import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class ModelTests(unittest.TestCase):
    def test_end_to_end_shapes(self):
        from keaf_net import KEAFNet, KEAFNetConfig

        config = KEAFNetConfig(
            visual_dim=8, fact_dim=6, vocab_size=30, answer_size=7,
            hidden_dim=12, top_k=3, graph_layers=2, reasoning_hops=3,
        )
        model = KEAFNet(config)
        output = model(
            visual=torch.randn(2, 4, 8),
            question_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
            facts=torch.randn(2, 5, 6),
            fact_mask=torch.ones(2, 5, dtype=torch.bool),
        )
        self.assertEqual(tuple(output["logits"].shape), (2, 7))
        self.assertEqual(tuple(output["fact_indices"].shape), (2, 3))
        self.assertEqual(tuple(output["hop_attention"].shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()
