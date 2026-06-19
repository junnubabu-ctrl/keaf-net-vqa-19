"""Dataset interfaces for KEAF-Net.

Replace the placeholder random tensors with real pre-extracted ViT/Faster R-CNN
features, BERT token embeddings, and Sentence-BERT triplet embeddings.
"""
from __future__ import annotations

from typing import Dict
import torch
from torch.utils.data import Dataset


class PlaceholderVQADataset(Dataset):
    def __init__(self, length: int = 32, hidden_dim: int = 768, num_answers: int = 3129):
        self.length = length
        self.hidden_dim = hidden_dim
        self.num_answers = num_answers

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "visual": torch.randn(232, self.hidden_dim),
            "text": torch.randn(32, self.hidden_dim),
            "triplets": torch.randn(50, self.hidden_dim),
            "target": torch.zeros(self.num_answers).scatter_(0, torch.randint(0, self.num_answers, (1,)), 1.0),
        }
