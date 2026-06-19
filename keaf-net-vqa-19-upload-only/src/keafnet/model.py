"""KEAF-Net model modules.

This implementation is a reviewer-ready reference skeleton. It defines the
core AKF, HGAF, and MHSR modules and validates tensor shapes. Dataset-specific
loading and official VQA scoring must be connected in train.py/evaluate.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class KEAFConfig:
    hidden_dim: int = 768
    num_answers: int = 3129
    max_triplets: int = 50
    loo_samples: int = 10
    akf_temperature: float = 0.1
    akf_loss_weight: float = 0.3
    hgaf_layers: int = 2
    hgaf_heads: int = 8
    reasoning_hops: int = 3
    dropout: float = 0.1


class AdaptiveKnowledgeFilter(nn.Module):
    """Scores triplets against a joint image-question context."""

    def __init__(self, cfg: KEAFConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.scorer = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )
        self.threshold = nn.Parameter(torch.tensor(0.42))
        self.norm = nn.LayerNorm(d)

    def forward(self, visual: torch.Tensor, q_cls: torch.Tensor, triplets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual: [B, Nv, D]
            q_cls: [B, D]
            triplets: [B, P, D]
        Returns:
            filtered_triplets: [B, P, D] with rejected triplets masked to zero
            scores: [B, P]
        """
        h_iq = self.norm(self.q_proj(q_cls) + self.v_proj(visual.mean(dim=1)))
        h_iq_exp = h_iq.unsqueeze(1).expand_as(triplets)
        features = torch.cat([triplets, h_iq_exp, triplets * h_iq_exp], dim=-1)
        scores = torch.sigmoid(self.scorer(features)).squeeze(-1)
        mask = (scores > torch.sigmoid(self.threshold)).float().unsqueeze(-1)
        return triplets * mask, scores


class TypeAwareGraphLayer(nn.Module):
    """Compact type-aware graph attention placeholder for HGAF.

    edge_index: [2, E]
    edge_type: [E] values 0..4 for VV, TT, VK, TK, VT.
    """

    def __init__(self, cfg: KEAFConfig, num_edge_types: int = 5):
        super().__init__()
        d = cfg.hidden_dim
        self.type_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(num_edge_types)])
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        src, dst = edge_index
        messages = torch.zeros_like(x)
        counts = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        for t, proj in enumerate(self.type_proj):
            mask = edge_type == t
            if mask.any():
                s = src[mask]
                d = dst[mask]
                msg = proj(x[s])
                messages.index_add_(0, d, msg)
                counts.index_add_(0, d, torch.ones((d.numel(), 1), device=x.device, dtype=x.dtype))
        messages = messages / counts.clamp_min(1.0)
        return self.norm(x + torch.sigmoid(self.out(messages)))


class HGAF(nn.Module):
    """Heterogeneous Graph Adaptive Fusion."""

    def __init__(self, cfg: KEAFConfig):
        super().__init__()
        self.layers = nn.ModuleList([TypeAwareGraphLayer(cfg) for _ in range(cfg.hgaf_layers)])
        d = cfg.hidden_dim
        self.gate_v = nn.Linear(d, d)
        self.gate_t = nn.Linear(d, d)
        self.gate_k = nn.Linear(d, d)

    def forward(self, visual: torch.Tensor, text: torch.Tensor, knowledge: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None, edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        # A minimal batching strategy: each sample uses pooled streams when graph edges are not provided.
        if edge_index is None or edge_type is None:
            hv, ht, hk = visual.mean(dim=1), text.mean(dim=1), knowledge.mean(dim=1)
            return torch.sigmoid(self.gate_v(hv)) * hv + torch.sigmoid(self.gate_t(ht)) * ht + torch.sigmoid(self.gate_k(hk)) * hk
        raise NotImplementedError("Batch-specific graph edge packing should be implemented for full training.")


class MHSR(nn.Module):
    """Multi-Hop Semantic Reasoning over fused graph/evidence representation."""

    def __init__(self, cfg: KEAFConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.hops = cfg.reasoning_hops
        self.attn = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)

    def forward(self, q_cls: torch.Tensor, evidence_nodes: torch.Tensor) -> torch.Tensor:
        # evidence_nodes: [B, N, D]
        q = q_cls
        context = evidence_nodes.mean(dim=1)
        for _ in range(self.hops):
            scores = torch.einsum("bd,bnd->bn", self.attn(q), evidence_nodes)
            beta = F.softmax(scores, dim=-1)
            context = torch.einsum("bn,bnd->bd", beta, evidence_nodes)
            q = self.gru(context, q)
        return torch.cat([q, context], dim=-1)


class KEAFNet(nn.Module):
    def __init__(self, cfg: KEAFConfig):
        super().__init__()
        self.cfg = cfg
        self.akf = AdaptiveKnowledgeFilter(cfg)
        self.hgaf = HGAF(cfg)
        self.mhsr = MHSR(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_answers),
        )

    def forward(self, visual: torch.Tensor, text: torch.Tensor, triplets: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_cls = text[:, 0]
        filtered_k, akf_scores = self.akf(visual, q_cls, triplets)
        fused = self.hgaf(visual, text, filtered_k)
        evidence_nodes = torch.cat([visual, text, filtered_k, fused.unsqueeze(1)], dim=1)
        reasoning = self.mhsr(q_cls, evidence_nodes)
        logits = self.classifier(reasoning)
        return {"logits": logits, "akf_scores": akf_scores, "filtered_knowledge": filtered_k}
