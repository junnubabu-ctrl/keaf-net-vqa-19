"""End-to-end KEAF-Net model."""

from dataclasses import dataclass

from .modules import AdaptiveKnowledgeFilter, GraphFusionLayer, MultiHopSemanticReasoner, _require_torch

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object


@dataclass(frozen=True)
class KEAFNetConfig:
    visual_dim: int = 2048
    fact_dim: int = 768
    vocab_size: int = 32000
    answer_size: int = 3000
    hidden_dim: int = 512
    top_k: int = 50
    graph_layers: int = 2
    reasoning_hops: int = 3


if torch is not None:
    class KEAFNet(nn.Module):
        def __init__(self, config: KEAFNetConfig):
            super().__init__()
            self.config = config
            d = config.hidden_dim
            self.visual_projection = nn.Linear(config.visual_dim, d)
            self.fact_projection = nn.Linear(config.fact_dim, d)
            self.tokens = nn.Embedding(config.vocab_size, d, padding_idx=0)
            self.filter = AdaptiveKnowledgeFilter(d, config.top_k)
            self.graph = nn.ModuleList([GraphFusionLayer(d) for _ in range(config.graph_layers)])
            self.reasoner = MultiHopSemanticReasoner(d, config.reasoning_hops)
            self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.answer_size))

        def encode_question(self, token_ids, token_mask=None):
            embedded = self.tokens(token_ids)
            if token_mask is None:
                token_mask = token_ids.ne(0)
            weights = token_mask.unsqueeze(-1).to(embedded.dtype)
            return (embedded * weights).sum(1) / weights.sum(1).clamp_min(1.0)

        def forward(self, visual, question_ids, facts, visual_mask=None, question_mask=None, fact_mask=None):
            visual = self.visual_projection(visual)
            facts = self.fact_projection(facts)
            question = self.encode_question(question_ids, question_mask)
            selected, fact_weights, fact_indices, fact_logits = self.filter(facts, question, fact_mask)

            batch, nv, _ = visual.shape
            nk = selected.size(1)
            qnode = question.unsqueeze(1)
            nodes = torch.cat([visual, qnode, selected], dim=1)
            node_types = torch.cat([
                torch.zeros(batch, nv, dtype=torch.long, device=nodes.device),
                torch.ones(batch, 1, dtype=torch.long, device=nodes.device),
                torch.full((batch, nk), 2, dtype=torch.long, device=nodes.device),
            ], dim=1)
            if visual_mask is None:
                visual_mask = torch.ones(batch, nv, dtype=torch.bool, device=nodes.device)
            node_mask = torch.cat([
                visual_mask.bool(),
                torch.ones(batch, 1, dtype=torch.bool, device=nodes.device),
                torch.ones(batch, nk, dtype=torch.bool, device=nodes.device),
            ], dim=1)
            graph_attention = []
            for layer in self.graph:
                nodes, attention = layer(nodes, node_types, question, node_mask)
                graph_attention.append(attention)
            state, hop_attention = self.reasoner(nodes, question, node_mask)
            return {
                "logits": self.classifier(state),
                "fact_indices": fact_indices,
                "fact_weights": fact_weights,
                "fact_logits": fact_logits,
                "graph_attention": graph_attention,
                "hop_attention": hop_attention,
            }
else:
    class KEAFNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs): _require_torch()
