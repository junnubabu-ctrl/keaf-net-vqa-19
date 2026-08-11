"""Neural building blocks for KEAF-Net.

PyTorch is imported lazily so retrieval utilities and repository metadata remain
usable without the optional training dependency.
"""

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by smoke.py
    torch = None
    nn = object


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required; install with: pip install -e '.[train]'")


if torch is not None:
    class AdaptiveKnowledgeFilter(nn.Module):
        """Question-conditioned fact scorer with masked, deterministic top-k selection."""

        def __init__(self, hidden_dim: int, top_k: int = 50):
            super().__init__()
            self.top_k = top_k
            self.score = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            )

        def forward(self, facts, question, fact_mask=None):
            q = question.unsqueeze(1).expand_as(facts)
            features = torch.cat([facts, q, facts * q, torch.abs(facts - q)], dim=-1)
            logits = self.score(features).squeeze(-1)
            if fact_mask is not None:
                logits = logits.masked_fill(~fact_mask.bool(), torch.finfo(logits.dtype).min)
            k = min(self.top_k, facts.size(1))
            values, indices = torch.topk(logits, k=k, dim=1, sorted=True)
            selected = facts.gather(1, indices.unsqueeze(-1).expand(-1, -1, facts.size(-1)))
            weights = torch.softmax(values, dim=1)
            return selected, weights, indices, logits


    class GraphFusionLayer(nn.Module):
        """Typed message passing with question-conditioned residual gating."""

        def __init__(self, hidden_dim: int, node_types: int = 3):
            super().__init__()
            self.type_embedding = nn.Embedding(node_types, hidden_dim)
            self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.scale = hidden_dim ** -0.5

        def forward(self, nodes, node_types, question, node_mask=None, adjacency=None):
            typed = nodes + self.type_embedding(node_types)
            scores = torch.matmul(self.query(typed), self.key(typed).transpose(-1, -2)) * self.scale
            if adjacency is not None:
                scores = scores.masked_fill(~adjacency.bool(), torch.finfo(scores.dtype).min)
            if node_mask is not None:
                scores = scores.masked_fill(~node_mask[:, None, :].bool(), torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=-1)
            messages = torch.matmul(attention, self.value(typed))
            q = question.unsqueeze(1).expand_as(nodes)
            gate = torch.sigmoid(self.gate(torch.cat([nodes, q], dim=-1)))
            output = self.norm(nodes + gate * messages)
            if node_mask is not None:
                output = output * node_mask.unsqueeze(-1)
            return output, attention


    class MultiHopSemanticReasoner(nn.Module):
        """Recurrent attention over fused evidence for a fixed number of hops."""

        def __init__(self, hidden_dim: int, hops: int = 3):
            super().__init__()
            self.hops = hops
            self.node = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.state = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.energy = nn.Linear(hidden_dim, 1, bias=False)
            self.update = nn.GRUCell(hidden_dim, hidden_dim)

        def forward(self, nodes, initial_state, node_mask=None):
            state = initial_state
            traces = []
            for _ in range(self.hops):
                logits = self.energy(torch.tanh(self.node(nodes) + self.state(state).unsqueeze(1))).squeeze(-1)
                if node_mask is not None:
                    logits = logits.masked_fill(~node_mask.bool(), torch.finfo(logits.dtype).min)
                attention = torch.softmax(logits, dim=-1)
                context = torch.sum(attention.unsqueeze(-1) * nodes, dim=1)
                state = self.update(context, state)
                traces.append(attention)
            return state, torch.stack(traces, dim=1)
else:
    class AdaptiveKnowledgeFilter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs): _require_torch()

    class GraphFusionLayer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs): _require_torch()

    class MultiHopSemanticReasoner:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs): _require_torch()
