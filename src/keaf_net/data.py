"""JSONL dataset and padding utilities."""

import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            required = {"question_id", "visual", "question", "facts", "answer"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            yield row


def pad_2d(sequences, value=0):
    """Pad Python sequences without introducing a framework dependency."""
    width = max((len(sequence) for sequence in sequences), default=0)
    return [list(sequence) + [value] * (width - len(sequence)) for sequence in sequences]


def torch_collate(rows):
    """Collate variable-length JSONL records into KEAF-Net tensors."""
    import torch

    batch = len(rows)
    max_regions = max(len(row["visual"]) for row in rows)
    max_facts = max(len(row["facts"]) for row in rows)
    max_tokens = max(len(row["question"]) for row in rows)
    visual_dim = len(rows[0]["visual"][0])
    fact_dim = len(rows[0]["facts"][0])
    visual = torch.zeros(batch, max_regions, visual_dim, dtype=torch.float32)
    facts = torch.zeros(batch, max_facts, fact_dim, dtype=torch.float32)
    question = torch.zeros(batch, max_tokens, dtype=torch.long)
    visual_mask = torch.zeros(batch, max_regions, dtype=torch.bool)
    fact_mask = torch.zeros(batch, max_facts, dtype=torch.bool)
    for index, row in enumerate(rows):
        nr, nf, nq = len(row["visual"]), len(row["facts"]), len(row["question"])
        visual[index, :nr] = torch.tensor(row["visual"])
        facts[index, :nf] = torch.tensor(row["facts"])
        question[index, :nq] = torch.tensor(row["question"])
        visual_mask[index, :nr] = True
        fact_mask[index, :nf] = True
    return {
        "question_id": [row["question_id"] for row in rows],
        "answer": torch.tensor([row["answer"] for row in rows], dtype=torch.long),
        "inputs": {
            "visual": visual,
            "question_ids": question,
            "facts": facts,
            "visual_mask": visual_mask,
            "fact_mask": fact_mask,
        },
    }


def make_loader(path, batch_size, shuffle=False):
    """Create a PyTorch DataLoader after validating all JSONL records."""
    from torch.utils.data import DataLoader

    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=torch_collate)
