"""Training/evaluation functions with explicit metric provenance."""

import json
from pathlib import Path


def train_epoch(model, batches, optimizer, device="cpu"):
    import torch

    model.train()
    total_loss = 0.0
    examples = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        output = model(**{k: v.to(device) for k, v in batch["inputs"].items()})
        target = batch["answer"].to(device)
        loss = torch.nn.functional.cross_entropy(output["logits"], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * target.size(0)
        examples += target.size(0)
    return {"loss": total_loss / max(examples, 1), "examples": examples}


def evaluate(model, batches, device="cpu", prediction_path=None):
    import torch

    model.eval()
    correct = total = 0
    predictions = []
    with torch.no_grad():
        for batch in batches:
            output = model(**{k: v.to(device) for k, v in batch["inputs"].items()})
            predicted = output["logits"].argmax(-1).cpu()
            target = batch["answer"].cpu()
            correct += predicted.eq(target).sum().item()
            total += target.numel()
            for qid, answer in zip(batch["question_id"], predicted.tolist()):
                predictions.append({"question_id": qid, "answer": answer})
    if prediction_path:
        path = Path(prediction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}
