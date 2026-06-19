from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import PlaceholderVQADataset
from .model import KEAFConfig, KEAFNet
from .utils import set_seed, append_csv, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_file = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    mcfg = KEAFConfig(**cfg_file.get("model", {}))
    model = KEAFNet(mcfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = PlaceholderVQADataset(num_answers=mcfg.num_answers)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg_file["training"]["learning_rate"]), weight_decay=float(cfg_file["training"]["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss()

    fieldnames = ["seed", "epoch", "train_loss", "val_accuracy", "best_val_accuracy", "learning_rate"]
    best = 0.0
    for epoch in range(1, int(cfg_file["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            triplets = batch["triplets"].to(device)
            target = batch["target"].to(device)
            out_dict = model(visual, text, triplets)
            loss = loss_fn(out_dict["logits"], target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        # Placeholder validation. Replace with official VQA soft-accuracy evaluation.
        val_acc = 0.0
        best = max(best, val_acc)
        append_csv(out / "training_log.csv", {
            "seed": args.seed, "epoch": epoch, "train_loss": round(total_loss / max(1, len(loader)), 6),
            "val_accuracy": val_acc, "best_val_accuracy": best, "learning_rate": opt.param_groups[0]["lr"],
        }, fieldnames)
        torch.save({"model": model.state_dict(), "config": cfg_file, "seed": args.seed}, out / "last.pt")
    write_json(out / "run_metadata.json", {"seed": args.seed, "device": str(device), "note": "Replace placeholder dataset/evaluator with official data loaders."})


if __name__ == "__main__":
    main()
