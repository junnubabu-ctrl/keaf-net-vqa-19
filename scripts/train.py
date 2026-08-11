#!/usr/bin/env python3
"""Configuration-driven training entry point.

Dataset-specific feature loading is intentionally an explicit integration point:
the repository does not manufacture or silently download manuscript data.
"""

import argparse
import json
from pathlib import Path
import random

import yaml

from keaf_net import KEAFNet, KEAFNetConfig
from keaf_net.data import make_loader
from keaf_net.engine import evaluate, train_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    random.seed(config["seed"])
    import numpy as np
    import torch

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KEAFNet(KEAFNetConfig(**config["model"]))
    model.to(device)
    training = config["training"]
    train_loader = make_loader(config["data"]["train"], training["batch_size"], shuffle=True)
    val_loader = make_loader(config["data"]["val"], training["batch_size"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, training["epochs"] + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device, output_dir / f"predictions-{epoch}.jsonl")
            record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record))
            if val_metrics["accuracy"] > best:
                best = val_metrics["accuracy"]
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, output_dir / "best.pt")


if __name__ == "__main__":
    main()
