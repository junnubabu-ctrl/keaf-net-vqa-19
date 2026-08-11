#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml

from keaf_net import KEAFNet, KEAFNetConfig
from keaf_net.data import make_loader
from keaf_net.engine import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="predictions.jsonl")
    args = parser.parse_args()
    import torch

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = KEAFNet(KEAFNetConfig(**config["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = make_loader(config["data"]["val"], config["training"]["batch_size"])
    print(evaluate(model, loader, device, args.output))


if __name__ == "__main__":
    main()
