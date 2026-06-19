from __future__ import annotations

import argparse
from pathlib import Path
from .utils import write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    # Connect this script to official VQA/OK-VQA/A-OKVQA evaluation APIs.
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "status": "template",
        "message": "Run official evaluator here and write exact accuracy, per-type accuracy, and confidence intervals.",
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
