#!/usr/bin/env bash
set -euo pipefail
python -m keafnet.train --config configs/keaf_okvqa.yaml --seed 2024 --output runs/okvqa_seed2024
python -m keafnet.evaluate --checkpoint runs/okvqa_seed2024/last.pt --split okvqa_eval --output runs/okvqa_seed2024/eval.json
