#!/usr/bin/env bash
set -euo pipefail
python -m keafnet.train --config configs/keaf_okvqa.yaml --seed 42 --output runs/okvqa_seed42
python -m keafnet.evaluate --checkpoint runs/okvqa_seed42/last.pt --split okvqa_eval --output runs/okvqa_seed42/eval.json
