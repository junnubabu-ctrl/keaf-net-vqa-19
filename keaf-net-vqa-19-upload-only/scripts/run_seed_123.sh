#!/usr/bin/env bash
set -euo pipefail
python -m keafnet.train --config configs/keaf_okvqa.yaml --seed 123 --output runs/okvqa_seed123
python -m keafnet.evaluate --checkpoint runs/okvqa_seed123/last.pt --split okvqa_eval --output runs/okvqa_seed123/eval.json
