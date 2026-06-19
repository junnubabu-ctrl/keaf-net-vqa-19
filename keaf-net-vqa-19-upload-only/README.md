# KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network for VQA

This repository provides a reviewer-ready implementation structure for KEAF-Net, including modules for:

- Adaptive Knowledge Filter (AKF)
- Heterogeneous Graph Adaptive Fusion (HGAF)
- Multi-Hop Semantic Reasoning (MHSR)
- deterministic seed control
- CSV logging for training and evaluation
- per-seed result export

## Repository status

This is a reproducibility release template. Dataset files, pretrained Faster R-CNN features, Sentence-BERT triplet embeddings, and official benchmark annotations must be downloaded from their original sources and placed according to `docs/dataset_setup.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run training for one seed
python -m keafnet.train --config configs/keaf_okvqa.yaml --seed 42 --output runs/okvqa_seed42

# Evaluate
python -m keafnet.evaluate --checkpoint runs/okvqa_seed42/best.pt --split okvqa_eval --output runs/okvqa_seed42/eval.json
```

## GitHub upload commands

```bash
git init
git add .
git commit -m "Initial KEAF-Net reproducibility release"
# Create an empty public repo in GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/keaf-net-vqa-19.git
git branch -M main
git push -u origin main
```

After upload, use this in the manuscript:

> Code Availability: The implementation and reproducibility scripts are available at: https://github.com/YOUR_USERNAME/keaf-net-vqa-19

## What reviewers expect

- exact commit hash
- dataset download/setup instructions
- config files
- training logs for each seed
- per-seed and mean +/- std results
- hardware and runtime details
- license information
