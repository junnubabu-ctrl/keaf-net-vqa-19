# KEAF-Net

Reference implementation of **Knowledge-Enhanced Adaptive Fusion Network for Visual Question Answering with Multi-Hop Graph Reasoning**.

The repository implements the paper's core research pipeline:

1. pluggable external-knowledge retrieval (ConceptNet/CSKG adapters plus an offline JSONL store);
2. Adaptive Knowledge Filtering (AKF) with deterministic top-*k* selection;
3. Heterogeneous Graph Adaptive Fusion (HGAF) over visual, textual, and knowledge nodes;
4. recurrent Multi-Hop Semantic Reasoning (MHSR); and
5. answer classification, training, evaluation, checkpointing, and prediction export.

> **Research status:** this code is a clean reference implementation. The numerical results reported in the manuscript are not claimed as reproduced here. Reproduction requires the authors' exact dataset splits, extracted features, retrieval snapshots, checkpoints, seeds, and raw predictions.

## Quick start

Python 3.10+ is supported.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python -m keaf_net.smoke
python -m unittest discover -s tests -v
```

Optional PyTorch training support:

```bash
pip install -e '.[train]'
python scripts/train.py --config configs/okvqa.yaml
python scripts/evaluate.py --config configs/okvqa.yaml --checkpoint runs/best.pt
```

## Expected data format

Each JSONL record contains pre-extracted region features and token IDs:

```json
{"question_id": 1, "visual": [[0.1, 0.2]], "question": [12, 47], "facts": [[0.3, 0.4]], "answer": 5}
```

`visual` and `facts` are projected to the configured hidden dimension. Production experiments should pin the precise feature extractor, vocabulary, knowledge snapshot, and split hashes.

## Repository map

- `src/keaf_net/model.py`: end-to-end AKF → HGAF → MHSR model
- `src/keaf_net/modules.py`: research modules and masking utilities
- `src/keaf_net/knowledge.py`: deterministic retrieval and caching interfaces
- `src/keaf_net/data.py`: JSONL dataset/collation
- `src/keaf_net/engine.py`: training and evaluation loops
- `configs/`: experiment configuration templates
- `tests/`: unit and integration tests
- `docs/REPRODUCIBILITY.md`: evidence and release checklist

## Citation

The manuscript is under communication. A verified BibTeX record will be added after publication. Until then, cite the repository URL and commit hash used in your experiment.

## License

Apache License 2.0. See [LICENSE](LICENSE).
