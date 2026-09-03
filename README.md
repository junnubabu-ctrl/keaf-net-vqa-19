# EviTrust-VQA

Reference implementation for **Evidence-Aware Knowledge Verification for Reliable
Knowledge-Enhanced Visual Question Answering** (Paper 4).

The repository implements the paper's testable mechanism: provenance-aware evidence records,
a signed support/contradiction graph, set-level sufficiency, answer grounding, and
validation-calibrated selective answering. It does **not** claim benchmark improvements. Any
reported result must be generated from immutable experiment logs after the declared datasets,
knowledge snapshots, baselines, and seeds are installed.

## Research boundary

- Paper 4 concerns evidence reliability and answer grounding.
- Adaptive question-dependent reasoning depth is reserved for Paper 5 and is not claimed here.
- Paper 3 values such as 68.7% on OK-VQA and 62.4% on A-OKVQA are unverified manuscript
  claims. They are neither encoded as constants nor presented as reproduced results.
- The core implementation is offline and uses no OpenAI API or proprietary judge.

## Implemented components

1. Unified `EvidenceRecord` schema with immutable provenance.
2. Deterministic signed evidence graph for support, contradiction, and redundancy.
3. Provenance-calibrated evidence scoring with signed message propagation.
4. Set-level evidence sufficiency and post-answer grounding.
5. Selective-answer calibration from validation data only.
6. Controlled corruption operators at declared rates.
7. Retrieval, verification, calibration, risk-coverage, robustness, and VQA metrics.
8. Structured run manifests and an auditable CPU fixture.
9. Dataset/knowledge manifest validation, paired bootstrap intervals, and Holm correction.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
evitrust-vqa smoke --fixture data/fixtures/tiny_fixture.json
evitrust-vqa corrupt-fixture --fixture data/fixtures/tiny_fixture.json --kind contradictory --rate 0.25
```

The smoke command emits JSON. It must produce the same evidence IDs, graph edges, scores, and
decision for the same seed and fixture.

## Repository map

```text
src/evitrust_vqa/   Core schemas, graph, verifier, calibration, metrics, pipeline and CLI
tests/               Unit and integration tests, including no-label-leakage checks
configs/             Versioned experiment configuration
data/fixtures/       Small synthetic/public-safe deterministic fixture
data/manifests/      Dataset and knowledge snapshot manifest templates
docs/                Architecture, experiments, integrity, datasets and claim ledger
scripts/             Reproducible local entry points
```

## Dataset installation

Full benchmark data and knowledge corpora are intentionally excluded. Follow
[`docs/DATASETS.md`](docs/DATASETS.md), accept each upstream license, and record checksums in
the manifest templates. Do not commit restricted images, annotations, checkpoints, indexes, or
credentials.

## Status

This is a research implementation baseline (v0.1.0), not a completed benchmark study. The CPU
fixture validates interfaces and calculations; publication claims require the full locked,
multi-seed protocol in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## License

Code is released under Apache License 2.0. Dataset, model-weight, and knowledge-source licenses
remain with their respective owners.
