# PhD-level Paper 4 implementation roadmap

## Objective
Complete the executable and auditable implementation required to evaluate evidence verification and selective answering for knowledge-based VQA without altering Papers 1-3 or inventing benchmark results.

## Work packages

### WP1 — Dataset layer
- OK-VQA adapter
- A-OKVQA adapter
- KRVQA adapter with explicit field mapping
- official split validation
- image-path resolver
- SHA-256 dataset manifest generation
- leakage checks across train/validation/test identifiers

### WP2 — Knowledge layer
- knowledge snapshot manifests
- passage/triple normalization
- provenance retention (source, snapshot, URI/identifier, license metadata)
- deterministic corpus chunking
- corpus/index checksums

### WP3 — Retrieval
- lexical baseline (BM25 or equivalent)
- dense text retrieval adapter
- multimodal query representation adapter where justified
- frozen retrieval budget and top-k grid
- Recall@k, MRR and nDCG@k instrumentation

### WP4 — Evidence verification
- current provenance calibrator
- current signed evidence graph
- learned/lightweight verifier adapter
- support/refute/neutral labels
- validation-only calibration
- source-stratified diagnostics

### WP5 — Answering
- no-knowledge baseline
- unfiltered retrieval baseline
- relevance-only baseline
- evidence-gated answerer
- LLM/VLM adapter behind an explicit interface
- prompt/template version hashing
- no ground-truth answer in inference inputs

### WP6 — Selective answering
- validation-fit confidence calibration
- sufficiency gate
- grounding gate
- target-risk threshold selection
- risk-coverage curves and AURC

### WP7 — Robustness
- irrelevant, duplicate, contradictory, missing, and stale/source-swapped evidence
- rates: 10%, 25%, 50%
- deterministic corruption manifests listing modified evidence IDs

### WP8 — Experiment runner
- dataset × system × seed matrix
- at least 3 independent training seeds; 5 preferred for lightweight verifier
- immutable run manifest
- prediction JSONL
- evidence-decision JSONL
- latency/memory resource log
- failed-run retention (no silent omission)

### WP9 — Statistical analysis
- per-seed aggregation
- mean, SD and 95% confidence intervals
- paired bootstrap (10,000 replicates recommended)
- effect sizes
- Holm correction for comparison families
- predeclared comparisons

### WP10 — Publication artifacts
- machine-generated Tables 1-12 and S1-S9
- risk-coverage curve
- calibration/reliability plot
- robustness degradation plot
- architecture diagram source
- qualitative audit traces selected by deterministic criteria

### WP11 — Reproducibility
- environment lock
- hardware capture
- git commit/config digest/dataset digest/knowledge digest in every run
- CI unit/integration tests
- deterministic smoke fixture
- README reproduction commands
- claim ledger linking every manuscript number to an artifact

## Completion gate
The implementation is not considered publication-complete until:
1. licensed datasets are locally installed and manifests validated;
2. all benchmark adapters pass integrity tests;
3. matched baselines run under the locked retrieval/backbone budget;
4. the complete multi-seed experiment matrix finishes;
5. manuscript tables are generated from run artifacts;
6. statistical tests complete without manual cell editing;
7. every numerical manuscript claim has an artifact reference;
8. final test partitions are not reused for tuning.
