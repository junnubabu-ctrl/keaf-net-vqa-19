# Paper 4 publication table plan

This document defines the tables that the implementation must generate from immutable experiment artifacts. Numbers are never hand-entered into manuscript tables. Every reported cell must trace to a run manifest, prediction file, evidence-decision log, or statistics artifact.

## Scope

Paper 4: **Evidence-Gated Knowledge Verification for Hallucination-Resistant Large Language Model-Augmented Visual Question Answering**.

Primary benchmark datasets: OK-VQA, A-OKVQA, and KRVQA. Papers 1-3 remain unchanged. Paper 4 evaluates evidence verification, grounding, corruption robustness, calibration, and selective answering.

## Main-manuscript tables

### Table 1. Dataset and evaluation protocol
Columns:
- Dataset
- Knowledge requirement / task characteristic
- Official split(s) used
- Number of questions used in each split
- Image source / image split
- Answer format
- Primary answer metric
- Calibration split
- Test-touch policy
- Dataset / annotation version
- Manifest checksum

Purpose: establishes that datasets, versions, splits, and evaluation rules were fixed before final testing.

### Table 2. Knowledge sources and retrieval configuration
Columns:
- Knowledge source
- Source type
- Snapshot / release date
- Corpus size
- Passage / triple unit
- Retrieval method
- Top-k
- Index configuration
- Provenance fields retained
- License / access note
- Manifest checksum

Purpose: makes external evidence reproducible and separates source reliability from retrieval relevance.

### Table 3. Compared systems and controlled variables
Rows:
1. No-knowledge answerer
2. Unfiltered retrieval
3. Relevance-only filtering
4. Provenance-only calibration
5. Conflict-aware verification without selective answering
6. Full EviTrust-VQA
7+. Directly comparable published baselines under matched backbone, corpus, and retrieval budget

Columns:
- System
- Visual backbone
- Text / multimodal backbone
- Retriever
- Knowledge budget
- Provenance calibration
- Conflict graph
- Sufficiency gate
- Selective answering
- Trainable parameters
- Notes on matched controls

Purpose: prevents unfair baseline comparisons.

### Table 4. Main VQA performance
Rows: each compared system.
Columns grouped by dataset:
- OK-VQA official answer score: mean, SD, 95% CI
- A-OKVQA official direct-answer score: mean, SD, 95% CI
- KRVQA official answer score: mean, SD, 95% CI
- Macro average where defensible

Additional columns:
- Number of seeds
- Best validation epoch selection rule

Purpose: answers whether the proposed verification pipeline improves answer quality under the locked protocol.

### Table 5. Evidence verification quality
Rows: relevant systems / verifier ablations.
Columns:
- Evidence precision
- Evidence recall
- Evidence macro-F1
- MCC
- AUROC when labels permit
- Support F1
- Refute F1
- Neutral F1
- Source-stratified macro-F1

Purpose: tests the mechanism directly rather than inferring it only from answer accuracy.

### Table 6. Selective answering and hallucination-risk control
Rows: no selective gate, confidence-only gate, sufficiency-only gate, full selective gate.
Columns:
- Coverage at target risk
- Selective risk
- AURC
- Accuracy on answered subset
- Abstention rate
- Unsupported-answer rate
- Grounded-answer rate
- Risk at fixed coverage levels (e.g., 50%, 70%, 90%)

Purpose: establishes whether abstention reduces unsupported answers without hiding poor performance.

### Table 7. Robustness to evidence corruption
Rows grouped by corruption family:
- Irrelevant evidence
- Duplicate evidence
- Contradictory evidence
- Missing evidence
- Stale / source-swapped evidence

Columns for corruption rate 0%, 10%, 25%, 50%:
- Answer-score change
- Evidence-F1 change
- AURC change
- Unsupported-answer-rate change
- Coverage change

Purpose: tests the central reliability claim under controlled evidence failures.

### Table 8. A1-A9 ablation study
Required ablations:
- A1 remove provenance prior
- A2 remove support/contradiction graph
- A3 remove signed message propagation
- A4 remove source-diversity term
- A5 remove query-coverage term
- A6 remove set-level sufficiency
- A7 remove answer-grounding score
- A8 replace validation-calibrated selective threshold with fixed confidence threshold
- A9 full model

Columns:
- Answer score
- Evidence macro-F1
- MCC
- AURC
- Unsupported-answer rate
- Coverage
- Robustness degradation at 25% contradiction

Purpose: identifies which components causally contribute to reliability.

### Table 9. Calibration quality
Rows: raw confidence, post-hoc calibrated confidence, full selective confidence.
Columns:
- NLL
- Brier score
- ECE
- MCE (supplementary if space is limited)
- Reliability slope/intercept where estimated

Purpose: demonstrates that confidence values used for abstention are empirically calibrated.

### Table 10. Statistical significance and effect sizes
Rows: predeclared pairwise comparisons, primarily full EviTrust-VQA versus strongest matched baseline and key ablations.
Columns:
- Dataset
- Endpoint
- Mean paired difference
- 95% bootstrap CI
- Raw p-value
- Holm-adjusted p-value
- Effect size
- Direction
- Significant after correction (yes/no)

Purpose: prevents conclusions based on single-seed or uncorrected comparisons.

### Table 11. Computational efficiency and resource footprint
Rows: principal compared systems.
Columns:
- Trainable parameters
- Total parameters
- Retrieval latency / question
- Verification latency / question
- End-to-end latency / question
- Peak CPU RAM
- Peak GPU memory
- Index size
- Model / checkpoint size
- Energy or GPU-hours if measured

Purpose: quantifies practical cost of reliability improvements.

### Table 12. Error and hallucination taxonomy
Rows:
- Visual recognition error
- Retrieval miss
- Irrelevant evidence dominance
- Contradictory evidence failure
- Stale knowledge
- Verifier false accept
- Verifier false reject
- Insufficient evidence not abstained
- Answerer grounding failure
- Annotation ambiguity / other

Columns:
- Count
- Percentage
- Baseline frequency
- Full-model frequency
- Relative change
- Representative trace ID

Purpose: supplies qualitative-mechanistic evidence without cherry-picking examples.

## Supplementary / appendix tables

### Table S1. Dataset file inventory and checksums
File path, role, byte size, SHA-256, version, acquisition source, split.

### Table S2. Hyperparameters and search ranges
Parameter, candidate range, selected value, selection split, selection criterion.

### Table S3. Seed-level results
One row per dataset × system × seed, including all primary endpoints.

### Table S4. Retrieval quality by k
Recall@k, MRR, nDCG@k, evidence recall, latency for k in the predeclared retrieval-budget grid.

### Table S5. Source-stratified verifier performance
Per source type / source reliability bucket: precision, recall, F1, MCC, calibration error.

### Table S6. Corruption robustness by dataset and seed
Raw non-aggregated robustness values to make Table 7 auditable.

### Table S7. Cross-dataset transfer / generalization
Train dataset, calibration dataset, test dataset, answer score, AURC, evidence F1, coverage. Run only if the protocol supports a leakage-safe transfer experiment.

### Table S8. Reproducibility audit
Run ID, git commit, config digest, dataset manifest digest, knowledge manifest digest, seed, environment, hardware, start/end time, artifact checksums, status.

### Table S9. Data leakage and integrity checks
Check name, split, expected invariant, observed result, pass/fail, artifact reference.

## Generation rule

The manuscript table generator must:
1. read machine-generated JSONL/CSV experiment artifacts only;
2. reject rows missing commit/config/dataset/knowledge hashes;
3. aggregate by dataset/system/seed without silently dropping failed runs;
4. compute confidence intervals and corrected paired tests from the locked analysis code;
5. mark unavailable results as `NA`, never fabricate or interpolate values;
6. preserve raw seed-level artifacts for every aggregate result.
