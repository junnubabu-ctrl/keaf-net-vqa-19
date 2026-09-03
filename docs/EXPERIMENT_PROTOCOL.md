# Locked experiment protocol

## Datasets and splits

Use official train/validation/test splits. Calibration and selective-threshold fitting use only the
declared validation partition. The final test partition is touched once after endpoints, thresholds,
corruption rates, baselines, seeds, and analysis scripts are frozen.

## Required comparisons

- No-knowledge answerer.
- Unfiltered retrieved evidence.
- Relevance-only filtering.
- Provenance-only calibration.
- Conflict-aware verification without selective answering.
- Full EviTrust-VQA.
- Directly comparable published baselines under matched backbone, corpus, and retrieval budget.

## Corruptions

Evaluate irrelevant, duplicate, contradictory, missing, and stale/source-swapped evidence at
10%, 25%, and 50%. Generate each corruption from a fixed seed and save the affected evidence IDs.

## Endpoints

Primary: official answer score, evidence macro-F1/MCC, AURC, and robustness degradation.
Secondary: Recall@k, MRR, nDCG@k, sufficiency/comprehensiveness, NLL, Brier score, ECE, coverage,
latency, peak memory, parameter count, and index/storage size.

Run at least three independent training seeds; five are preferred for the lightweight verifier.
Report mean, standard deviation, 95% confidence intervals, effect sizes, and paired bootstrap
comparisons (10,000 replicates recommended). Apply Holm correction to comparison families.

