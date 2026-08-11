# Reproducibility protocol

This repository separates **implementation completeness** from **experimental verification**.

Before manuscript scores may be described as reproduced, archive the following under a DOI-backed release or immutable commit:

- exact OK-VQA/A-OKVQA split identifiers and SHA-256 hashes;
- image-region extractor name, version, weights, preprocessing, and feature hashes;
- ConceptNet/CSKG snapshot dates, licenses, retrieval queries, cache, and hashes;
- tokenizer, answer vocabulary, normalization, and evaluation script version;
- configuration and environment lock file for every run;
- at least three independent seeds, per-seed logs, raw predictions, and checkpoints;
- mean, standard deviation, confidence interval, and paired significance procedure;
- measured parameter count, FLOPs, latency, memory, and hardware details;
- ablation configs that change exactly one factor at a time.

## Determinism

Record Python, NumPy, and PyTorch seeds. Enable deterministic algorithms where supported and document any non-deterministic GPU kernels. Retrieval tie-breaking in `JsonlKnowledgeStore` is deterministic.

## Evidence boundary

The checked-in implementation is not evidence that manuscript tables have been reproduced. Add a signed release manifest only after all artifacts above have been independently checked.
