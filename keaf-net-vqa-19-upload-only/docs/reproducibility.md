# Reproducibility Details to Report in the Manuscript

Add or confirm these details before submission:

- Python version and CUDA/cuDNN version
- GPU model and number of GPUs
- total training time per seed
- effective batch size and gradient accumulation
- optimizer, learning rate, scheduler, warmup ratio, weight decay
- early stopping patience and selected checkpoint criterion
- random seeds: 42, 123, 2024
- exact dataset splits
- answer vocabulary size: 3,129 most frequent answers
- external resources: ConceptNet 5.5 and CSKG
- feature extraction: ViT-B/16, Faster R-CNN Visual Genome, BERT-base, Sentence-BERT all-MiniLM-L6-v2

Recommended wording:

> To support reproducibility, we release configuration files, seed-control code, logging scripts, and evaluation scripts. All reported values are computed across three independent runs using seeds 42, 123, and 2024.
