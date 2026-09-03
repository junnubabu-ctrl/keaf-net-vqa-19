# Dataset and knowledge installation

Required primary benchmarks are OK-VQAv2, A-OKVQA, Encyclopedic-VQA, and
VLM-DeflectionBench. InfoSeek and CRAG-MM-Diagnostics are secondary generalization/diagnostic
sets. Download every resource from its official project page, accept its license, retain its
original split, and record version, URL, license, size, and SHA-256 checksum.

1. Copy `data/manifests/datasets.example.json` to an ignored run-specific manifest.
2. Replace every `UNSET` field; no unresolved manifest may enter a full experiment.
3. Store full data below `data/raw/`; this directory is ignored.
4. Generate processed data deterministically and record source-manifest hashes.
5. Never redistribute restricted images, annotations, corpora, or authentication tokens.

Knowledge sources must be fixed snapshots. The planned minimum is one structured source
(ConceptNet or CSKG) and one encyclopedic source (Wikidata/Wikipedia). Live web retrieval is not
permitted in the primary comparison because it breaks repeatability and can cause temporal leakage.

