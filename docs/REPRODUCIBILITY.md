# Reproducibility

Every full run must record: git commit, configuration hash, dataset/knowledge manifest hashes,
random seed, hardware, operating system, Python and package versions, start/end timestamps,
checkpoint hashes, metrics, and artifact paths.

The default code has no network call and no proprietary API dependency. CI uses the tiny fixture
and must not download full datasets or require a GPU. Production encoders/retrievers must be pinned
by immutable revision and accompanied by their license and weight checksum.

Release only source, configuration, small redistributable fixtures, manifests, and generated
aggregate results. Do not release restricted source data or licensed model weights.

