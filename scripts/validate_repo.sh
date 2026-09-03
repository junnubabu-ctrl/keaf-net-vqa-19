#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q src tests
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m evitrust_vqa.cli smoke --fixture data/fixtures/tiny_fixture.json >/dev/null
python -m json.tool configs/default.json >/dev/null
python -m json.tool data/manifests/datasets.example.json >/dev/null
python -m json.tool data/manifests/knowledge.example.json >/dev/null
echo "Repository validation passed."

