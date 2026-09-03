#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
python -m evitrust_vqa.cli smoke --fixture data/fixtures/tiny_fixture.json

