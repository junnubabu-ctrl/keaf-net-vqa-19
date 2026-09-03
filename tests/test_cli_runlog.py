from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evitrust_vqa.cli import smoke
from evitrust_vqa.runlog import RunManifest, atomic_write_json, manifest_payload, sha256_file

from helpers import FIXTURE, ROOT


class CliRunlogTests(unittest.TestCase):
    def test_smoke_keeps_evaluation_fields_outside_inference(self) -> None:
        result = smoke(FIXTURE)
        self.assertEqual(result["answer"], "London")
        self.assertEqual(
            result["fixture_evaluation_fields_present_but_not_passed_to_inference"],
            ["gold_answers"],
        )

    def test_atomic_json_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"answer": "London"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["answer"], "London")
            self.assertEqual(len(sha256_file(path)), 64)

    def test_manifest_records_reproducibility_fields(self) -> None:
        manifest = RunManifest.capture(
            run_id="unit-test",
            repo=ROOT,
            config_sha256="a" * 64,
            data_manifest_sha256="b" * 64,
            seed=7,
        )
        payload = manifest_payload(manifest, {"metric": 1.0})
        self.assertEqual(payload["manifest"]["seed"], 7)
        self.assertIn("python", payload["manifest"])
        self.assertEqual(payload["results"]["metric"], 1.0)


if __name__ == "__main__":
    unittest.main()

