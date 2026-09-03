from __future__ import annotations

import json
from pathlib import Path

from evitrust_vqa.schemas import EvidenceRecord


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "tiny_fixture.json"


def load_evidence() -> list[EvidenceRecord]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [EvidenceRecord.from_dict(item) for item in payload["evidence"]]

