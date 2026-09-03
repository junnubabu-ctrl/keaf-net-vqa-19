"""Dataset/knowledge manifest validation and checksum enforcement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runlog import sha256_file


UNRESOLVED_MARKERS = {"", "UNSET", "VERIFY_UPSTREAM", None}


@dataclass(frozen=True)
class ManifestFinding:
    path: str
    message: str


@dataclass(frozen=True)
class ManifestValidation:
    valid: bool
    findings: tuple[ManifestFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "findings": [finding.__dict__ for finding in self.findings],
        }


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    return payload


def _walk_unresolved(value: Any, path: str = "$") -> list[ManifestFinding]:
    findings: list[ManifestFinding] = []
    if isinstance(value, dict):
        for key, child in value.items():
            findings.extend(_walk_unresolved(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_walk_unresolved(child, f"{path}[{index}]"))
    elif value in UNRESOLVED_MARKERS:
        findings.append(ManifestFinding(path, "unresolved required value"))
    return findings


def validate_manifest(
    path: str | Path,
    *,
    verify_local_files: bool = False,
    workspace: str | Path | None = None,
) -> ManifestValidation:
    manifest_path = Path(path)
    payload = load_manifest(manifest_path)
    findings = _walk_unresolved(payload)
    if payload.get("schema_version") != "1.0":
        findings.append(ManifestFinding("$.schema_version", "expected schema_version 1.0"))

    if verify_local_files:
        root = Path(workspace or manifest_path.parent).resolve()
        records = payload.get("datasets", payload.get("sources", []))
        for index, record in enumerate(records):
            local_path = record.get("local_path")
            expected = record.get("sha256")
            if local_path in UNRESOLVED_MARKERS or expected in UNRESOLVED_MARKERS:
                continue
            candidate = (root / str(local_path)).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                findings.append(
                    ManifestFinding(f"$[{index}].local_path", "path escapes declared workspace")
                )
                continue
            if not candidate.is_file():
                findings.append(ManifestFinding(f"$[{index}].local_path", "file not found"))
            elif sha256_file(candidate) != expected:
                findings.append(ManifestFinding(f"$[{index}].sha256", "checksum mismatch"))
    return ManifestValidation(not findings, tuple(findings))

