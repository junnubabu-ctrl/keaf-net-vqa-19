"""Immutable-style run manifests and atomic JSON result writing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    started_at_utc: str
    git_commit: str
    config_sha256: str
    data_manifest_sha256: str
    seed: int
    python: str
    platform: str
    command: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        *,
        run_id: str,
        repo: str | Path,
        config_sha256: str,
        data_manifest_sha256: str,
        seed: int,
    ) -> "RunManifest":
        return cls(
            run_id=run_id,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_commit(Path(repo)),
            config_sha256=config_sha256,
            data_manifest_sha256=data_manifest_sha256,
            seed=seed,
            python=sys.version,
            platform=platform.platform(),
            command=tuple(sys.argv),
        )


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def manifest_payload(manifest: RunManifest, results: dict[str, Any]) -> dict[str, Any]:
    return {"manifest": asdict(manifest), "results": results}
