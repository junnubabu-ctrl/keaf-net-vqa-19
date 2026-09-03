"""Command-line entry point for CPU smoke and fixture inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .answering import InMemoryRetriever
from .config import EviTrustConfig
from .corruption import EvidenceCorruptor
from .manifests import validate_manifest
from .pipeline import EviTrustPipeline, InferenceRequest
from .schemas import EvidenceRecord


def load_fixture(path: str | Path) -> tuple[InferenceRequest, list[EvidenceRecord], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    question = payload["question"]
    request = InferenceRequest(
        question_id=question["question_id"],
        question=question["text"],
        image_id=question.get("image_id"),
    )
    evidence = [EvidenceRecord.from_dict(item) for item in payload["evidence"]]
    return request, evidence, payload.get("evaluation_only", {})


def smoke(fixture: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    request, evidence, evaluation = load_fixture(fixture)
    config = EviTrustConfig.from_json(config_path) if config_path else EviTrustConfig()
    pipeline = EviTrustPipeline(InMemoryRetriever(evidence), config=config)
    result = pipeline.infer(request)
    payload = result.to_dict()
    payload["fixture_evaluation_fields_present_but_not_passed_to_inference"] = sorted(evaluation)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evitrust-vqa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke", help="run the deterministic CPU fixture")
    smoke_parser.add_argument("--fixture", required=True)
    smoke_parser.add_argument("--config")
    manifest_parser = subparsers.add_parser("validate-manifest", help="validate a data manifest")
    manifest_parser.add_argument("--manifest", required=True)
    manifest_parser.add_argument("--verify-local-files", action="store_true")
    manifest_parser.add_argument("--workspace")
    corrupt_parser = subparsers.add_parser("corrupt-fixture", help="run one corruption fixture")
    corrupt_parser.add_argument("--fixture", required=True)
    corrupt_parser.add_argument("--kind", required=True, choices=sorted(EvidenceCorruptor.SUPPORTED))
    corrupt_parser.add_argument("--rate", required=True, type=float)
    corrupt_parser.add_argument("--seed", type=int, default=2_300_106_037)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(smoke(args.fixture, args.config), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-manifest":
        result = validate_manifest(
            args.manifest,
            verify_local_files=args.verify_local_files,
            workspace=args.workspace,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.valid else 2
    if args.command == "corrupt-fixture":
        request, evidence, _ = load_fixture(args.fixture)
        corruption = EvidenceCorruptor().apply(
            evidence,
            kind=args.kind,
            rate=args.rate,
            seed=args.seed,
        )
        pipeline = EviTrustPipeline(InMemoryRetriever(list(corruption.evidence)))
        payload = pipeline.infer(request).to_dict()
        payload["corruption"] = {
            "kind": corruption.kind,
            "rate": corruption.rate,
            "seed": corruption.seed,
            "affected_ids": list(corruption.affected_ids),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
