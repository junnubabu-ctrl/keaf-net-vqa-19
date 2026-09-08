"""Dataset adapters for Paper 4 benchmark execution.

The adapters deliberately avoid downloading or redistributing benchmark assets. They normalize
locally installed official annotations into a common schema and preserve split/image identifiers
needed for reproducible VQA evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class VQASample:
    dataset: str
    split: str
    question_id: str
    question: str
    image_id: str
    answers: tuple[str, ...]
    direct_answer: str | None = None
    choices: tuple[str, ...] = ()
    correct_choice_idx: int | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.dataset or not self.split:
            raise ValueError("dataset and split are required")
        if not self.question_id or not self.question or not self.image_id:
            raise ValueError("question_id, question and image_id are required")
        if self.correct_choice_idx is not None:
            if not self.choices:
                raise ValueError("correct_choice_idx requires choices")
            if not 0 <= self.correct_choice_idx < len(self.choices):
                raise ValueError("correct_choice_idx is out of range")


class DatasetAdapter(Protocol):
    dataset_name: str

    def load(self, split: str) -> list[VQASample]: ...


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_id(value: Any) -> str:
    if value is None:
        raise ValueError("missing required identifier")
    return str(value)


def _answers_from_vqa_annotation(annotation: Mapping[str, Any]) -> tuple[str, ...]:
    answers = annotation.get("answers", [])
    normalized: list[str] = []
    for item in answers:
        if isinstance(item, Mapping):
            value = item.get("answer")
        else:
            value = item
        if value is not None:
            text = str(value).strip()
            if text:
                normalized.append(text)
    return tuple(normalized)


class OKVQAAdapter:
    """Loader for official OK-VQA question and annotation JSON files."""

    dataset_name = "OK-VQA"

    def __init__(self, question_files: Mapping[str, str | Path], annotation_files: Mapping[str, str | Path]):
        self.question_files = {key: Path(value) for key, value in question_files.items()}
        self.annotation_files = {key: Path(value) for key, value in annotation_files.items()}

    def load(self, split: str) -> list[VQASample]:
        if split not in self.question_files or split not in self.annotation_files:
            raise KeyError(f"split {split!r} is not configured")
        q_payload = _read_json(self.question_files[split])
        a_payload = _read_json(self.annotation_files[split])
        questions = q_payload.get("questions", q_payload)
        annotations = a_payload.get("annotations", a_payload)
        if not isinstance(questions, list) or not isinstance(annotations, list):
            raise ValueError("expected official VQA list-based question/annotation structure")
        by_qid = {_as_id(item["question_id"]): item for item in annotations}
        samples: list[VQASample] = []
        for question in questions:
            qid = _as_id(question["question_id"])
            annotation = by_qid.get(qid)
            if annotation is None:
                raise ValueError(f"annotation missing for question_id={qid}")
            samples.append(
                VQASample(
                    dataset=self.dataset_name,
                    split=split,
                    question_id=qid,
                    question=str(question["question"]).strip(),
                    image_id=_as_id(question["image_id"]),
                    answers=_answers_from_vqa_annotation(annotation),
                    metadata={"question_type": annotation.get("question_type"), "answer_type": annotation.get("answer_type")},
                )
            )
        return samples


class AOKVQAAdapter:
    """Loader for A-OKVQA JSON annotations used for direct-answer evaluation."""

    dataset_name = "A-OKVQA"

    def __init__(self, split_files: Mapping[str, str | Path]):
        self.split_files = {key: Path(value) for key, value in split_files.items()}

    def load(self, split: str) -> list[VQASample]:
        if split not in self.split_files:
            raise KeyError(f"split {split!r} is not configured")
        payload = _read_json(self.split_files[split])
        records = payload.get("questions", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(records, list):
            raise ValueError("expected an A-OKVQA list of records")
        samples: list[VQASample] = []
        for item in records:
            qid = _as_id(item.get("question_id", item.get("id")))
            choices = tuple(str(x) for x in item.get("choices", []) if x is not None)
            answer_idx = item.get("correct_choice_idx")
            direct_answers = item.get("direct_answers", item.get("answers", []))
            if isinstance(direct_answers, str):
                direct_answers = [direct_answers]
            answers = tuple(str(x).strip() for x in direct_answers if str(x).strip())
            direct = answers[0] if answers else None
            samples.append(
                VQASample(
                    dataset=self.dataset_name,
                    split=split,
                    question_id=qid,
                    question=str(item["question"]).strip(),
                    image_id=_as_id(item.get("image_id")),
                    answers=answers,
                    direct_answer=direct,
                    choices=choices,
                    correct_choice_idx=int(answer_idx) if answer_idx is not None else None,
                    metadata={"rationales": item.get("rationales", [])},
                )
            )
        return samples


class KRVQAAdapter:
    """Configurable loader for KRVQA-style records.

    Public KRVQA distributions have appeared in more than one serialization shape. Rather than
    silently guessing fields, this adapter accepts a field map and validates every required field.
    """

    dataset_name = "KRVQA"

    def __init__(self, split_files: Mapping[str, str | Path], *, field_map: Mapping[str, str] | None = None):
        self.split_files = {key: Path(value) for key, value in split_files.items()}
        self.field_map = {
            "question_id": "question_id",
            "question": "question",
            "image_id": "image_id",
            "answer": "answer",
            **dict(field_map or {}),
        }

    def load(self, split: str) -> list[VQASample]:
        if split not in self.split_files:
            raise KeyError(f"split {split!r} is not configured")
        payload = _read_json(self.split_files[split])
        records = payload.get("questions", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(records, list):
            raise ValueError("expected a list of KRVQA records")
        samples: list[VQASample] = []
        for item in records:
            def field(name: str) -> Any:
                key = self.field_map[name]
                if key not in item:
                    raise ValueError(f"KRVQA field {key!r} missing from record")
                return item[key]

            answer_value = field("answer")
            answers = tuple(str(x).strip() for x in answer_value) if isinstance(answer_value, list) else (str(answer_value).strip(),)
            samples.append(
                VQASample(
                    dataset=self.dataset_name,
                    split=split,
                    question_id=_as_id(field("question_id")),
                    question=str(field("question")).strip(),
                    image_id=_as_id(field("image_id")),
                    answers=answers,
                    direct_answer=answers[0] if answers else None,
                    metadata={key: value for key, value in item.items() if key not in set(self.field_map.values())},
                )
            )
        return samples


def iter_samples(adapter: DatasetAdapter, splits: Iterable[str]) -> Iterable[VQASample]:
    for split in splits:
        yield from adapter.load(split)
