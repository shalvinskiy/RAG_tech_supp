"""Load articles, questions and ground-truth answers from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import settings


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        return f"{self.title}\n\n{self.text}"


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    answer: str


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def load_articles(path: Path | None = None) -> list[Article]:
    raw = _load_json(path or settings.articles_path)
    return [
        Article(id=str(item["id"]), title=item["title"], text=item["text"])
        for item in raw
    ]


def load_qa(path_questions: Path | None = None, path_answers: Path | None = None) -> list[QAItem]:
    questions = _load_json(path_questions or settings.questions_path)
    answers = _load_json(path_answers or settings.ground_truth_path)
    answer_by_id = {str(item["id"]): item["answer"] for item in answers}

    items: list[QAItem] = []
    missing: list[str] = []
    for q in questions:
        qid = str(q["id"])
        if qid not in answer_by_id:
            missing.append(qid)
            continue
        items.append(QAItem(id=qid, question=q["question"], answer=answer_by_id[qid]))

    if missing:
        raise ValueError(f"Missing ground-truth answers for: {missing}")
    return items


def iter_limited(items: Iterable, limit: int | None) -> list:
    seq = list(items)
    if limit is None or limit <= 0:
        return seq
    return seq[:limit]
