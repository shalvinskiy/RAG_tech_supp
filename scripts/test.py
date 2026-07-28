#!/usr/bin/env python3
"""Smoke-test: load data and ask one question via selected backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_articles, load_qa
from src.models.base import ZERO_SHOT_SYSTEM, build_zero_shot_prompt
from src.models.factory import get_llm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gemini", "ollama"], default="gemini")
    parser.add_argument("--question-id", default="q1")
    args = parser.parse_args()

    articles = load_articles()
    qa = {item.id: item for item in load_qa()}
    item = qa[args.question_id]

    print(f"Articles: {len(articles)}")
    print(f"QA pairs: {len(qa)}")
    print(f"Q: {item.question}")
    print(f"GT: {item.answer}")

    llm = get_llm(args.backend)
    out = llm.generate(build_zero_shot_prompt(item.question), system=ZERO_SHOT_SYSTEM)
    print(f"\n[{out.backend}/{out.model}] {out.text}")


if __name__ == "__main__":
    main()
