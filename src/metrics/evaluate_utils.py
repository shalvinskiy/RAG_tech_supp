"""Aggregate evaluation helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import settings
from src.data_loader import QAItem
from src.metrics.lexical import compute_lexical_metrics, scores_to_dict
from src.metrics.llm_judge import LLMJudge
from src.models.base import BaseLLM


def run_generation(
    llm: BaseLLM,
    items: list[QAItem],
    *,
    prompt_fn,
    system: str,
    desc: str = "generate",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in tqdm(items, desc=desc):
        prompt = prompt_fn(item)
        out = llm.generate(prompt, system=system)
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "reference": item.answer,
                "prediction": out.text,
                "model": out.model,
                "backend": out.backend,
            }
        )
    return rows


def score_rows(
    rows: list[dict[str, Any]],
    *,
    use_judge: bool = True,
    judge: LLMJudge | None = None,
) -> dict[str, Any]:
    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    corpus, per_item = compute_lexical_metrics(preds, refs)

    judge_scores = []
    if use_judge:
        judge = judge or LLMJudge()
        for row in tqdm(rows, desc="llm-judge"):
            js = judge.score(row["question"], row["reference"], row["prediction"])
            row["judge_score"] = js.score
            row["judge_normalized"] = js.normalized
            row["judge_rationale"] = js.rationale
            judge_scores.append(js.normalized)
        for row, lex in zip(rows, per_item):
            row["bleu"] = lex.bleu
            row["rouge1"] = lex.rouge1
            row["rouge2"] = lex.rouge2
            row["rougeL"] = lex.rougeL
    else:
        for row, lex in zip(rows, per_item):
            row["bleu"] = lex.bleu
            row["rouge1"] = lex.rouge1
            row["rouge2"] = lex.rouge2
            row["rougeL"] = lex.rougeL

    summary = {
        "n": len(rows),
        "bleu": corpus.bleu,
        "rouge1": corpus.rouge1,
        "rouge2": corpus.rouge2,
        "rougeL": corpus.rougeL,
    }
    if judge_scores:
        summary["llm_judge_mean"] = sum(judge_scores) / len(judge_scores)
        summary["llm_judge_mean_0_5"] = summary["llm_judge_mean"] * 5.0

    return {"summary": summary, "lexical_corpus": scores_to_dict(corpus), "rows": rows}


def save_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **report,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def default_results_path(prefix: str, backend: str) -> Path:
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_backend = backend.replace("/", "_")
    return settings.results_dir / f"{prefix}_{safe_backend}_{stamp}.json"
