#!/usr/bin/env python3
"""RAG evaluation (retrieve + generate) for Gemini API and/or local Ollama."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.config import settings
from src.data_loader import iter_limited, load_qa
from src.metrics.evaluate_utils import default_results_path, save_report, score_rows
from src.models.factory import get_llm
from src.models.ollama_client import check_ollama_health
from src.rag.index import load_index
from src.rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation")
    parser.add_argument(
        "--backend",
        choices=["gemini", "ollama"],
        required=True,
        help="Generator backend (same as in zero-shot for fair comparison)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if args.backend == "ollama" and not check_ollama_health():
        print(
            "WARNING: Ollama endpoint looks unreachable. "
            "Run `ollama serve`, pull the model, and set OLLAMA_BASE_URL in .env"
        )

    llm = get_llm(args.backend)
    store = load_index()
    pipeline = RAGPipeline(llm=llm, vectorstore=store, top_k=args.top_k)

    items = iter_limited(load_qa(), args.limit or None)
    print(f"Backend={args.backend} model={llm.name} n={len(items)} top_k={args.top_k}")

    rows = []
    for item in tqdm(items, desc=f"rag/{args.backend}"):
        result = pipeline.answer(item.question)
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "reference": item.answer,
                "prediction": result.answer,
                "retrieved_ids": [h.article_id for h in result.hits],
                "retrieved_titles": [h.title for h in result.hits],
                "model": result.model,
                "backend": result.backend,
            }
        )

    report = score_rows(rows, use_judge=not args.no_judge)
    report["mode"] = "rag"
    report["backend"] = args.backend
    report["model"] = llm.name
    report["top_k"] = args.top_k
    report["embedding_model"] = settings.embedding_model

    out = Path(args.out) if args.out else default_results_path("rag", args.backend)
    save_report(report, out)

    s = report["summary"]
    print("\n=== RAG summary ===")
    print(f"BLEU:          {s['bleu']:.4f}")
    print(f"ROUGE-1:       {s['rouge1']:.4f}")
    print(f"ROUGE-2:       {s['rouge2']:.4f}")
    print(f"ROUGE-L:       {s['rougeL']:.4f}")
    if "llm_judge_mean_0_5" in s:
        print(f"LLM-judge:     {s['llm_judge_mean_0_5']:.3f} / 5")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
