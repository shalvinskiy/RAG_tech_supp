#!/usr/bin/env python3
"""Zero-shot evaluation without RAG (Gemini API and/or local Ollama)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import iter_limited, load_qa
from src.metrics.evaluate_utils import (
    default_results_path,
    run_generation,
    save_report,
    score_rows,
)
from src.models.base import ZERO_SHOT_SYSTEM, build_zero_shot_prompt
from src.models.factory import get_llm
from src.models.ollama_client import check_ollama_health


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot LLM evaluation")
    parser.add_argument(
        "--backend",
        choices=["gemini", "ollama"],
        required=True,
        help="Inference backend",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge")
    parser.add_argument("--out", type=str, default="", help="Output JSON path")
    args = parser.parse_args()

    if args.backend == "ollama" and not check_ollama_health():
        print(
            "WARNING: Ollama endpoint looks unreachable. "
            "Run `ollama serve`, pull the model, and set OLLAMA_BASE_URL in .env"
        )

    llm = get_llm(args.backend)
    items = iter_limited(load_qa(), args.limit or None)
    print(f"Backend={args.backend} model={llm.name} n={len(items)}")

    rows = run_generation(
        llm,
        items,
        prompt_fn=lambda qa: build_zero_shot_prompt(qa.question),
        system=ZERO_SHOT_SYSTEM,
        desc=f"zeroshot/{args.backend}",
    )
    report = score_rows(rows, use_judge=not args.no_judge)
    report["mode"] = "zeroshot"
    report["backend"] = args.backend
    report["model"] = llm.name

    out = Path(args.out) if args.out else default_results_path("zeroshot", args.backend)
    save_report(report, out)

    s = report["summary"]
    print("\n=== Zero-shot summary ===")
    print(f"BLEU:          {s['bleu']:.4f}")
    print(f"ROUGE-1:       {s['rouge1']:.4f}")
    print(f"ROUGE-2:       {s['rouge2']:.4f}")
    print(f"ROUGE-L:       {s['rougeL']:.4f}")
    if "llm_judge_mean_0_5" in s:
        print(f"LLM-judge:     {s['llm_judge_mean_0_5']:.3f} / 5")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
