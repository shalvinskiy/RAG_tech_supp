#!/usr/bin/env python3
"""Pretty-print comparison of zero-shot vs RAG result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_from(report: dict) -> list:
    s = report["summary"]
    return [
        report.get("mode", "?"),
        report.get("backend", "?"),
        report.get("model", "?"),
        s.get("n", 0),
        f"{s.get('bleu', 0):.3f}",
        f"{s.get('rouge1', 0):.3f}",
        f"{s.get('rouge2', 0):.3f}",
        f"{s.get('rougeL', 0):.3f}",
        f"{s.get('llm_judge_mean_0_5', float('nan')):.3f}"
        if "llm_judge_mean_0_5" in s
        else "—",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path, help="Result JSON files")
    args = parser.parse_args()

    table = [row_from(load(p)) for p in args.reports]
    headers = ["mode", "backend", "model", "n", "BLEU", "R1", "R2", "RL", "Judge/5"]
    print(tabulate(table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
