"""BLEU / ROUGE metrics for generated answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LexicalScores:
    bleu: float
    rouge1: float
    rouge2: float
    rougeL: float


def _ensure_nltk() -> None:
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def compute_lexical_metrics(
    predictions: list[str],
    references: list[str],
) -> tuple[LexicalScores, list[LexicalScores]]:
    """Compute corpus-level and per-example BLEU/ROUGE."""
    _ensure_nltk()

    import evaluate

    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")

    # Corpus
    bleu_corpus = bleu_metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    rouge_corpus = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False,
    )
    corpus = LexicalScores(
        bleu=float(bleu_corpus["score"]) / 100.0,
        rouge1=float(rouge_corpus["rouge1"]),
        rouge2=float(rouge_corpus["rouge2"]),
        rougeL=float(rouge_corpus["rougeL"]),
    )

    # Per example
    per_item: list[LexicalScores] = []
    for pred, ref in zip(predictions, references):
        b = bleu_metric.compute(predictions=[pred], references=[[ref]])
        r = rouge_metric.compute(predictions=[pred], references=[ref], use_stemmer=False)
        per_item.append(
            LexicalScores(
                bleu=float(b["score"]) / 100.0,
                rouge1=float(r["rouge1"]),
                rouge2=float(r["rouge2"]),
                rougeL=float(r["rougeL"]),
            )
        )
    return corpus, per_item


def scores_to_dict(scores: LexicalScores) -> dict:
    return asdict(scores)
