"""LLM-as-judge scoring for factual correctness of answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.config import settings
from src.models.base import BaseLLM
from src.models.gemini_client import GeminiLLM


JUDGE_SYSTEM = (
    "Ты строгий судья качества ответов. "
    "Оцени, насколько prediction совпадает по смыслу и фактам с reference. "
    "Верни ТОЛЬКО JSON без markdown."
)

JUDGE_PROMPT = """Вопрос: {question}

Эталонный ответ (reference):
{reference}

Ответ модели (prediction):
{prediction}

Оцени по шкале 0..5:
0 — полностью неверно / галлюцинация
1 — почти неверно
2 — частично верно, много ошибок
3 — в целом верно, но есть существенные пропуски/неточности
4 — верно, мелкие отличия формулировки
5 — полностью эквивалентно по фактам

Верни JSON вида:
{{"score": <int 0-5>, "rationale": "<кратко на русском>"}}
"""


@dataclass
class JudgeScore:
    score: float  # 0..5
    rationale: str
    normalized: float  # 0..1


def _parse_judge_response(text: str) -> JudgeScore:
    text = text.strip()
    # Strip optional markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return JudgeScore(score=0.0, rationale=f"parse_error: {text[:200]}", normalized=0.0)
        data = json.loads(match.group(0))

    raw = float(data.get("score", 0))
    raw = max(0.0, min(5.0, raw))
    rationale = str(data.get("rationale", "")).strip()
    return JudgeScore(score=raw, rationale=rationale, normalized=raw / 5.0)


class LLMJudge:
    def __init__(self, llm: BaseLLM | None = None) -> None:
        if llm is not None:
            self.llm = llm
        else:
            # Prefer Gemini as judge if key is available
            if settings.gemini_api_key:
                self.llm = GeminiLLM(model=settings.judge_model)
            else:
                raise ValueError("No judge LLM configured. Set GEMINI_API_KEY.")

    def score(self, question: str, reference: str, prediction: str) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            prediction=prediction,
        )
        result = self.llm.generate(prompt, system=JUDGE_SYSTEM)
        return _parse_judge_response(result.text)
