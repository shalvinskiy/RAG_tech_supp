"""Central configuration loaded from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _path(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


class Settings:
    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # Ollama / OpenAI-compatible
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "ollama")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    # Judge
    judge_model: str = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")

    # RAG / paths
    top_k: int = int(os.getenv("TOP_K", "3"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    index_dir: Path = _path("INDEX_DIR", "indexes/faiss_articles")
    results_dir: Path = _path("RESULTS_DIR", "results")
    data_dir: Path = _path("DATA_DIR", "data")

    articles_path: Path = data_dir / "articles.json"
    questions_path: Path = data_dir / "questions.json"
    ground_truth_path: Path = data_dir / "ground_truth.json"

    # Generation defaults
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))


settings = Settings()
