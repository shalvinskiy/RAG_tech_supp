"""LangChain RAG pipeline: retrieve + generate."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS

from src.config import settings
from src.models.base import BaseLLM, RAG_SYSTEM, build_rag_prompt
from src.rag.index import load_index


@dataclass
class RetrievalHit:
    article_id: str
    title: str
    text: str
    score: float | None = None


@dataclass
class RAGAnswer:
    answer: str
    context: str
    hits: list[RetrievalHit]
    model: str
    backend: str


class RAGPipeline:
    def __init__(
        self,
        llm: BaseLLM,
        vectorstore: FAISS | None = None,
        top_k: int | None = None,
    ) -> None:
        self.llm = llm
        self.vectorstore = vectorstore or load_index()
        self.top_k = settings.top_k if top_k is None else top_k
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def retrieve(self, question: str) -> list[RetrievalHit]:
        docs = self.retriever.invoke(question)
        hits: list[RetrievalHit] = []
        for doc in docs:
            hits.append(
                RetrievalHit(
                    article_id=str(doc.metadata.get("article_id", "")),
                    title=str(doc.metadata.get("title", "")),
                    text=doc.page_content,
                )
            )
        return hits

    @staticmethod
    def format_context(hits: list[RetrievalHit]) -> str:
        parts = []
        for i, hit in enumerate(hits, start=1):
            parts.append(
                f"[{i}] {hit.title} (id={hit.article_id})\n{hit.text}"
            )
        return "\n\n".join(parts)

    def answer(self, question: str) -> RAGAnswer:
        hits = self.retrieve(question)
        context = self.format_context(hits)
        prompt = build_rag_prompt(question, context)
        result = self.llm.generate(prompt, system=RAG_SYSTEM)
        return RAGAnswer(
            answer=result.text,
            context=context,
            hits=hits,
            model=result.model,
            backend=result.backend,
        )
