"""Build and load FAISS index over corporate articles."""

from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.data_loader import Article, load_articles


def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    name = model_name or settings.embedding_model
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def articles_to_documents(articles: list[Article]) -> list[Document]:
    docs: list[Document] = []
    for art in articles:
        docs.append(
            Document(
                page_content=art.content,
                metadata={"article_id": art.id, "title": art.title},
            )
        )
    return docs


def split_documents(
    documents: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(
    index_dir: Path | None = None,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> FAISS:
    articles = load_articles()
    documents = articles_to_documents(articles)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)

    out = index_dir or settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))
    return store


def load_index(index_dir: Path | None = None) -> FAISS:
    path = index_dir or settings.index_dir
    if not path.exists():
        raise FileNotFoundError(
            f"Index not found at {path}. Run: python scripts/build_index.py"
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
