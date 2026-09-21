from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RAGHit:
    source: str
    text: str
    score: float


@dataclass
class RAGResult:
    hits: list[RAGHit]
    status: str
    error: str | None = None

    @property
    def context_text(self) -> str:
        if not self.hits:
            return ""
        blocks = []
        for index, hit in enumerate(self.hits, start=1):
            blocks.append(f"[{index}] Источник: {hit.source}\n{hit.text}")
        return "\n\n".join(blocks)


class RAGAdapter(ABC):
    @abstractmethod
    async def ensure_indexed(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> RAGResult:
        raise NotImplementedError


def _iter_kb_files(kb_dir: Path) -> list[Path]:
    if not kb_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(kb_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            files.append(path)
    return files


def _chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(cleaned) <= size:
        return [cleaned] if cleaned else []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        chunks.append(cleaned[start:end].strip())
        start += size - overlap
    return [chunk for chunk in chunks if chunk]


class InMemoryHybridRAGAdapter(RAGAdapter):
    """Keyword + overlap hybrid search used when Qdrant/embeddings are unavailable."""

    def __init__(self, kb_dir: Path) -> None:
        self.kb_dir = kb_dir
        self._chunks: list[tuple[str, str]] = []

    async def ensure_indexed(self) -> None:
        self._chunks = []
        for path in _iter_kb_files(self.kb_dir):
            text = path.read_text(encoding="utf-8")
            for chunk in _chunk_text(text):
                self._chunks.append((path.name, chunk))
        logger.info(
            "rag_indexed",
            extra={"event": "rag_indexed", "provider": "stub", "status": f"chunks={len(self._chunks)}"},
        )

    async def retrieve(self, query: str, top_k: int) -> RAGResult:
        if not self._chunks:
            await self.ensure_indexed()
        tokens = {token for token in re.findall(r"[а-яёa-z0-9]{3,}", query.lower())}
        scored: list[RAGHit] = []
        for source, chunk in self._chunks:
            hay = chunk.lower()
            overlap = sum(1 for token in tokens if token in hay)
            if overlap:
                scored.append(RAGHit(source=source, text=chunk, score=float(overlap)))
        scored.sort(key=lambda item: item.score, reverse=True)
        hits = scored[:top_k]
        status = "ok" if hits else "empty"
        return RAGResult(hits=hits, status=status)


class LlamaIndexQdrantRAGAdapter(RAGAdapter):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._index = None
        self._fallback = InMemoryHybridRAGAdapter(settings.knowledge_base_dir)

    async def ensure_indexed(self) -> None:
        try:
            self._index = await _build_llamaindex(self.settings)
            logger.info("rag_indexed", extra={"event": "rag_indexed", "provider": "llamaindex", "success": True})
        except Exception as exc:
            logger.warning(
                "rag_index_fallback",
                extra={"event": "rag_index_fallback", "provider": "stub", "error": str(exc), "success": False},
            )
            await self._fallback.ensure_indexed()
            self._index = None

    async def retrieve(self, query: str, top_k: int) -> RAGResult:
        if self._index is None:
            result = await self._fallback.retrieve(query, top_k)
            result.status = "fallback"
            return result
        try:
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            hits: list[RAGHit] = []
            for node_with_score in nodes:
                node = node_with_score.node
                source = "unknown"
                if node.metadata:
                    source = str(node.metadata.get("file_name") or node.metadata.get("source") or "unknown")
                hits.append(
                    RAGHit(
                        source=source,
                        text=node.get_content(),
                        score=float(node_with_score.score or 0.0),
                    )
                )
            keyword = await self._fallback.retrieve(query, top_k)
            merged = _rrf_merge(hits, keyword.hits, top_k)
            return RAGResult(hits=merged, status="ok")
        except Exception as exc:
            logger.warning(
                "rag_retrieve_fallback",
                extra={"event": "rag_retrieve_fallback", "error": str(exc), "success": False},
            )
            result = await self._fallback.retrieve(query, top_k)
            result.status = "fallback"
            result.error = str(exc)
            return result


def _rrf_merge(dense: list[RAGHit], sparse: list[RAGHit], top_k: int, k: int = 60) -> list[RAGHit]:
    scores: dict[tuple[str, str], float] = {}
    payload: dict[tuple[str, str], RAGHit] = {}
    for rank, hit in enumerate(dense):
        key = (hit.source, hit.text[:160])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        payload[key] = hit
    for rank, hit in enumerate(sparse):
        key = (hit.source, hit.text[:160])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        payload.setdefault(key, hit)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    merged: list[RAGHit] = []
    for key, score in ordered:
        hit = payload[key]
        merged.append(RAGHit(source=hit.source, text=hit.text, score=score))
    return merged


def _build_llamaindex(settings: Settings):
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    kb_dir = settings.knowledge_base_dir
    if not kb_dir.exists():
        raise FileNotFoundError(f"Каталог базы знаний не найден: {kb_dir}")

    client = QdrantClient(url=settings.qdrant_url, timeout=10)
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    if client.collection_exists(settings.qdrant_collection):
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = SimpleDirectoryReader(str(kb_dir), required_exts=[".txt", ".md"]).load_data()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=False,
    )


def build_rag_adapter(settings: Settings) -> RAGAdapter:
    if settings.rag_provider == "stub":
        adapter = InMemoryHybridRAGAdapter(settings.knowledge_base_dir)
        logger.info("rag_adapter_selected", extra={"event": "rag_adapter_selected", "provider": "stub"})
        return adapter
    logger.info("rag_adapter_selected", extra={"event": "rag_adapter_selected", "provider": "llamaindex"})
    return LlamaIndexQdrantRAGAdapter(settings)
