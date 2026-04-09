from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

import chromadb

try:
    from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
except Exception:  # pragma: no cover - chromadb optional internals
    OllamaEmbeddingFunction = None  # type: ignore[assignment]


def _stable_float_vector(text: str, *, dimension: int = 64) -> list[float]:
    digest = sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dimension:
        for byte in digest:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) >= dimension:
                break
        digest = sha256(digest).digest()
    return values


class DeterministicEmbeddingFunction:
    def name(self) -> str:  # pragma: no cover - simple interface hook
        return "deterministic-hash-embedding"

    def default_space(self) -> str:  # pragma: no cover - interface hook
        return "cosine"

    def supported_spaces(self) -> list[str]:  # pragma: no cover - interface hook
        return ["cosine"]

    def get_config(self) -> dict[str, object]:  # pragma: no cover - interface hook
        return {"name": self.name(), "default_space": self.default_space(), "supported_spaces": self.supported_spaces()}

    def is_legacy(self) -> bool:  # pragma: no cover - interface hook
        return False

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [_stable_float_vector(text) for text in input]

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(input, list):
            return [_stable_float_vector(text) for text in input]
        return _stable_float_vector(input)

    def __call__(self, input: list[str] | str) -> list[list[float]]:  # noqa: A002
        texts = input if isinstance(input, list) else [input]
        return self.embed_documents(texts)


class SafeOllamaEmbeddingFunction:
    def __init__(self, *, url: str, model_name: str, timeout: int = 10) -> None:
        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self._backend = None
        self._deterministic = DeterministicEmbeddingFunction()
        if OllamaEmbeddingFunction is not None:
            try:
                backend = OllamaEmbeddingFunction(url=url, model_name=model_name, timeout=timeout)
                backend(["embedding health check"])
            except Exception:
                backend = None
            self._backend = backend

    def name(self) -> str:  # pragma: no cover - simple interface hook
        if self._backend is not None:
            backend_name = getattr(self._backend, "name", None)
            if callable(backend_name):
                return str(backend_name())
        return self._deterministic.name()

    def default_space(self) -> str:  # pragma: no cover - interface hook
        backend_method = getattr(self._backend, "default_space", None)
        if callable(backend_method):
            try:
                return str(backend_method())
            except Exception:
                return self._deterministic.default_space()
        return self._deterministic.default_space()

    def supported_spaces(self) -> list[str]:  # pragma: no cover - interface hook
        backend_method = getattr(self._backend, "supported_spaces", None)
        if callable(backend_method):
            try:
                result = backend_method()
                return list(result) if isinstance(result, list) else ["cosine"]
            except Exception:
                return self._deterministic.supported_spaces()
        return self._deterministic.supported_spaces()

    def get_config(self) -> dict[str, object]:  # pragma: no cover - interface hook
        return {
            "name": self.name(),
            "default_space": self.default_space(),
            "supported_spaces": self.supported_spaces(),
            "url": self.url,
            "model_name": self.model_name,
            "timeout": self.timeout,
        }

    def is_legacy(self) -> bool:  # pragma: no cover - interface hook
        return False

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        if self._backend is not None:
            backend_method = getattr(self._backend, "embed_documents", None)
            if callable(backend_method):
                return backend_method(input)  # type: ignore[no-any-return]
            return self._backend(input)  # type: ignore[no-any-return]
        return self._deterministic.embed_documents(input)

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        if self._backend is not None:
            backend_method = getattr(self._backend, "embed_query", None)
            if callable(backend_method):
                try:
                    return backend_method(input)  # type: ignore[no-any-return]
                except Exception:
                    return self._deterministic.embed_query(input)
            if isinstance(input, list):
                return self._backend(input)  # type: ignore[no-any-return]
            result = self._backend([input])  # type: ignore[no-any-return]
            return result[0] if result else self._deterministic.embed_query(input)
        return self._deterministic.embed_query(input)

    def __call__(self, input: list[str] | str) -> list[list[float]]:  # noqa: A002
        if self._backend is not None:
            try:
                if isinstance(input, list):
                    return self.embed_documents(input)
                return [self.embed_query(input)]
            except Exception:
                return self._deterministic(input)
        return self._deterministic(input)


@dataclass(frozen=True)
class MemoryFact:
    calendar_id: str
    category: str
    normalized_value: str
    confidence: float
    summary: str
    source_session_ids: list[str]
    source_run_ids: list[str]
    created_at: str
    updated_at: str

    @property
    def document_id(self) -> str:
        key = f"{self.calendar_id}::{self.category}::{self.normalized_value}"
        return sha256(key.encode("utf-8")).hexdigest()[:24]


class LongTermMemoryStore:
    def __init__(self, data_root: str | Path, *, ollama_url: str = "http://127.0.0.1:11434") -> None:
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.data_root)
        self.embedding_function = SafeOllamaEmbeddingFunction(url=ollama_url, model_name="embeddinggemma", timeout=10)
        self.collection = self.client.get_or_create_collection(name="long_term_memory")

    def query(self, *, calendar_id: str, query_text: str, recent_context: str = "", limit: int = 5) -> dict[str, Any]:
        text = "\n".join(part for part in [query_text.strip(), recent_context.strip()] if part).strip()
        if not text:
            return {"facts": [], "summary_text": ""}

        try:
            result = self.collection.query(
                query_embeddings=[self.embedding_function.embed_query(text)],
                n_results=max(1, limit),
                where={"calendar_id": calendar_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return {"facts": [], "summary_text": ""}

        facts: list[dict[str, Any]] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            facts.append(
                {
                    "category": metadata.get("fact_category", ""),
                    "value": metadata.get("normalized_fact_value", document),
                    "confidence": metadata.get("confidence", 0.0),
                    "summary": document,
                    "distance": distances[index] if index < len(distances) else None,
                    "source_session_ids": metadata.get("source_session_ids", ""),
                }
            )

        summary_parts = [
            f"{fact['category']}={fact['value']} ({fact['confidence']})"
            for fact in facts
            if fact.get("category") and fact.get("value")
        ]
        return {"facts": facts, "summary_text": "; ".join(summary_parts)}

    def upsert_facts(self, facts: list[MemoryFact]) -> list[str]:
        if not facts:
            return []

        ids = [fact.document_id for fact in facts]
        documents = [fact.summary for fact in facts]
        metadatas = [
            {
                "calendar_id": fact.calendar_id,
                "fact_category": fact.category,
                "normalized_fact_value": fact.normalized_value,
                "confidence": float(fact.confidence),
                "source_session_ids": ",".join(fact.source_session_ids),
                "source_run_ids": ",".join(fact.source_run_ids),
                "created_at": fact.created_at,
                "updated_at": fact.updated_at,
            }
            for fact in facts
        ]
        embeddings = self.embedding_function.embed_documents(documents)
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return ids


def sanitize_query_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:500]
