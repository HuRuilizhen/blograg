"""Shared test helpers for blograg."""

from __future__ import annotations

from collections.abc import Sequence


class FakeEmbeddingProvider:
    """Deterministic embedding provider for offline tests."""

    provider_name = "fake"
    model_name = "fake-1"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        keywords = ("front", "matter", "paragraph", "retrieval", "metadata")
        keyword_counts = [float(normalized.count(keyword)) for keyword in keywords]
        return [*keyword_counts, float(len(normalized))]
