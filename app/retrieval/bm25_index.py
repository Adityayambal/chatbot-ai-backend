from __future__ import annotations

from rank_bm25 import BM25Okapi

from app.content.chunker import Chunk
from app.retrieval.tokenize import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(
        self, query_tokens: list[str], top_n: int, restrict_to: set[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        if self._bm25 is None or not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        results: list[tuple[Chunk, float]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            if restrict_to is not None and chunk.chunk_id not in restrict_to:
                continue
            results.append((chunk, float(score)))
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:top_n]
