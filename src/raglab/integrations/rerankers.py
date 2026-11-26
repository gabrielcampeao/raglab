from __future__ import annotations

import os
import urllib.error
from collections.abc import Sequence
from typing import Any

from raglab.generation import GenerationError, Transport, urllib_transport
from raglab.models import SearchResult
from raglab.reranking import Reranker

DEFAULT_RERANK_URL = "https://api.cohere.com/v2"


class CohereCompatibleReranker(Reranker):
    def __init__(
        self,
        model: str = "rerank-v3.5",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("RERANK_BASE_URL") or DEFAULT_RERANK_URL
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("COHERE_API_KEY")
        self.timeout = timeout
        self._transport = transport or urllib_transport

    def rerank(
        self, query: str, results: Sequence[SearchResult], k: int | None = None
    ) -> list[SearchResult]:
        if k is not None and k <= 0:
            raise ValueError("k must be positive")
        if not results:
            return []
        limit = k if k is not None else len(results)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [item.chunk.text for item in results],
            "top_n": min(limit, len(results)),
        }
        url = f"{self.base_url}/rerank"
        try:
            response = self._transport(url, headers, payload, self.timeout)
        except (urllib.error.URLError, TimeoutError) as error:
            raise GenerationError(f"Rerank request to {url} failed: {error}") from error
        try:
            scored = [
                SearchResult(
                    chunk=results[int(entry["index"])].chunk,
                    score=float(entry["relevance_score"]),
                )
                for entry in response["results"]
            ]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise GenerationError("Rerank API returned an unexpected payload") from error
        scored.sort(key=lambda item: (-item.score, item.chunk.id))
        return scored[:limit]
