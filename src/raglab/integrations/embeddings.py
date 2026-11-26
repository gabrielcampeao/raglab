from __future__ import annotations

import os
import urllib.error
from collections.abc import Sequence
from typing import Any

from raglab.embeddings import Embedder
from raglab.generation import DEFAULT_BASE_URL, GenerationError, Transport, urllib_transport
from raglab.vectormath import Vector

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.dimensions = dimensions
        self.timeout = timeout
        self._transport = transport or urllib_transport

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        url = f"{self.base_url}/embeddings"
        try:
            response = self._transport(url, headers, payload, self.timeout)
        except (urllib.error.URLError, TimeoutError) as error:
            raise GenerationError(f"Embedding request to {url} failed: {error}") from error
        try:
            ordered = sorted(response["data"], key=lambda item: int(item["index"]))
            return [[float(val) for val in item["embedding"]] for item in ordered]
        except (KeyError, TypeError, ValueError) as error:
            raise GenerationError("Embedding API returned an unexpected payload") from error


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).rstrip(
            "/"
        )
        self.timeout = timeout
        self._transport = transport or urllib_transport

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        url = f"{self.base_url}/api/embed"
        try:
            response = self._transport(url, headers, payload, self.timeout)
        except (urllib.error.URLError, TimeoutError) as error:
            raise GenerationError(f"Ollama embedding request to {url} failed: {error}") from error
        try:
            return [[float(val) for val in vec] for vec in response["embeddings"]]
        except (KeyError, TypeError, ValueError) as error:
            raise GenerationError("Ollama embedding API returned an unexpected payload") from error
