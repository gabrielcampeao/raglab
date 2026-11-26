from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from raglab.chunking import split_sentences
from raglab.models import SearchResult
from raglab.text import tokenize

NO_ANSWER = "I could not find an answer to that in the indexed documents."

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Ground every statement in the provided context "
    "and cite sources by number."
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"

Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class GenerationError(RuntimeError):
    pass


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, prompt: str, results: Sequence[SearchResult]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(Generator):
    def __init__(
        self,
        max_sentences: int = 2,
        *,
        min_shared_terms: int = 1,
        min_overlap_ratio: float = 0.0,
    ) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if min_shared_terms <= 0:
            raise ValueError("min_shared_terms must be positive")
        if not 0.0 <= min_overlap_ratio <= 1.0:
            raise ValueError("min_overlap_ratio must be between 0 and 1")
        self.max_sentences = max_sentences
        self.min_shared_terms = min_shared_terms
        self.min_overlap_ratio = min_overlap_ratio

    def generate(self, question: str, prompt: str, results: Sequence[SearchResult]) -> str:
        query_terms = set(tokenize(question))
        if not query_terms:
            return NO_ANSWER
        required_terms = min(self.min_shared_terms, len(query_terms))
        candidates: list[tuple[int, int, int, str]] = []
        seen: set[str] = set()
        for source_number, result in enumerate(results, start=1):
            section_terms = set(tokenize(result.chunk.metadata.get("section", "")))
            for position, sentence in enumerate(split_sentences(result.chunk.text)):
                if sentence.startswith(("#", "```")) or sentence in seen:
                    continue
                seen.add(sentence)
                sentence_terms = set(tokenize(sentence))
                direct_overlap = len(query_terms & sentence_terms)
                if direct_overlap == 0:
                    continue
                effective_overlap = len(query_terms & (sentence_terms | section_terms))
                coverage = effective_overlap / len(query_terms)
                if effective_overlap >= required_terms and coverage >= self.min_overlap_ratio:
                    candidates.append((direct_overlap, source_number, position, sentence))
        if not candidates:
            return NO_ANSWER
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
        selected = candidates[: self.max_sentences]
        return " ".join(f"{sentence} [{number}]" for _, number, _, sentence in selected)


def urllib_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleGenerator(Generator):
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        self.system_prompt = system_prompt
        self._transport = transport or urllib_transport

    def build_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

    def generate(self, question: str, prompt: str, results: Sequence[SearchResult]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        try:
            data = self._transport(url, headers, self.build_payload(prompt), self.timeout)
        except (urllib.error.URLError, TimeoutError) as error:
            raise GenerationError(f"Request to {url} failed: {error}") from error
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise GenerationError(
                "The language model API returned an unexpected response"
            ) from error
