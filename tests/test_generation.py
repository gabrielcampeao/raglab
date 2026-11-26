import urllib.error
from typing import Any

import pytest

from raglab.generation import (
    NO_ANSWER,
    ExtractiveGenerator,
    GenerationError,
    OpenAICompatibleGenerator,
)
from raglab.models import Chunk, SearchResult


def _result(source: str, text: str) -> SearchResult:
    chunk = Chunk(
        id=f"{source}#0", text=text, document_id=source, position=0, metadata={"source": source}
    )
    return SearchResult(chunk=chunk, score=1.0)


def test_extractive_generator_picks_best_sentence_with_citation() -> None:
    results = [
        _result("pricing.md", "The Plus plan costs 6 dollars. It syncs fast."),
        _result("refunds.md", "Refunds are accepted within 30 days of purchase. Contact billing."),
    ]

    answer = ExtractiveGenerator(max_sentences=1).generate(
        "Within how many days are refunds accepted?", "", results
    )

    assert answer == "Refunds are accepted within 30 days of purchase. [2]"


def test_extractive_generator_says_when_nothing_matches() -> None:
    results = [_result("pricing.md", "The Plus plan costs 6 dollars.")]

    assert ExtractiveGenerator().generate("Where are servers located?", "", results) == NO_ANSWER


def test_extractive_generator_skips_headings_and_duplicate_sentences() -> None:
    results = [
        _result("a.md", "# Refund policy\nRefunds take 5 days."),
        _result("b.md", "Refunds take 5 days."),
    ]

    answer = ExtractiveGenerator(max_sentences=2).generate("refund policy days", "", results)

    assert answer == "Refunds take 5 days. [1]"


def test_extractive_generator_rejects_invalid_max_sentences() -> None:
    with pytest.raises(ValueError):
        ExtractiveGenerator(max_sentences=0)


class RecordingTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        self.calls.append((url, headers, payload, timeout))
        return self.response


def test_openai_generator_sends_prompt_and_parses_reply() -> None:
    transport = RecordingTransport({"choices": [{"message": {"content": "  Thirty days. [1] "}}]})
    generator = OpenAICompatibleGenerator(
        "test-model",
        base_url="http://localhost:11434/v1/",
        api_key="secret",
        timeout=5.0,
        transport=transport,
    )

    assert generator.generate("How long?", "PROMPT", []) == "Thirty days. [1]"

    url, headers, payload, timeout = transport.calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.0
    assert payload["messages"][-1] == {"role": "user", "content": "PROMPT"}
    assert timeout == 5.0


def test_openai_generator_reads_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.test/v1")

    generator = OpenAICompatibleGenerator("model")

    assert generator.api_key == "from-env"
    assert generator.base_url == "http://example.test/v1"


def test_openai_generator_omits_authorization_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    transport = RecordingTransport({"choices": [{"message": {"content": "ok"}}]})

    OpenAICompatibleGenerator("model", transport=transport).generate("q", "p", [])

    assert "Authorization" not in transport.calls[0][1]


def test_openai_generator_wraps_network_errors() -> None:
    def failing_transport(*_: Any) -> dict[str, Any]:
        raise urllib.error.URLError("connection refused")

    generator = OpenAICompatibleGenerator("model", api_key="", transport=failing_transport)

    with pytest.raises(GenerationError, match="connection refused"):
        generator.generate("q", "p", [])


@pytest.mark.parametrize("response", [{}, {"choices": []}, {"choices": [{"message": None}]}])
def test_openai_generator_rejects_unexpected_responses(response: dict[str, Any]) -> None:
    generator = OpenAICompatibleGenerator(
        "model", api_key="", transport=RecordingTransport(response)
    )

    with pytest.raises(GenerationError):
        generator.generate("q", "p", [])


def test_extractive_generator_abstains_on_weak_single_term_coincidence() -> None:
    results = [
        _result("slo.md", "Service timeout budget is 200 milliseconds at p99."),
    ]
    generator = ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34)

    answer = generator.generate(
        "What is our GraphQL federation gateway timeout?",
        "",
        results,
    )

    assert answer == NO_ANSWER


def test_extractive_generator_clamps_min_shared_terms_for_single_token_queries() -> None:
    results = [
        _result("pg.md", "PgBouncer pools connections in transaction mode."),
    ]
    generator = ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34)

    answer = generator.generate("What is PgBouncer?", "", results)

    assert answer == "PgBouncer pools connections in transaction mode. [1]"
