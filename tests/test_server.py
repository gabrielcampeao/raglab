from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

import pytest

from raglab import CrossTermReranker, Document, ExtractiveGenerator, RAGPipeline
from raglab.cache import SemanticCache
from raglab.security import ContextSanitizer
from raglab.server import create_server


def _json_request(url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    method = "POST" if payload is not None else "GET"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_http_server_serves_health_inspector_search_and_verified_ask() -> None:
    pipeline = RAGPipeline(
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34),
        reranker=CrossTermReranker(),
        sanitizer=ContextSanitizer(),
        cache=SemanticCache(),
        markdown_aware=True,
    )
    pipeline.ingest(
        [
            Document(
                id="pg.md",
                text=(
                    "# Postgres saturation\n"
                    "## Mitigation\n"
                    "Switch PgBouncer pool_mode to transaction during SQLSTATE 53300."
                ),
                metadata={"source": "pg.md", "kind": "runbook"},
            )
        ]
    )

    server = create_server(pipeline, host="127.0.0.1", port=0)
    host, port = server.server_address[0], server.server_address[1]
    base = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = _json_request(f"{base}/healthz")
        assert status == 200
        assert health["status"] == "ok"
        assert health["chunks"] == 1

        with urllib.request.urlopen(f"{base}/", timeout=5) as ui_resp:
            html = ui_resp.read().decode("utf-8")
            assert ui_resp.status == 200
            assert "raglab Inspector" in html
            assert "retrieval_ms" in html
            assert "rerank_ms" in html
            assert "generation_ms" in html
            assert "&quot;" in html
            assert "&#39;" in html

        status, search_body = _json_request(
            f"{base}/v1/search",
            {"query": "SQLSTATE 53300", "k": 1, "filters": {"kind": "runbook"}},
        )
        assert status == 200
        assert search_body["results"][0]["source"] == "pg.md"

        status, ask_body = _json_request(
            f"{base}/v1/ask",
            {
                "question": "How do we mitigate SQLSTATE 53300 PgBouncer saturation?",
                "filters": {"kind": "runbook"},
            },
        )
        assert status == 200
        assert "transaction" in ask_body["answer"]
        assert ask_body["grounding"]["verified"] is True
        assert ask_body["trace"]["cache_hit"] is False

        _, cached_body = _json_request(
            f"{base}/v1/ask",
            {
                "question": "How do we mitigate SQLSTATE 53300 PgBouncer saturation?",
                "filters": {"kind": "runbook"},
            },
        )
        assert cached_body["trace"]["cache_hit"] is True

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _json_request(f"{base}/v1/ask", {"question": "   "})
        assert exc_info.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _json_request(f"{base}/v1/ask", {"question": None})
        assert exc_info.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _json_request(f"{base}/v1/search", {"query": None})
        assert exc_info.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _json_request(f"{base}/v1/search", {"query": "SQLSTATE", "k": True})
        assert exc_info.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _json_request(f"{base}/v1/search", {"query": "SQLSTATE", "filters": {"kind": None}})
        assert exc_info.value.code == 400

        unknown_req = urllib.request.Request(f"{base}/v1/unknown", data=b"", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(unknown_req, timeout=5)
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_server_returns_json_error_for_empty_pipeline() -> None:
    server = create_server(RAGPipeline(), host="127.0.0.1", port=0)
    host, port = server.server_address[0], server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _json_request(f"http://{host}:{port}/v1/ask", {"question": "Hello?"})
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
