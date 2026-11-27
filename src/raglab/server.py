from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from raglab._version import __version__
from raglab.generation import GenerationError
from raglab.pipeline import RAGPipeline

MAX_BODY_BYTES = 262_144

INSPECTOR_HTML = "\n".join(
    [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>raglab Inspector</title>",
        "<style>",
        "  body { margin: 0; background: #0d1117; color: #e6edf3;",
        "    font-family: Segoe UI, Inter, system-ui, sans-serif; }",
        "  main { max-width: 920px; margin: 40px auto; padding: 0 20px; }",
        "  h1 { font-size: 26px; margin: 0 0 6px; }",
        "  p.sub { color: #8b949e; margin: 0 0 24px; font-size: 14px; }",
        "  .panel { background: #161b22; border: 1px solid #30363d;",
        "    border-radius: 12px; padding: 20px; margin-bottom: 18px; }",
        "  label { display: block; font-size: 12px; font-weight: 600;",
        "    color: #8b949e; margin-bottom: 6px; text-transform: uppercase; }",
        "  input { width: 100%; box-sizing: border-box; background: #0d1117;",
        "    color: #e6edf3; border: 1px solid #30363d; border-radius: 8px;",
        "    padding: 10px 12px; font-size: 14px; margin-bottom: 14px; }",
        "  button { background: #238636; color: #ffffff; border: 0;",
        "    border-radius: 8px; padding: 10px 18px; font-size: 14px;",
        "    font-weight: 600; cursor: pointer; }",
        "  button:hover { background: #2ea043; }",
        "  .meta { display: flex; gap: 14px; flex-wrap: wrap;",
        "    font-size: 12.5px; color: #8b949e; margin-top: 12px; }",
        "  .badge { padding: 2px 8px; border-radius: 999px;",
        "    font-weight: 600; font-size: 12px; }",
        "  .ok { background: rgba(46, 160, 67, 0.2); color: #3fb950; }",
        "  .warn { background: rgba(210, 153, 34, 0.2); color: #d29922; }",
        "  .src { border-top: 1px solid #21262d; padding-top: 10px;",
        "    margin-top: 10px; font-size: 13px; }",
        "  .code { font-family: ui-monospace, Consolas, monospace; color: #58a6ff; }",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "  <h1>raglab Inspector</h1>",
        '  <p class="sub">Interactive retrieval, reranking, caching, and verification console.</p>',
        '  <div class="panel">',
        '    <label for="q">Question</label>',
        '    <input id="q" value="Workers emit SQLSTATE 53300. What is the immediate mitigation?">',
        '    <label for="f">Metadata filters (optional comma separated key=value)</label>',
        '    <input id="f" placeholder="kind=runbook, service=postgres">',
        '    <button id="run" type="button">Ask and verify</button>',
        "  </div>",
        '  <div id="out" class="panel" style="display:none"></div>',
        "</main>",
        "<script>",
        'document.getElementById("run").addEventListener("click", async () => {',
        '  const question = document.getElementById("q").value;',
        '  const rawFilters = document.getElementById("f").value.trim();',
        "  const filters = {};",
        "  if (rawFilters) {",
        '    for (const pair of rawFilters.split(",")) {',
        '      const idx = pair.indexOf("=");',
        "      if (idx > 0) filters[pair.slice(0, idx).trim()] = pair.slice(idx + 1).trim();",
        "    }",
        "  }",
        "  const payload = {question, filters: Object.keys(filters).length ? filters : null};",
        '  const res = await fetch("/v1/ask", {',
        '    method: "POST",',
        '    headers: {"Content-Type": "application/json"},',
        "    body: JSON.stringify(payload)",
        "  });",
        "  const data = await res.json();",
        '  const out = document.getElementById("out");',
        '  out.style.display = "block";',
        '  if (!res.ok) { out.textContent = data.error || "Request failed"; return; }',
        '  const cls = data.grounding.verified ? "ok" : "warn";',
        '  const lbl = data.grounding.abstained ? "abstained"',
        '    : (data.grounding.verified ? "verified" : "unverified");',
        '  let html = `<div><span class="badge ${cls}">${lbl}</span></div>`;',
        '  html += `<p style="line-height:1.55">${data.answer}</p>`;',
        '  html += `<div class="meta"><span>latency: ${data.trace.total_ms.toFixed(2)} ms</span>`',
        "    + `<span>tokens: ${data.trace.context_tokens}</span>`",
        "    + `<span>cache hit: ${data.trace.cache_hit}</span></div>`;",
        "  for (const [i, s] of data.sources.entries()) {",
        '    const sec = s.section ? ` (${s.section})` : "";',
        '    html += `<div class="src"><div class="code">[${i + 1}] ${s.source}${sec}`',
        "      + ` | score ${s.score.toFixed(4)}</div>`",
        '      + `<div style="color:#8b949e;margin-top:4px">${s.text}</div></div>`;',
        "  }",
        "  out.innerHTML = html;",
        "});",
        "</script>",
        "</body>",
        "</html>",
    ]
)


def _parse_filters(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("'filters' must be a JSON object")
    return {str(key): str(val) for key, val in raw.items()}


def create_server(
    pipeline: RAGPipeline,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return None

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("Request body must be a non empty JSON object")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            return body

        def do_GET(self) -> None:
            if self.path == "/healthz":
                with lock:
                    count = len(pipeline.chunks)
                self._send_json(
                    HTTPStatus.OK,
                    {"status": "ok", "version": __version__, "chunks": count},
                )
                return
            if self.path == "/":
                raw = INSPECTOR_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                filters = _parse_filters(body.get("filters"))
                raw_k = body.get("k")
                k_val = int(raw_k) if raw_k is not None else None

                if self.path == "/v1/search":
                    query = str(body.get("query", ""))
                    with lock:
                        results = pipeline.retrieve(query, k=k_val, filters=filters)
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "results": [
                                {
                                    "chunk_id": item.chunk.id,
                                    "source": item.chunk.source,
                                    "section": item.chunk.metadata.get("section", ""),
                                    "score": round(item.score, 6),
                                    "text": item.chunk.text,
                                }
                                for item in results
                            ]
                        },
                    )
                    return

                if self.path == "/v1/ask":
                    question = str(body.get("question", ""))
                    with lock:
                        answer = pipeline.ask(question, k=k_val, filters=filters)
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "question": answer.question,
                            "answer": answer.text,
                            "grounding": {
                                **asdict(answer.grounding),
                                "verified": answer.grounding.is_verified,
                            },
                            "trace": asdict(answer.trace),
                            "sources": [
                                {
                                    "chunk_id": item.chunk.id,
                                    "source": item.chunk.source,
                                    "section": item.chunk.metadata.get("section", ""),
                                    "score": round(item.score, 6),
                                    "text": item.chunk.text,
                                }
                                for item in answer.sources
                            ],
                        },
                    )
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except GenerationError as error:
                self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})

    return ThreadingHTTPServer((host, port), _Handler)
