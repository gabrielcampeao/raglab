# Changelog

Notable changes to `raglab` follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [0.4.0] November 27, 2026

### Added

* `ContextSanitizer` (`src/raglab/security.py`) stripping invisible Unicode smuggling characters and blocking or redacting indirect prompt injection payloads before prompt assembly.
* Filter isolated `SemanticCache` (`src/raglab/cache.py`) with cosine threshold matching, TTL expiration, bounded capacity, and `PipelineTrace.cache_hit` tracking.
* `AcronymQueryExpander` and `MultiQueryRetriever` (`src/raglab/query.py`) fusing expanded operational queries via Reciprocal Rank Fusion.
* Structured telemetry sinks (`TraceEvent`, `InMemoryTraceSink`, `JsonlTraceSink`) in `src/raglab/telemetry.py`.
* Zero dependency HTTP API and browser `raglab Inspector` (`src/raglab/server.py` and `raglab serve`) serving `/healthz`, `/v1/search`, `/v1/ask`, and `/` UI.
* PEP 561 `py.typed` marker, `mypy --strict` static type checking, and dedicated `typecheck` and `security` GitHub Actions jobs.
* Repository governance files (`SECURITY.md`, `CODE_OF_CONDUCT.md`, GitHub issue templates, Dependabot configuration) and centered hero header with geometric SVG brand mark.

## [0.3.0] November 26, 2026

### Added

* `VectorStore` interface plus `QdrantVectorStore`, `PgVectorStore`, `ChromaVectorStore`, `OpenAIEmbedder`, `OllamaEmbedder`, and `CohereCompatibleReranker` in `src/raglab/integrations`.
* `--backend {memory,qdrant}` CLI support, `examples/06_qdrant_hybrid.py`, `deploy/compose.yaml`, `docs/ecosystem_and_tools.md`, and `docs/assets/ecosystem_tooling_map.svg`.

### Removed

* Removed MIT license file and metadata.

## [0.2.0] November 26, 2026

### Added

* Structure aware Markdown chunking, metadata prefiltering, `CrossTermReranker`, `diversify_mmr`, token budget packing, `verify_answer` grounding guardrail, `NDCG@k`, `projects/incident_copilot`, and deterministic SVG benchmark charts.

## [0.1.0] November 26, 2026

### Added

* Core document loader, sentence chunker, `HashingEmbedder`, `TfidfEmbedder`, `InMemoryVectorStore`, `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, `ExtractiveGenerator`, `OpenAICompatibleGenerator`, evaluation metrics, and `raglab` CLI.
