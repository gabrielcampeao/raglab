# Changelog

Notable changes to `raglab` follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [1.0.0] November 27, 2026

### Added

* Pure Python `HNSWVectorStore` (`src/raglab/hnsw.py`) implementing layered Hierarchical Navigable Small World approximate nearest neighbor search, heuristic neighbor selection, and exact filtered fallback, exposed via `--backend hnsw` in `src/raglab/cli.py`.
* Multi hop `KnowledgeGraph`, `EntityGraphRetriever`, and `extract_entities` (`src/raglab/graph.py`) with decaying activation propagation and entity IDF weighting.
* `CorrectiveRAG`, `CorrectiveStep`, and `CorrectiveOutcome` (`src/raglab/corrective.py`) implementing confidence gated query rewriting, automatic metadata filter relaxation, and verified abstention.
* Runnable script `examples/08_hnsw_graph_corrective.py` and real scaling benchmark chart `docs/assets/hnsw_scaling_benchmark.svg` comparing exact cosine scan against `HNSWVectorStore`.
* Per stage millisecond breakdown (`retrieval_ms`, `rerank_ms`, `generation_ms`) and escaped DOM rendering inside `raglab Inspector` (`src/raglab/server.py`).

### Changed

* Upgraded GitHub Actions CI workflows in `.github/workflows/ci.yml` to `actions/checkout@v7` and `actions/setup-python@v7` for native Node 24 runner execution with zero deprecation annotations.

## [0.5.0] November 27, 2026

### Added

* `ParentChildRetriever` and `build_parent_child_chunks` (`src/raglab/parent_child.py`) decoupling fine grained child search (`180` chars) from full parent section context (`750` chars).
* `SubQuestionDecomposer` with round robin interleave fusion, `decompose_query`, and `HyDERetriever` in `src/raglab/query.py`.
* `reorder_lost_in_middle` in `src/raglab/prompting.py` and `reorder_context=True` on `RAGPipeline`.
* `ConversationalRAG` and `condense_follow_up` (`src/raglab/conversation.py`) resolving multi turn pronouns against prior turn section anchors.
* Runnable script `examples/07_advanced_patterns.py`, `docs/assets/advanced_rag_taxonomy.svg`, and two research study guides (`docs/advanced_rag_patterns.md` and `docs/evaluation_and_failure_taxonomy.md`).

## [0.4.0] November 27, 2026

### Added

* `ContextSanitizer` (`src/raglab/security.py`) stripping invisible Unicode smuggling characters and blocking or redacting indirect prompt injection payloads before prompt assembly.
* Filter isolated `SemanticCache` (`src/raglab/cache.py`) with cosine threshold matching, TTL expiration, bounded capacity, and `PipelineTrace.cache_hit` tracking.
* `AcronymQueryExpander` and `MultiQueryRetriever` (`src/raglab/query.py`) fusing expanded operational queries via Reciprocal Rank Fusion.
* Structured telemetry sinks (`TraceEvent`, `InMemoryTraceSink`, `JsonlTraceSink`) in `src/raglab/telemetry.py`.
* Zero dependency HTTP API and browser `raglab Inspector` (`src/raglab/server.py` and `raglab serve`) serving `/healthz`, `/v1/search`, `/v1/ask`, and `/` UI.
* PEP 561 `py.typed` marker, `mypy --strict` static type checking, and dedicated `typecheck` and `security` GitHub Actions jobs.
* Repository governance files (`SECURITY.md`, `CODE_OF_CONDUCT.md`, GitHub issue templates) and centered hero header with geometric SVG brand mark.

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
