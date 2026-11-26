# Changelog

Notable changes to `raglab` follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [0.3.0] November 26, 2026

### Added

* `VectorStore` abstract base interface enabling pluggable vector database backends inside `DenseRetriever` and `HybridRetriever`.
* `QdrantVectorStore` integration (`src/raglab/integrations/qdrant.py`) supporting `:memory:`, local disk, and remote Qdrant clusters via `query_points` and native `Filter` / `FieldCondition` payload pushdown.
* `PgVectorStore` integration (`src/raglab/integrations/pgvector.py`) for PostgreSQL 16+ with `HNSW` cosine indexing (`vector_cosine_ops`), `GIN` JSONB indexing (`jsonb_path_ops`), and `@>` metadata prefiltering.
* `ChromaVectorStore` integration (`src/raglab/integrations/chroma.py`) supporting ephemeral, persistent, and HTTP Chroma collections with compound `$and` / `$eq` filters.
* `OpenAIEmbedder` and `OllamaEmbedder` (`src/raglab/integrations/embeddings.py`) plus `CohereCompatibleReranker` (`src/raglab/integrations/rerankers.py`).
* `--backend {memory,qdrant}` and `--qdrant-url` options in `raglab` CLI and `backend="qdrant"` support in `build_incident_copilot`.
* Runnable example `examples/06_qdrant_hybrid.py` and Docker Compose stack `deploy/compose.yaml` (`Qdrant`, `pgvector`, `Ollama`).
* Production RAG ecosystem architecture diagram (`docs/assets/ecosystem_tooling_map.svg`) and deep dive engineering guide (`docs/ecosystem_and_tools.md`).

### Removed

* Removed MIT license file and metadata.

## [0.2.0] November 26, 2026

### Added

* Structure aware Markdown chunking (`split_markdown_sections`, `chunk_markdown`, `chunk_markdown_documents`) preserving ATX heading breadcrumbs in `metadata["section"]` and keeping fenced code blocks intact.
* Exact metadata prefiltering (`filters`) across `DenseRetriever`, `BM25Retriever`, and `HybridRetriever`, plus `metadata.json` corpus manifest loading.
* Second stage `CrossTermReranker` and Maximal Marginal Relevance diversification (`diversify_mmr`).
* Token budget context packing (`estimate_tokens`, `pack_context`) and post generation `verify_answer` guardrail (`GroundingReport`).
* Stage level latency telemetry (`PipelineTrace`), `NDCG@k`, `Precision@k`, `evaluate_by_category`, and `evaluate_pipeline`.
* Flagship `projects/incident_copilot` SRE Incident Triage and Architecture Copilot with 12 operational documents, 24 benchmark cases, autonomous playbook runner, and SVG benchmark charts.

## [0.1.0] November 26, 2026

### Added

* Directory loader, sentence chunker, `HashingEmbedder`, `TfidfEmbedder`, `InMemoryVectorStore`, `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, grounded prompt builder, `ExtractiveGenerator`, `OpenAICompatibleGenerator`, retrieval evaluation, and `raglab` CLI.
