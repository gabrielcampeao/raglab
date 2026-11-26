# Changelog

Notable changes to `raglab` follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [0.2.0] November 26, 2026

### Added

* Structure aware Markdown chunking (`split_markdown_sections`, `chunk_markdown`, `chunk_markdown_documents`) preserving ATX heading breadcrumbs in `metadata["section"]` and keeping fenced code blocks intact.
* Exact metadata prefiltering (`filters`) across `DenseRetriever`, `BM25Retriever`, and `HybridRetriever`, plus section heading indexing.
* Second stage `CrossTermReranker` scoring query term coverage, ordered bigram proximity, and section heading alignment.
* Maximal Marginal Relevance diversification (`diversify_mmr`) to suppress redundant sibling chunks.
* Token budget context packing (`estimate_tokens`, `pack_context`) and section aware context headers.
* Post generation `verify_answer` guardrail returning `GroundingReport` (`cited_indices`, `invalid_citations`, `grounded_ratio`, `abstained`, `is_verified`).
* Stage level latency and candidate funnel telemetry (`PipelineTrace`) on every `Answer`.
* `NDCG@k`, `Precision@k`, per category slice evaluation (`evaluate_by_category`), and full pipeline evaluation (`evaluate_pipeline` / `EndToEndReport`).
* CLI flags `--markdown`, `--rerank`, `--mmr`, `--max-tokens`, `--filter`, and `--show-trace`.
* Flagship `projects/incident_copilot` SRE Incident Triage and Architecture Copilot with 12 operational documents, 24 benchmark cases, autonomous playbook runner, and ablation suite.
* Deterministic pure Python SVG diagram and benchmark chart generator (`scripts/generate_charts.py`) producing `docs/assets/*.svg`.

## [0.1.0] November 26, 2026

### Added

* Directory loader for Markdown and plain text files.
* Sentence packing chunker with configurable character limit and overlap.
* Stateless `HashingEmbedder` and corpus fitted `TfidfEmbedder`.
* `InMemoryVectorStore` with exact cosine search and JSON persistence.
* `DenseRetriever`, `BM25Retriever`, and `HybridRetriever` with Reciprocal Rank Fusion.
* Grounded prompt builder, offline `ExtractiveGenerator`, and `OpenAICompatibleGenerator`.
* Retrieval evaluation (`Hit@k`, `MRR`, `Recall@k`), `raglab` CLI, and five progressive examples.
