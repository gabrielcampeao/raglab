# raglab 0.2 Senior Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB SKILL: Use `superpowers:executing-plans` with `superpowers:test-driven-development` to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `raglab` into a senior two stage RAG engine with structure aware Markdown chunking, metadata prefiltering, lexical cross feature reranking, MMR diversification, token budget packing, citation grounding verification, richer evaluation metrics, reproducible SVG charts, and the `projects/incident_copilot` flagship SRE system.

**Architecture:** Extend `src/raglab` backward compatibly with focused pure Python modules (`reranking.py`, `verification.py`) and surgical upgrades to `chunking.py`, `retrievers.py`, `prompting.py`, `generation.py`, `evaluation.py`, `pipeline.py`, and `cli.py`. Build `projects/incident_copilot` on top of the public package API and render deterministic SVG diagrams and benchmark plots from measured outputs via `scripts/generate_charts.py`.

**Tech Stack:** Python 3.10+ standard library only (`dataclasses`, `math`, `re`, `time`, `inspect`, `json`, `xml.etree.ElementTree`), `pytest`, `pytest-cov`, `ruff`.

**Spec:** `docs/architecture_spec.md`

## Global Constraints

* Runtime dependencies must remain empty (`dependencies = []` in `pyproject.toml`).
* Zero Python comments (`#`) and zero docstrings anywhere in `.py` files.
* All documentation, UI labels, CLI help, and commit messages in natural English following Humanizer rules, with zero em dashes, zero en dashes, and zero hyphens in prose or commit subjects.
* Every behavioral addition follows strict TDD: write the failing test first, run pytest to confirm failure, implement minimal clean code, run full pytest suite to confirm pass, then commit with a Conventional Commit message containing no hyphens or dashes.

## Review Focus

* Fenced Markdown code blocks larger than `max_chars` must still split cleanly without infinite loops while smaller fenced blocks stay atomic.
* Single token queries after stopword removal (such as `"What is PgBouncer?"`) must remain answerable even when `ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.35)` is configured.
* Metadata prefiltering that matches zero chunks must return `[]` cleanly across `DenseRetriever`, `BM25Retriever`, and `HybridRetriever` without division by zero or dimension errors.
* Hallucinated out of range citation markers such as `[9]` when only 2 sources were retrieved must be flagged in `GroundingReport.invalid_citations` and never raise `IndexError`.
* `evaluate_retrieval` and `raglab evaluate` must handle mixed datasets containing `answerable=False` cases (`relevant_sources=()`) without crashing, while still rejecting `answerable=True` cases that omit `relevant_sources`.

---

### Task 1: Structure aware Markdown chunking

**Files:**
* Modify: `src/raglab/chunking.py`
* Modify: `src/raglab/__init__.py`
* Modify: `tests/test_chunking.py`

**Interfaces:**
* Consumes: `Document`, `Chunk` from `raglab.models`
* Produces: `MarkdownSection(heading_path: tuple[str, ...], body: str)`, `split_markdown_sections(text: str) -> list[MarkdownSection]`, `chunk_markdown(text: str, max_chars: int = 500, overlap: int = 1) -> list[tuple[str, str]]`, `chunk_markdown_documents(documents: Iterable[Document], max_chars: int = 500, overlap: int = 1) -> list[Chunk]` (populating `metadata["section"]` with `" > ".join(heading_path)` when headings exist and keeping fenced code blocks intact when they fit `max_chars`)

* [x] **Step 1: Write failing tests in `tests/test_chunking.py`** for heading hierarchy tracking, atomic fenced code blocks within `max_chars`, oversized fenced code block fallback splitting, and `chunk_markdown_documents` setting `metadata["section"]`.
* [x] **Step 2: Run pytest on `tests/test_chunking.py` and confirm failure (`ImportError`).**
* [x] **Step 3: Implement `MarkdownSection`, `split_markdown_sections`, `chunk_markdown`, and `chunk_markdown_documents` in `src/raglab/chunking.py` and export them in `src/raglab/__init__.py`.**
* [x] **Step 4: Run `pytest` and `ruff check .` and confirm full pass.**
* [x] **Step 5: Commit `feat(chunking): add markdown section and code block aware chunker`.**

### Task 2: Metadata prefiltering across retrievers

**Files:**
* Modify: `src/raglab/retrievers.py`
* Modify: `src/raglab/__init__.py`
* Modify: `tests/test_retrievers.py`

**Interfaces:**
* Consumes: `Chunk`, `SearchResult` from `raglab.models`
* Produces: `MetadataFilter = Mapping[str, str]`, `matches_filters(chunk: Chunk, filters: MetadataFilter | None) -> bool`, and optional `filters: MetadataFilter | None = None` on `Retriever.retrieve`, `DenseRetriever.retrieve`, `BM25Retriever.retrieve`, and `HybridRetriever.retrieve`

* [x] **Step 1: Write failing tests in `tests/test_retrievers.py`** covering exact single and multi key `filters` across all four named retrievers (`dense`, `tfidf`, `bm25`, `hybrid`) plus empty result when no chunk matches `filters`.
* [x] **Step 2: Run pytest on `tests/test_retrievers.py` and confirm failure (`TypeError: unexpected keyword argument 'filters'`).**
* [x] **Step 3: Implement `matches_filters` and prefiltered retrieval in `src/raglab/retrievers.py` (searching full store in `DenseRetriever` before filtering and slicing `[:k]`).**
* [x] **Step 4: Run `pytest` and `ruff check .` and confirm full pass.**
* [x] **Step 5: Commit `feat(retrieval): support metadata prefiltering across all retrievers`.**

### Task 3: Cross term reranking and MMR diversification

**Files:**
* Create: `src/raglab/reranking.py`
* Modify: `src/raglab/__init__.py`
* Create: `tests/test_reranking.py`

**Interfaces:**
* Consumes: `SearchResult` from `raglab.models`, `tokenize` from `raglab.text`
* Produces: `Reranker(ABC)` with `rerank(query: str, results: Sequence[SearchResult], k: int | None = None) -> list[SearchResult]`, `CrossTermReranker`, and `diversify_mmr(results: Sequence[SearchResult], k: int, lambda_weight: float = 0.75) -> list[SearchResult]`

* [x] **Step 1: Write failing tests in `tests/test_reranking.py`** verifying that `CrossTermReranker` promotes exact ordered phrase and section heading matches above scattered term matches, and `diversify_mmr` demotes near duplicate chunks in favor of complementary evidence while validating `k` and `lambda_weight`.
* [x] **Step 2: Run pytest on `tests/test_reranking.py` and confirm failure (`ModuleNotFoundError`).**
* [x] **Step 3: Implement `src/raglab/reranking.py` and export `Reranker`, `CrossTermReranker`, and `diversify_mmr` in `src/raglab/__init__.py`.**
* [x] **Step 4: Run `pytest` and `ruff check .` and confirm full pass.**
* [x] **Step 5: Commit `feat(reranking): add cross feature reranker and mmr diversification`.**

### Task 4: Token budget context packing, stricter extractive abstention, and citation grounding verification

**Files:**
* Modify: `src/raglab/prompting.py`
* Modify: `src/raglab/generation.py`
* Create: `src/raglab/verification.py`
* Modify: `src/raglab/__init__.py`
* Modify: `tests/test_prompting.py`
* Modify: `tests/test_generation.py`
* Create: `tests/test_verification.py`

**Interfaces:**
* Produces in `prompting.py`: `estimate_tokens(text: str) -> int`, `pack_context(results: Sequence[SearchResult], max_tokens: int) -> list[SearchResult]` (plus section label formatting in `format_context` when `chunk.metadata.get("section")` is present)
* Produces in `generation.py`: `ExtractiveGenerator(max_sentences: int = 2, *, min_shared_terms: int = 1, min_overlap_ratio: float = 0.0)`
* Produces in `verification.py`: `GroundingReport(abstained: bool, cited_indices: tuple[int, ...], invalid_citations: tuple[int, ...], grounded_sentences: int, total_sentences: int, grounded_ratio: float)` with `.is_verified` property (`invalid_citations == () and (abstained or grounded_ratio >= 0.5)`), and `verify_answer(answer_text: str, sources: Sequence[SearchResult]) -> GroundingReport`

* [x] **Step 1: Write failing tests in `tests/test_prompting.py`, `tests/test_generation.py`, and `tests/test_verification.py`** covering token estimation, budget packing, single token query clamping under `min_shared_terms=2`, abstention on weak single word coincidences, valid citation grounding, and out of range citation detection (`[9]`).
* [x] **Step 2: Run pytest on the three test files and confirm failure.**
* [x] **Step 3: Implement updates in `prompting.py`, `generation.py`, and `verification.py`, and export new symbols in `src/raglab/__init__.py`.**
* [x] **Step 4: Run `pytest` and `ruff check .` and confirm full pass.**
* [x] **Step 5: Commit `feat(verification): add context token budgeting and grounding verifier`.**

### Task 5: Instrumented two stage pipeline, richer evaluation metrics, and CLI flags

**Files:**
* Modify: `src/raglab/pipeline.py`
* Modify: `src/raglab/evaluation.py`
* Modify: `src/raglab/cli.py`
* Modify: `src/raglab/__init__.py`
* Modify: `tests/test_pipeline.py`
* Modify: `tests/test_evaluation.py`
* Modify: `tests/test_cli.py`

**Interfaces:**
* Produces in `pipeline.py`: `PipelineTrace(retrieval_ms: float, rerank_ms: float, generation_ms: float, total_ms: float, candidates_retrieved: int, candidates_selected: int, context_tokens: int)`, updated `Answer` with `grounding: GroundingReport` and `trace: PipelineTrace`, updated `RAGPipeline` supporting `reranker`, `candidate_pool`, `mmr_lambda`, `max_context_tokens`, `markdown_aware`, and `filters`
* Produces in `evaluation.py`: `EvalCase` with `category`, `answerable`, `filters`, `RetrievalReport` with `ndcg` and `precision`, `EndToEndReport(cases: int, answerable_cases: int, unanswerable_cases: int, retrieval: RetrievalReport, grounded_ratio: float, verified_rate: float, abstention_accuracy: float, mean_latency_ms: float)`, `evaluate_by_category`, and `evaluate_pipeline`
* Produces in `cli.py`: `--markdown`, `--rerank`, `--mmr`, `--max-tokens`, `--filter key=value`, `--show-trace`, plus `ndcg` column in `raglab evaluate`

* [x] **Step 1: Write failing tests in `tests/test_pipeline.py`, `tests/test_evaluation.py`, and `tests/test_cli.py`.**
* [x] **Step 2: Run pytest on those three files and confirm failure.**
* [x] **Step 3: Implement `pipeline.py`, `evaluation.py`, `cli.py`, and `__init__.py` updates.**
* [x] **Step 4: Run `pytest` and `ruff check .` and confirm full pass.**
* [x] **Step 5: Commit `feat(pipeline): wire reranking, telemetry, ndcg metrics and cli flags`.**

### Task 6: Flagship project `projects/incident_copilot` with autonomous Q and A playbook

**Files:**
* Create: `projects/incident_copilot/corpus/runbooks/*.md` (6 files)
* Create: `projects/incident_copilot/corpus/postmortems/*.md` (3 files)
* Create: `projects/incident_copilot/corpus/architecture/*.md` (3 files)
* Create: `projects/incident_copilot/benchmark.json` (24 labeled cases across 4 categories)
* Create: `projects/incident_copilot/__init__.py`
* Create: `projects/incident_copilot/copilot.py`
* Create: `projects/incident_copilot/run_playbook.py`
* Create: `projects/incident_copilot/README.md`
* Create: `tests/test_incident_copilot.py`

**Interfaces:**
* Produces in `copilot.py`: `load_incident_corpus(root: Path | None = None) -> list[Document]` (attaching `kind` and `service` metadata from directory and header conventions), `build_incident_copilot() -> RAGPipeline`, `PLAYBOOK_QUESTIONS: tuple[tuple[str, dict[str, str] | None], ...]`, `run_autonomous_playbook() -> list[Answer]`, `run_ablation_suite() -> dict[str, Any]`, `run_chunk_sweep() -> list[dict[str, float | int]]`

* [x] **Step 1: Write failing tests in `tests/test_incident_copilot.py`** checking corpus ingestion (`12` documents with `kind` metadata), autonomous playbook execution (answering incident questions with verified citations and abstaining on out of scope traps), and ablation benchmark quality (`senior_full` beating `dense` baseline and achieving `>= 0.90` hit rate at 3 and `1.0` abstention accuracy).
* [x] **Step 2: Run pytest on `tests/test_incident_copilot.py` and confirm failure.**
* [x] **Step 3: Create the 12 SRE documents, `benchmark.json`, `copilot.py`, `run_playbook.py`, and `projects/incident_copilot/README.md`.**
* [x] **Step 4: Run `pytest` and `ruff check .` and confirm full pass.**
* [x] **Step 5: Commit `feat(copilot): build incident triage copilot with self answering playbook`.**

### Task 7: Reproducible architecture and benchmark SVG charts plus senior documentation overhaul

**Files:**
* Create: `scripts/generate_charts.py`
* Create: `projects/incident_copilot/benchmark_results.json`
* Create: `docs/assets/architecture_overview.svg`
* Create: `docs/assets/retrieval_ablation.svg`
* Create: `docs/assets/chunk_size_tradeoff.svg`
* Create: `docs/assets/latency_vs_quality.svg`
* Create: `tests/test_charts.py`
* Modify: `README.md`, `docs/how_rag_works.md`, `docs/use_cases.md`, `docs/design_notes.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `pyproject.toml`

* [x] **Step 1: Write failing tests in `tests/test_charts.py`** verifying that `benchmark_results.json` and all four SVG files exist, parse as valid XML with `ElementTree`, contain zero em/en dashes, and match live `generate_charts` output.
* [x] **Step 2: Run pytest on `tests/test_charts.py` and confirm failure.**
* [x] **Step 3: Implement `scripts/generate_charts.py`, run it to generate `benchmark_results.json` and `docs/assets/*.svg`, and upgrade `README.md` and `docs/*` with embedded charts, quantitative analysis, senior positioning, and zero hyphens/dashes/AI prose.**
* [x] **Step 4: Run `pytest --cov`, `ruff check .`, `ruff format --check .`, comment/docstring AST check, and prose dash check.**
* [x] **Step 5: Commit `docs: add benchmark charts, architecture diagram and senior guide`.**
