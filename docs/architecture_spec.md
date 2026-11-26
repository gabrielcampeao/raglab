# Architecture specification for raglab 0.2

## Intent and positioning

raglab is a zero dependency Retrieval Augmented Generation reference engine and benchmark suite in Python. Rather than wrapping opaque third party frameworks, it implements every stage of a modern two stage RAG architecture in compact, typed modules using only the Python standard library: structure aware Markdown chunking, sparse and dense indexing, metadata prefiltering, Reciprocal Rank Fusion, lexical cross feature reranking, Maximal Marginal Relevance diversification, token budgeted context assembly, inline citation verification, and retrieval plus faithfulness evaluation.

Alongside the core library, `projects/incident_copilot` ships a complete operational system built on raglab: an SRE Incident Triage and Architecture Copilot for a payment platform (`Atlas Pay`), complete with runbooks, postmortems, architecture decision records, an autonomous question answering playbook that poses and verifies operational incidents on its own, and reproducible ablation benchmarks.

## Goals

* Implement a production shaped two stage retrieval and verification pipeline in pure Python 3.10+ with zero runtime dependencies.
* Preserve document structure during ingestion so headings and fenced code blocks survive chunking intact.
* Support metadata filtering across dense, BM25, and hybrid retrievers.
* Improve top k precision and diversity through second stage lexical cross feature reranking and Maximal Marginal Relevance selection.
* Enforce token budgets during prompt assembly and verify every generated answer for valid citations, lexical grounding, and appropriate abstention.
* Instrument every query with stage latency and candidate funnel telemetry.
* Provide reproducible charts in `docs/assets` generated directly from measured benchmark numbers, embedded and analyzed in `README.md` and `docs/how_rag_works.md`.
* Ship `projects/incident_copilot` as a realistic flagship application with autonomous Q and A execution and multi tier evaluation.

## Non goals

* Neural transformer weights or external vector databases inside the core package. Interfaces (`Embedder`, `Retriever`, `Reranker`, `Generator`) remain open so external models drop in cleanly.
* Distributed indexing across multiple machines. Single process memory structures suffice for corpora up to tens of thousands of chunks.

## Anti goals

* Comments or docstrings in Python files. Module structure, type annotations, and small functions must carry all intent.
* Marketing prose, vague claims, emojis, hyphens in prose, or em and en dashes anywhere in documentation or commit messages.
* Synthetic numbers in charts. Every chart in `docs/assets` must render measurements produced by running the repository benchmark script.

## Self answered architectural Q and A

Before locking the design, we pose and settle the critical engineering questions that separate a toy RAG script from a senior architecture.

### 1. Why does naive sentence chunking fail on real engineering documents, and how should raglab chunk Markdown?

Engineering runbooks mix ATX headings (`#`, `##`, `###`), prose paragraphs, shell commands, and SQL blocks. Splitting strictly on punctuation slices fenced code blocks mid command and detaches remediation steps from the heading that scopes them. `chunk_markdown` tracks the active heading stack as `section` metadata (for example `Postgres pool saturation > Immediate mitigation`), keeps fenced code blocks atomic whenever they fit within `max_chars`, and prepends the section path to the searchable chunk text or stores it in `metadata["section"]` so both retrieval and citations preserve structural context.

### 2. When should metadata filtering happen: before or after retrieval?

Post filtering top k results starves the caller whenever the matching metadata slice is small relative to the corpus. Filtering belongs inside each retriever before top k truncation (`prefiltering`), matching every field in `filters: Mapping[str, str]` against `chunk.metadata` before ranking candidates.

### 3. Why add a second stage reranker on top of Hybrid RRF if BM25 and Dense already ran?

BM25 and feature hashed vectors score unigrams and unordered bigrams independently across the whole corpus. They ignore whether query terms appear adjacent in order, whether they match the chunk's section heading, and what fraction of distinct query terms a single chunk covers. `CrossTermReranker` inspects only the top candidate pool (for example 20 chunks) and combines RRF prior score, distinct query term coverage, ordered phrase window proximity, and section heading overlap. Running that richer feature set on 20 candidates adds sub millisecond latency while lifting rank 1 precision (`Hit@1` and `MRR`).

### 4. Why do overlapping chunks hurt generation quality, and how do we prevent context redundancy?

Overlapping chunks or repetitive sections often cause the top 3 results to repeat nearly identical sentences from the same document, wasting prompt budget and crowding out secondary evidence needed for multi section questions. `diversify_mmr` applies Maximal Marginal Relevance over candidate token sets (`lambda_weight * relevance - (1 - lambda_weight) * max_jaccard_similarity`), trading redundant near duplicates for complementary chunks.

### 5. How should the prompt builder handle strict context window budgets?

Fixed `top_k` counts chunks, not tokens, so three large chunks can overflow a tight latency or cost budget while five short chunks fit easily. `pack_context` estimates token counts (`max(1, round(len(text) / 4))` or word based heuristic) and greedily packs ranked results until `max_tokens` would be exceeded, always keeping at least the top result if it fits or truncating cleanly.

### 6. How do we verify grounding and catch unanswerable questions without calling a second LLM judge?

`verify_answer` inspects the generated answer against the retrieved `SearchResult` sequence and returns a `GroundingReport`:
* `abstained`: true when the answer matches `NO_ANSWER` or explicit refusal patterns.
* `cited_indices` and `invalid_citations`: parses `[n]` markers and flags indices outside `1..len(sources)`.
* `grounded_sentences` and `total_sentences`: checks each non abstaining claim sentence for content token overlap (`>= 0.4` of sentence content terms or at least 2 shared terms) against the cited chunk (or all retrieved chunks if uncited).
* `grounded_ratio`: `grounded_sentences / total_sentences` (`1.0` on valid abstention with empty claims, `0.0` when invalid citations appear without support).

In `ExtractiveGenerator`, requiring a minimum query overlap ratio (`min_term_matches` / `min_coverage`) prevents weak single word coincidences on common technical nouns from fabricating answers to out of scope questions.

### 7. Which retrieval metrics matter beyond Hit Rate and MRR?

`Hit@k` is binary and ignores rank; `MRR` stops at the first relevant hit and ignores second or third required documents. Adding `ndcg` (`NDCG@k` using binary document relevance deduplicated by first appearance of each source, or chunk level relevance) and `precision` (`Precision@k`) exposes ranking decay and noise in the context window. Grouping metrics by query category (`exact_signal`, `paraphrase`, `scoped_triage`, `unanswerable`) reveals *where* each retriever wins or breaks.

### 8. How do we produce publication quality charts without adding heavy plotting dependencies or risking stale numbers?

A pure Python SVG renderer in `scripts/generate_charts.py` executes the benchmark suite and chunking sweep, writes raw measurements to `projects/incident_copilot/benchmark_results.json`, and emits crisp, self contained vector charts (`docs/assets/*.svg`) with embedded styling, exact axis ticks, data labels, and legends. Automated tests parse every SVG as XML and verify its synchronization with the benchmark runner.

### 9. How do 0.2 features preserve backward compatibility with 0.1 callers, short queries, unanswerable benchmarks, and per case filters?

* `RAGPipeline()` keeps 0.1 defaults (`markdown_aware=False`, `reranker=None`, `mmr_lambda=None`, `max_context_tokens=None`), while `IncidentCopilot` and opt in CLI flags activate the full two stage stack.
* `ExtractiveGenerator` defaults to `min_shared_terms=1` and `min_overlap_ratio=0.0`. When stricter values are configured, `effective_min_terms = min(min_shared_terms, len(query_terms))` ensures single token queries remain answerable, evaluating overlap against chunk body plus section terms and accepting candidates that satisfy `effective_overlap >= effective_min_terms` along with either `coverage >= min_overlap_ratio` or at least `max(3, min_shared_terms)` direct sentence term matches.
* `EvalCase` stays frozen and hashable by storing `category: str = "general"`, `answerable: bool = True`, and `filters: tuple[tuple[str, str], ...] = ()`. `evaluate_retrieval` scores only `answerable=True` cases (raising `ValueError` if an answerable case has no `relevant_sources` or if no answerable cases exist) and forwards `filters=case.filter_map` whenever `retrieve` accepts a `filters` keyword parameter. `evaluate_by_category` returns `dict[str, RetrievalReport]` across categories that contain answerable cases, while `evaluate_pipeline` evaluates both answerable grounding and unanswerable abstention across the full dataset.

## Module and directory design

* `src/raglab/chunking.py`: adds `MarkdownSection`, `split_markdown_sections`, `chunk_markdown`, and `chunk_markdown_documents`.
* `src/raglab/retrievers.py`: adds optional `filters: Mapping[str, str] | None = None` to `Retriever.retrieve`, `DenseRetriever`, `BM25Retriever`, and `HybridRetriever`, plus `matches_filters`.
* `src/raglab/reranking.py`: defines `Reranker`, `CrossTermReranker`, and `diversify_mmr`.
* `src/raglab/prompting.py`: adds `estimate_tokens` and `pack_context`.
* `src/raglab/verification.py`: defines `GroundingReport` and `verify_answer`.
* `src/raglab/generation.py`: upgrades `ExtractiveGenerator` with configurable `min_overlap_ratio` / `min_shared_terms` so unanswerable queries abstain reliably instead of matching a stray word.
* `src/raglab/pipeline.py`: adds `PipelineTrace`, optional `reranker`, `mmr_lambda`, `max_context_tokens`, `markdown_aware` chunking, `filters` in `retrieve`/`ask`, and attaches `trace` and `grounding` to `Answer`.
* `src/raglab/evaluation.py`: adds `EvalCase.category` and `EvalCase.answerable`, `ndcg` and `precision` on `RetrievalReport`, `EndToEndReport`, `evaluate_by_category`, and `evaluate_pipeline`.
* `src/raglab/cli.py`: supports `--rerank`, `--max-tokens`, `--filter key=value`, and a `copilot` / `benchmark` flow or clean CLI flags.
* `projects/incident_copilot/`:
  * `corpus/runbooks/*.md`, `corpus/postmortems/*.md`, `corpus/architecture/*.md` (12 realistic SRE documents)
  * `benchmark.json` (24 labeled queries across `exact_signal`, `paraphrase`, `scoped_triage`, and `unanswerable`)
  * `copilot.py` (reusable `IncidentCopilot` assembling metadata enriched ingestion, hybrid RRF, cross term reranking, MMR, token budget packing, and grounding verification, plus autonomous playbook runner and ablation suite)
  * `run_playbook.py` (CLI entry script that runs the self answering incident triage playbook and ablation table)
  * `README.md` (dedicated engineering writeup of the flagship system)
* `scripts/generate_charts.py` and `docs/assets/*.svg`:
  * `architecture_overview.svg`
  * `retrieval_ablation.svg`
  * `chunk_size_tradeoff.svg`
  * `latency_vs_quality.svg`
