# Engineering design notes

These notes record the core architectural tradeoffs behind `raglab` so changes can be evaluated against explicit constraints. See [architecture_spec.md](architecture_spec.md) for the full 0.2 specification and design review Q and A.

## Core constraints

* **Zero runtime dependencies:** Implementing tokenization, feature hashing, TFIDF, Okapi BM25, Reciprocal Rank Fusion, cross feature reranking, Maximal Marginal Relevance, token budgeting, citation verification, and SVG benchmark plotting in pure Python keeps every equation transparent and ensures the package installs cleanly in airgapped or minimal environments.
* **Two stage precision over single stage recall:** Single stage vector or BM25 lookup struggles on multi document questions and sibling chunk redundancy. Separating candidate generation (`HybridRetriever`) from rescoring (`CrossTermReranker`) and diversification (`diversify_mmr`) mirrors production search stacks at miniature scale.
* **Explicit abstention and verification:** Every `Answer` carries both a `PipelineTrace` and a `GroundingReport`. A pipeline that scores `1.00` on answerable questions while fabricating citations on out of scope traps is unsafe in production.
* **Self documenting code:** Python modules omit inline comments and docstrings entirely, relying on small pure functions, frozen dataclasses, and explicit type signatures.

## Summary of technical choices

| Subsystem | Choice | Rationale |
| --- | --- | --- |
| Language floor | Python 3.10+ standard library | Structural pattern matching, `pairwise`, modern union types, zero external wheel churn |
| Structural ingestion | ATX heading stack + atomic code fences | Keeps SQL and shell commands intact and attaches `section` breadcrumbs to every chunk |
| First stage retrieval | Dense cosine + Okapi BM25 fused via RRF (`k=60`) | Scale invariant rank fusion combining exact identifier recall with n gram similarity |
| Metadata filtering | Prefiltering inside each retriever | Prevents dominant services from starving small metadata slices |
| Second stage ranking | `CrossTermReranker` + `diversify_mmr` | Captures phrase proximity and heading alignment while eliminating redundant sibling chunks |
| Guardrails | Coverage gated extraction + `verify_answer` | Catches out of bounds `[n]` markers and forces abstention on unsupported queries |
| Benchmarks and visuals | Deterministic pure Python SVG generator | Eliminates stale screenshots and guarantees charts match live benchmark JSON |
| License | MIT | Permissive reuse across commercial and academic work |
