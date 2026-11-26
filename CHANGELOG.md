# Changelog

This file records notable changes to raglab. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] November 26, 2026

### Added

* Directory loader for Markdown and text files.
* Sentence based chunking with configurable size and overlap.
* Hashing and TFIDF embedders.
* In memory vector store with cosine search and JSON persistence.
* Dense, BM25 and hybrid retrievers, the last one using Reciprocal Rank Fusion.
* Prompt builder with numbered sources and grounding rules.
* Offline extractive generator and an OpenAI compatible generator.
* Retrieval evaluation with hit rate, MRR and recall at k.
* `raglab` command line tool with `ask`, `search` and `evaluate`.
* Sample knowledge base, evaluation questions and five runnable examples.
* Guides on how RAG works and where it is useful.
