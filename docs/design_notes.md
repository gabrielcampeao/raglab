# Design notes

These notes record the decisions made before writing code, so later changes can be checked against the original intent.

## Intent

raglab is a study project. It exists to explain how Retrieval Augmented Generation works by implementing every stage in code short enough to read in one sitting. Someone who finishes the README, the guide in `docs/how_rag_works.md` and the examples should be able to explain what each stage does, why it is there, and what changes when its settings change.

## Goals

* Cover the full loop: load, chunk, embed, store, retrieve, augment, generate and evaluate.
* Run offline with no runtime dependencies, so `pip install` and the tests work anywhere Python 3.10 runs.
* Show more than one retrieval strategy and a way to measure them against each other.
* Make the jump to a real language model a one line change.
* Include realistic examples of how RAG is used.

## Non goals

* Production performance. Brute force cosine search and dense Python lists are fine for thousands of chunks.
* Neural embeddings in the core package. They need large downloads and would hide the math. The `Embedder` interface leaves room to plug one in.
* Vendor SDKs. One small HTTP client covers OpenAI and every server that copies its API.

## Anti goals

* Magic. If a stage cannot be understood by reading its module, the design has failed even if the tests pass.
* Answers without sources. Every answer path keeps track of which chunks it used.

## Decisions

| Area | Decision | Reason |
| --- | --- | --- |
| Language | Python 3.10 or newer, standard library only | Readable and widely taught |
| Chunking | Sentence packing with sentence overlap | Easier to reason about than token windows |
| Embeddings | Feature hashing and TFIDF | Transparent, deterministic, no downloads |
| Retrieval | Dense, BM25 and hybrid with RRF | Shows the tradeoff between keyword and vector search |
| Generation | Extractive by default, OpenAI compatible client optional | Tests stay offline, and real models are one flag away |
| Evaluation | Hit rate, MRR and recall at k | The standard first metrics for retrieval |
| Style | No comments or docstrings; names and types carry the meaning | Keeps modules short and pushes toward clear names |
| Commits | Conventional Commits checked by a local hook | Readable history for a study log |
| License | MIT | Simple and permissive |

## Left to the builder

* The sample knowledge base topic. A fictional note taking app was chosen because support questions are the most common RAG use case.
* Exact default values (`max_chars=500`, `overlap=1`, `top_k=3`, `rrf_k=60`), based on common starting points.
