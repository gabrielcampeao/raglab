# Evaluation mathematics, the RAG Triad, and failure taxonomy

Shipping changes to chunk sizes, embedding models, or prompt templates without an offline evaluation harness turns RAG engineering into guesswork. Fixing one query by hand routinely regresses three others. This study covers the exact metrics implemented in [evaluation.py](../src/raglab/evaluation.py) and [verification.py](../src/raglab/verification.py), shows a worked calculation, and catalogs the ten failure modes most common in production systems.

## 1. Rank unaware versus rank aware retrieval metrics

Suppose a benchmark query `q` has two relevant documents `R = {A, B}` and a retriever returns `top_3 = [X, A, B]`, where `X` is irrelevant.

| Metric | Formula | Value on `[X, A, B]` | What it measures and what it misses |
| --- | --- | ---: | --- |
| **`Hit@3`** | `1` if `top_k` intersects `R` else `0` | `1.00` | Binary sanity check. Cannot tell rank 1 apart from rank 3, and ignores whether `B` was found. |
| **`Recall@3`** | `|top_k intersect R| / |R|` | `1.00` | Completeness of retrieved evidence (`2 / 2`). Insensitive to `X` sitting at rank 1. |
| **`Precision@3`** | `count(d in R for d in top_k) / k` | `0.67` | Signal to noise ratio inside the prompt window (`2 / 3`). Penalizes irrelevant chunks that waste tokens and distract the generator. |
| **`MRR@3`** | `1 / rank_first_relevant` | `0.50` | Rewards placing at least one relevant document at rank 1 (`1 / 2` here). Ignores the rank of `B`. |
| **`NDCG@3`** | `DCG@k / IDCG@k` | `0.69` | Discounts every relevant hit logarithmically by rank (`1 / log2(rank + 1)`), rewarding both completeness and top heavy ordering. |

### Worked `NDCG@3` step by step

In `raglab`, binary document gains are deduplicated on first appearance of each relevant source so multiple sibling chunks from `A` cannot inflate `DCG` above `IDCG`:

```text
DCG@3([X, A, B]) = 0 / log2(2) + 1 / log2(3) + 1 / log2(4)
                 = 0.0000 + 0.6309 + 0.5000
                 = 1.1309

IDCG@3({A, B})   = 1 / log2(2) + 1 / log2(3)
                 = 1.0000 + 0.6309
                 = 1.6309

NDCG@3           = 1.1309 / 1.6309 = 0.6934
```

Whenever benchmark questions require **multi document synthesis** (`|R| >= 2`), optimize `NDCG@k` and `Recall@k` rather than `Hit@k` or `MRR@k`.

## 2. The RAG Triad for end to end quality

A RAG pipeline can fail even when `NDCG@3 = 1.00`, and it can appear to succeed when hallucinating from parametric memory. Evaluating end to end behavior requires checking three directional links known as the **RAG Triad**:

1. **Context Relevance (`Query -> Retrieved Chunks`):** Do the retrieved chunks focus on the question without dragging in noisy filler? Measured offline via `Precision@k` and `NDCG@k` against labeled sources, or online via reranker margin distributions.
2. **Groundedness / Faithfulness (`Retrieved Chunks -> Answer`):** Is every claim in the answer supported by the cited passages? Measured deterministically in `raglab` by `GroundingReport.grounded_ratio` and `invalid_citations == ()`.
3. **Answer Relevance & Calibration (`Query -> Answer`):** Does the answer address what the user asked when evidence exists, and **abstain cleanly** when evidence is absent? Measured in `EndToEndReport` via `verified_rate` and `abstention_accuracy`.

## 3. Calibration matrix on answerable versus unanswerable queries

Production traffic always contains questions outside the indexed corpus. Evaluating only answerable questions rewards overly permissive generators that never say *"I do not know"*.

| Corpus state \ System action | Emits cited answer | Abstains (`NO_ANSWER`) |
| --- | --- | --- |
| **Evidence present (`answerable=True`)** | Grounded hit (verify `[n]` support) | False abstention (recall or threshold too strict) |
| **Evidence absent (`answerable=False`)** | **Hallucinated citation** (severe trust failure) | **Valid trap refusal** (`abstention_accuracy`) |

Include at least `15%` to `20%` adversarial out of scope questions in every benchmark suite (`projects/incident_copilot/benchmark.json` dedicates 4 of 24 cases to unanswerable infrastructure traps sharing surface nouns like `timeout`, `gateway`, and `table`).

## 4. Ten production failure modes and remediation playbook

| # | Failure mode | Typical symptom | Telemetry signal | Concrete fix in raglab |
| --- | --- | --- | --- | --- |
| 1 | **Identifier dilution** | Error codes (`SQLSTATE 53300`, `HTTP 429`) miss rank 1 | Dense score high on wrong doc, BM25 score high on right doc | Pair `BM25Retriever` with dense vectors inside `HybridRetriever` |
| 2 | **Orphaned headings** | Extracted steps lack context about which service they govern | Chunks start mid procedure without section title | Enable `markdown_aware=True` so ATX paths populate `metadata["section"]` |
| 3 | **Sliced code blocks** | Partial SQL or shell flags emitted without `SELECT` or `WHERE` | Chunks contain unmatched ```` ``` ```` fences | Use `chunk_markdown_documents` and `ExtractiveGenerator` atomic fence units |
| 4 | **Sibling crowding** | All top 3 slots come from one verbose document | High `Hit@1` alongside low multi doc `Recall@3` | Enable `diversify_mmr(lambda_weight=0.65)` after reranking |
| 5 | **Compound query starvation** | Two part question answers part A and ignores part B | Both sub topics exist in corpus, only one retrieved | Wrap retriever in `MultiQueryRetriever(..., SubQuestionDecomposer())` |
| 6 | **Small vs big tension** | Short chunks match well but omit needed surrounding lines | High `MRR`, incomplete answers | Switch to `ParentChildRetriever` (`180c` children to `750c` parents) |
| 7 | **Pronoun drift in chat** | Follow up `"When do we escalate it?"` pulls wrong runbook | Turn 2 standalone tokens lack Turn 1 entity | Route multi turn sessions through `ConversationalRAG` |
| 8 | **Middle context burial** | Model ignores rank 2 passage in a 6+ chunk prompt | Relevant source retrieved at slot 3 or 4, omitted in answer | Pass `reorder_context=True` (`reorder_lost_in_middle`) |
| 9 | **Cross scope cache leak** | Filtered query returns answer cached from another service | `cache_hit=True` across mismatched `filters` | Use `SemanticCache`, which partitions buckets by canonical `filters` |
| 10 | **Indirect prompt injection** | Ingested ticket overrides system rules or hides zero width chars | `ContextSanitizer.last_findings` non empty | Pass `sanitizer=ContextSanitizer(drop_suspicious=True)` |
