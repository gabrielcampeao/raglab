# Atlas Pay Incident Copilot

`incident_copilot` is an end to end operational engineering assistant built on `raglab`. It indexes production runbooks, incident postmortems, and architecture decision records for a payment and double entry ledger platform (`Atlas Pay`), then runs an autonomous incident triage playbook and a multi stage retrieval ablation benchmark.

## Why this exists inside raglab

Toy RAG examples usually stop at half a dozen short FAQ paragraphs where any retriever scores `1.00`. Operational engineering corpora stress the failure modes that show up in production:

1. **Exact error tokens mixed with natural symptom descriptions:** On call engineers search both literal identifiers (`SQLSTATE 53300`, `exit code 137`, `CREATE INDEX CONCURRENTLY`, `14.4x`) and paraphrased symptoms (`mobile shoppers charged twice during packet loss`).
2. **Hierarchical Markdown structure:** Remediation commands live under headings such as `## Immediate mitigation` alongside SQL diagnostic blocks that lose meaning if split mid query.
3. **Scoped triage:** Postmortem review questions should not surface emergency runbooks, and runbook lookups during an active outage should filter to `kind=runbook` and the affected `service`.
4. **Out of scope trap questions:** Asking an internal assistant about unindexed infrastructure (`GraphQL federation timeouts`, `NVIDIA H100 training nodes`, `Snowflake archive policies`) must trigger clean abstention rather than fabricated guidance stitched from single word overlaps.

## Corpus layout

| Directory | Documents | Metadata tags | Examples |
| --- | --- | --- | --- |
| `corpus/runbooks/` | 6 | `kind=runbook`, `service` | Postgres pool saturation, Kafka lag, Redis stampede, Kubernetes OOMKilled, mTLS rotation, webhook backlog |
| `corpus/postmortems/` | 3 | `kind=postmortem`, `service` | April duplicate charge storm, July Frankfurt failover, September CDN `HTTP 429` false positives |
| `corpus/architecture/` | 3 | `kind=architecture`, `service` | Double entry ledger and idempotency contract, expand and contract migrations, multi window SLO burn rates |

## Pipeline configuration

`build_incident_copilot()` in [copilot.py](copilot.py) assembles the full two stage `raglab` stack:

* **Ingestion:** `markdown_aware=True` (`max_chars=340`, `overlap=1`), tracking ATX heading breadcrumbs in `metadata["section"]`, loading `corpus/metadata.json` facets (`kind`, `service`), and keeping fenced SQL blocks intact.
* **First stage retrieval:** `HybridRetriever` fusing `HashingEmbedder` dense cosine similarity and `BM25Retriever` via Reciprocal Rank Fusion over a candidate pool of 16 chunks, with metadata prefiltering (`kind`, `service`).
* **Second stage reranking and diversity:** `CrossTermReranker` scoring query term coverage, ordered bigram proximity, and section heading overlap, followed by `diversify_mmr(lambda_weight=0.65)` to drop redundant sibling chunks.
* **Budgeting and verification:** `pack_context(max_tokens=260)` bounds prompt size, `ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34)` refuses weak coincidences, and `verify_answer` checks citation bounds and lexical support on every claim sentence.

## Running the autonomous playbook and benchmark

```bash
python projects/incident_copilot/run_playbook.py
```

Or query the corpus directly through the `raglab` CLI using the **Qdrant** vector backend, metadata filters, and execution traces:

```bash
raglab ask "Workers emit SQLSTATE 53300. What is the immediate mitigation?" \
  --data projects/incident_copilot/corpus \
  --backend qdrant \
  --filter kind=runbook --filter service=postgres \
  --markdown --rerank --mmr 0.65 --max-tokens 260 --show-trace
```
