# Production RAG patterns and parameter regimes

Retrieval Augmented Generation works whenever answers reside in mutable documents that require source attribution and scoped access. Different domains stress different stages of the pipeline. Each pattern below specifies what to index, how to configure `raglab`, and what failure modes to watch.

## 1. On call SRE incident triage and architecture review

Implemented end to end in [projects/incident_copilot](../projects/incident_copilot/README.md).

* **Corpus:** Operational runbooks, blameless postmortems, architecture decision records, and SLO specifications.
* **Recommended configuration:** `markdown_aware=True`, `max_chars=340`, `HybridRetriever`, `CrossTermReranker`, `mmr_lambda=0.65`, `max_context_tokens=260`, plus `filters={"kind": "runbook"}` during active paging.
* **Why this regime fits:** Incident queries combine high entropy literals (`SQLSTATE 53300`, `exit code 137`, `max.poll.interval.ms`) with paraphrased symptoms (`workers keep leaving the group`). Markdown section breadcrumbs keep diagnostic SQL blocks tied to the right runbook phase, while MMR prevents one verbose postmortem from monopolizing all prompt slots.

## 2. Customer support and policy resolution

Demonstrated in `examples/data` and `examples/01_quickstart.py`.

* **Corpus:** Help center articles, billing and refund terms, platform release notes.
* **Recommended configuration:** `HybridRetriever`, `CrossTermReranker`, `max_chars=400`, `overlap=1`, strict abstention thresholds so billing exceptions never guess.
* **Operational benefit:** Updating a refund window or regional availability rule requires editing one Markdown file and reindexing rather than fine tuning model weights, and every reply links the exact policy file for human audit.

## 3. Regulatory, legal, and security compliance review

* **Corpus:** Master service agreements, SOC 2 controls, data processing addenda, internal security standards.
* **Recommended configuration:** `markdown_aware=True`, smaller `max_chars` (`250` to `320`), `overlap=1`, `ExtractiveGenerator` or low temperature `OpenAICompatibleGenerator` (`temperature=0.0`), mandatory check on `answer.grounding.is_verified` and `answer.grounding.invalid_citations == ()`.
* **Operational benefit:** Legal reviewers often prefer verbatim clause extraction with `[n]` section citations over paraphrased prose that can subtly alter contractual obligations.

## 4. Internal platform and SDK documentation

* **Corpus:** Versioned API guides, migration notes, CLI manuals, architecture specs.
* **Recommended configuration:** Prefilter by `filters={"version": "v2"}` so deprecated v1 flags never leak into v2 answers; pair `BM25` and `CrossTermReranker` so exact symbol names (`CREATE INDEX CONCURRENTLY`, `Idempotency_Key`) outrank prose summaries.

## When RAG is the wrong architecture

* **Global aggregation queries:** Questions such as *count every Sev2 postmortem in Q3* or *summarize all 400 customer tickets from Friday* cannot be answered from `top_k = 3` passages. Route structured aggregations to SQL or analytics stores.
* **Live transactional state:** Account balances, deployment health, and order statuses belong behind deterministic tool calls or database reads, not static text indexes.
* **Behavioral or format adaptation:** Teaching a model a terse output schema or domain DSL without changing factual knowledge is handled better by system prompts, structured output schemas, or fine tuning.
* **Small static corpora:** When the entire authoritative reference is under a few thousand tokens and rarely changes, passing the full text in a cached system prompt avoids retrieval misses altogether.
