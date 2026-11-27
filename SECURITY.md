# Security policy and RAG threat model

## Supported versions

| Version | Supported |
| --- | --- |
| `0.4.x` | Yes |
| `< 0.4.0` | No |

## Reporting a vulnerability

Do not open a public GitHub issue for suspected security vulnerabilities. Report security findings privately through GitHub Security Advisories (`Security > Advisories > Report a vulnerability`) with a minimal reproduction script, affected module path, and impact assessment. Reports receive an initial triage response within 72 hours.

## RAG security threat model and built in defenses

Retrieval Augmented Generation introduces attack surfaces that traditional web services do not face because untrusted external text is injected directly into model context. `raglab` implements defense in depth across four boundaries:

### 1. Indirect prompt injection and Unicode smuggling (`src/raglab/security.py`)

Attackers who control indexed documents (for example public tickets, wiki pages, or scraped Markdown) may embed adversarial instructions (`Ignore previous instructions...`, `System override:`, role boundary tags) or hide tokens with zero width Unicode characters (`U+200B`, `U+200C`, `U+200D`, `U+FEFF`, directional overrides).

`ContextSanitizer` inspects retrieved chunks before prompt construction:

* Strips invisible zero width and ASCII control characters prior to tokenization and prompt formatting.
* Scans chunks against configurable prompt injection signatures and records `SecurityFinding` audit events.
* Drops tainted chunks (`drop_suspicious=True`) or strips offending lines before any text reaches the generator.

### 2. Cross scope and multi tenant cache isolation (`src/raglab/cache.py`)

Naive semantic caches index only the query embedding, allowing User B querying under `filters={"tenant": "acme"}` to receive a cached answer computed earlier from `filters={"tenant": "globex"}` documents. `SemanticCache` partitions entries by a canonicalized metadata filter key (`tuple(sorted(filters.items()))`), ensuring similarity lookups never cross filter scopes.

### 3. Database identifier and parameter safety (`src/raglab/integrations/pgvector.py`)

`PgVectorStore` validates table identifiers against a strict `^[A-Za-z_][A-Za-z0-9_]*$` allowlist before DDL construction and passes all document content, JSONB metadata filters, and vector literals exclusively through parameterized driver queries (`%s`).

### 4. Citation bounds and hallucination verification (`src/raglab/verification.py`)

`verify_answer` validates that every `[n]` marker falls within `1..len(sources)`, checks lexical claim support per sentence against the cited passage, and rejects responses that mix refusal phrases with unsupported claims.
