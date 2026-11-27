# Production RAG ecosystem and tooling guide

Building a RAG system in production means deciding which components can stay simple in process structures and which ones require dedicated infrastructure. This guide explains how modern vector databases, neural embedding models, rerankers, and serving engines work under the hood, when to pick each tool, and how to wire them into `raglab` through `src/raglab/integrations`.

![Production RAG ecosystem map](assets/ecosystem_tooling_map.svg)

## 1. Vector databases and approximate nearest neighbor engines

Linear cosine scans (`InMemoryVectorStore`) inspect every stored vector on every query (`O(N * D)`). That is fast up to roughly `50,000` vectors in a single process, after which latency and memory persistence demand an Approximate Nearest Neighbor (`ANN`) index.

Most production engines build on **HNSW (Hierarchical Navigable Small World)** graphs: layered proximity graphs where upper layers have long range skip edges for fast coarse navigation and layer 0 holds dense local neighborhoods. Query complexity drops to roughly `O(log N)` while maintaining `0.95` to `0.99` recall against exact search. `raglab` includes a transparent pure Python implementation in [hnsw.py](../src/raglab/hnsw.py) (`HNSWVectorStore`, selectable via `--backend hnsw`) so you can inspect greedy descent, beam search, heuristic neighbor pruning, and cardinality aware filter fallback directly before scaling out to Rust or C++ services.

![HNSW approximate search scaling benchmark](assets/hnsw_scaling_benchmark.svg)

### Engineering comparison across vector stores

| Engine | Storage & language | Index types | Filtered search mechanism | Best fit | Adapter in raglab |
| --- | --- | --- | --- | --- | --- |
| **raglab HNSW** | Pure Python stdlib (`in process`) | Layered HNSW (`m`, `ef_construction`, `ef_search`) | Beam prefilter with exact filtered fallback when fewer than `k` beam hits match | Teaching, prototyping, deterministic CI, and zero dependency deployments up to tens of thousands of vectors | [hnsw.py](../src/raglab/hnsw.py) (`HNSWVectorStore`) |
| **Qdrant** | Rust (`mmap` + WAL, gRPC/REST) | Filterable HNSW, sparse inverted index, scalar `int8` & binary quantization | Cardinality aware planner switching between payload index scan and filterable HNSW graph links | Dedicated production vector & hybrid search from `100k` to `100M+` vectors with heavy metadata filtering | [qdrant.py](../src/raglab/integrations/qdrant.py) (`QdrantVectorStore`) |
| **pgvector** | C extension inside PostgreSQL 16+ | HNSW (`vector_cosine_ops`), `IVFFlat` | Iterative HNSW index scans combined with native `JSONB` GIN indexes (`@>`) and relational SQL joins | Teams already running Postgres that need ACID transactional consistency between source rows and embeddings | [pgvector.py](../src/raglab/integrations/pgvector.py) (`PgVectorStore`) |
| **Chroma** | Python + Rust core over SQLite + HNSW | HNSW (`cosine`, `l2`, `ip`) | Metadata predicate bitmap applied alongside HNSW traversal | Local desktop apps, notebooks, and single node services | [chroma.py](../src/raglab/integrations/chroma.py) (`ChromaVectorStore`) |
| **Milvus** | Go + C++ (`Knowhere`) object storage architecture | HNSW, `IVF_FLAT`, `IVF_PQ`, `DiskANN`, GPU `CAGRA` | Partition keys + bitset prefiltering before ANN search | Multi hundred million to billion vector clusters needing separated compute and S3 storage | Custom `VectorStore` subclass |
| **Weaviate** | Go (`LSM` object store + HNSW) | HNSW + built in BM25 inverted index | Inverted roaring bitmaps precomputed per filter predicate | Multi tenant SaaS search with native BM25 + vector fusion | Custom `VectorStore` subclass |
| **FAISS** | C++ SIMD / CUDA library (no server) | `IndexFlatIP`, `IndexHNSWFlat`, `IndexIVFPQ` | IDSelector bitmask (no built in metadata store or WAL) | High throughput batch retrieval inside a single Python/C++ worker | Custom `VectorStore` subclass |

### Sizing memory for HNSW indexes

Before choosing an embedding dimension `D` and vector count `N`, estimate RAM footprint:

```text
Raw float32 vectors = N * D * 4 bytes
HNSW graph links    = N * M * 2 * 4 bytes   (where M = 16..32 edges per node)
```

For `10,000,000` chunks at `D = 1536` (`float32`) with `M = 16`:
* Raw vectors require `61.4 GB` of RAM.
* HNSW graph links add roughly `1.3 GB`, plus payload indexes.
* Enabling **scalar `int8` quantization** in Qdrant cuts vector memory by `4x` (`~15.4 GB`) with under `1%` recall loss; **binary quantization** on high dimensional models (`1024d` to `3072d`) cuts memory by `32x` when paired with `2x` to `4x` oversampling and float rescoring.

### Using Qdrant in raglab

`QdrantVectorStore` runs in three modes without changing a line of pipeline code:

1. **In memory (`location=":memory:"`):** Zero daemon, ideal for CI and fast integration tests.
2. **Local disk (`path="./qdrant_local"`):** Embedded persistence for single node services.
3. **Production server or Qdrant Cloud (`url="http://localhost:6333", api_key=...`):** Full Rust HNSW engine with gRPC/REST and payload indexes.

```python
from raglab import BM25Retriever, CrossTermReranker, DenseRetriever, HybridRetriever, RAGPipeline
from raglab.integrations import OpenAIEmbedder, QdrantVectorStore

qdrant_store = QdrantVectorStore(
    collection_name="atlas_pay_runbooks",
    url="http://localhost:6333",
)
embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=512)

pipeline = RAGPipeline(
    retriever=HybridRetriever([DenseRetriever(embedder, qdrant_store), BM25Retriever()]),
    reranker=CrossTermReranker(),
    mmr_lambda=0.65,
    markdown_aware=True,
)
```

### Using PostgreSQL + pgvector in raglab

When embeddings must update inside the same database transaction as operational records, `PgVectorStore` provisions the `vector` extension, creates an HNSW cosine index (`vector_cosine_ops`) alongside a `JSONB` GIN index (`jsonb_path_ops`), and pushes metadata filters down via `@>`:

```python
import psycopg
from raglab import DenseRetriever, RAGPipeline
from raglab.integrations import OllamaEmbedder, PgVectorStore

conn = psycopg.connect("postgresql://raglab:raglab_local_password@localhost:5432/raglab")
store = PgVectorStore(conn, table_name="sre_chunks", dimensions=768)
store.ensure_schema(hnsw_m=16, ef_construction=64)

pipeline = RAGPipeline(
    retriever=DenseRetriever(OllamaEmbedder(model="nomic-embed-text"), store),
    markdown_aware=True,
)
```

## 2. Neural embedding models

Lexical vectorizers (`HashingEmbedder`, `TfidfEmbedder`, `BM25Retriever`) match normalized surface tokens. Dense neural bi encoders map queries and passages into a continuous semantic space so `"shoppers charged twice"` lands near `"duplicate card captures"` even with zero shared words.

| Model family | Dimensions | Context limit | Architectural notes | Adapter |
| --- | ---: | ---: | --- | --- |
| **OpenAI `text-embedding-3-small` / `large`** | `1536` / `3072` (Matryoshka `256+`) | `8191` tokens | Trained with Matryoshka Representation Learning so truncating `1536d` to `512d` keeps most retrieval accuracy at one third the storage cost | `OpenAIEmbedder` |
| **BAAI `bge-m3`** | `1024` | `8192` tokens | Simultaneously outputs dense vectors, learned sparse lexical weights, and multi vector token representations across 100+ languages | `OpenAIEmbedder` (via vLLM / Infinity / TEI) or `OllamaEmbedder` |
| **`nomic-embed-text-v1.5`** | `768` (Matryoshka `64..768`) | `8192` tokens | Open weights, strong long context retrieval on local CPUs and GPUs | `OllamaEmbedder` |
| **FastEmbed / SentenceTransformers** | `384` to `1024` | `512` to `8192` | ONNX quantized local execution (`BAAI/bge-small-en-v1.5`) when external API calls are prohibited | Subclass `Embedder.embed` |

Even with strong neural embeddings, **keep `BM25Retriever` in a `HybridRetriever`**. Dense models compress an entire passage into one fixed vector and routinely blur exact alphanumeric identifiers (`SQLSTATE 53300`, `CVE` numbers, SKU codes, function signatures).

## 3. Second stage rerankers: bi encoders vs cross encoders

A bi encoder embeds the query and the chunk **independently**, reducing their interaction to a single dot product. A **cross encoder** concatenates `[CLS] query [SEP] chunk [SEP]` and runs full self attention across every query token and every document token simultaneously. That joint attention catches negation, argument order, and fine grained relevance, at the cost of `O(candidates)` transformer forward passes at query time.

* **Cohere Rerank 3.5 / Jina Reranker / HuggingFace TEI / vLLM `/rerank`:** Supported directly via [CohereCompatibleReranker](../src/raglab/integrations/rerankers.py). Retrieve `20` to `50` candidates with fast Qdrant + BM25 RRF (`~5 ms`), then pass only those `20` candidates to `CohereCompatibleReranker` (`~40` to `90 ms`) before `diversify_mmr`.
* **ColBERTv2 (Late Interaction):** Precomputes one small vector *per token* rather than one vector per chunk, scoring via sum of maximum cosine similarities (`MaxSim`). Trades higher storage footprint for cross encoder accuracy at near bi encoder speed.
* **`CrossTermReranker` (built into `raglab` core):** Zero network latency lexical span, coverage, and heading matcher for airgapped or sub 20 ms budgets.

```python
from raglab import HybridRetriever, RAGPipeline
from raglab.integrations import CohereCompatibleReranker

pipeline = RAGPipeline(
    retriever=HybridRetriever(),
    reranker=CohereCompatibleReranker(model="rerank-v3.5"),
    candidate_pool=25,
    mmr_lambda=0.65,
)
```

## 4. Local Docker infrastructure stack

[deploy/compose.yaml](../deploy/compose.yaml) provisions **Qdrant**, **PostgreSQL 16 + pgvector**, and **Ollama** locally:

```bash
docker compose -f deploy/compose.yaml up -d
docker exec -it $(docker compose -f deploy/compose.yaml ps -q ollama) ollama pull llama3.2
docker exec -it $(docker compose -f deploy/compose.yaml ps -q ollama) ollama pull nomic-embed-text
```

Once the containers are healthy, run the `raglab` CLI directly against the live Qdrant container and local Ollama model:

```bash
raglab ask "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?" \
  --data projects/incident_copilot/corpus \
  --backend qdrant --qdrant-url http://localhost:6333 \
  --filter kind=runbook --filter service=postgres \
  --markdown --rerank --mmr 0.65 \
  --generator openai --model llama3.2 --base-url http://localhost:11434/v1 \
  --show-trace
```
