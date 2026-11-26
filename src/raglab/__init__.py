from raglab.chunking import (
    MarkdownSection,
    chunk_documents,
    chunk_markdown,
    chunk_markdown_documents,
    chunk_text,
    split_markdown_sections,
    split_sentences,
)
from raglab.documents import load_directory
from raglab.embeddings import Embedder, HashingEmbedder, TfidfEmbedder
from raglab.evaluation import (
    EndToEndReport,
    EvalCase,
    RetrievalReport,
    evaluate_by_category,
    evaluate_pipeline,
    evaluate_retrieval,
    load_cases,
)
from raglab.generation import (
    NO_ANSWER,
    ExtractiveGenerator,
    GenerationError,
    Generator,
    OpenAICompatibleGenerator,
)
from raglab.models import Chunk, Document, SearchResult
from raglab.pipeline import Answer, PipelineTrace, RAGPipeline
from raglab.prompting import (
    DEFAULT_TEMPLATE,
    build_prompt,
    estimate_tokens,
    format_context,
    pack_context,
)
from raglab.reranking import CrossTermReranker, Reranker, diversify_mmr
from raglab.retrievers import (
    RETRIEVER_NAMES,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MetadataFilter,
    Retriever,
    create_retriever,
    matches_filters,
)
from raglab.vectorstore import InMemoryVectorStore
from raglab.verification import GroundingReport, verify_answer

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_TEMPLATE",
    "NO_ANSWER",
    "RETRIEVER_NAMES",
    "Answer",
    "BM25Retriever",
    "Chunk",
    "CrossTermReranker",
    "DenseRetriever",
    "Document",
    "Embedder",
    "EndToEndReport",
    "EvalCase",
    "ExtractiveGenerator",
    "GenerationError",
    "Generator",
    "GroundingReport",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryVectorStore",
    "MarkdownSection",
    "MetadataFilter",
    "OpenAICompatibleGenerator",
    "PipelineTrace",
    "RAGPipeline",
    "Reranker",
    "RetrievalReport",
    "Retriever",
    "SearchResult",
    "TfidfEmbedder",
    "__version__",
    "build_prompt",
    "chunk_documents",
    "chunk_markdown",
    "chunk_markdown_documents",
    "chunk_text",
    "create_retriever",
    "diversify_mmr",
    "estimate_tokens",
    "evaluate_by_category",
    "evaluate_pipeline",
    "evaluate_retrieval",
    "format_context",
    "load_cases",
    "load_directory",
    "matches_filters",
    "pack_context",
    "split_markdown_sections",
    "split_sentences",
    "verify_answer",
]
