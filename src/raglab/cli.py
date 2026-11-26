from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from raglab import __version__
from raglab.documents import load_directory
from raglab.evaluation import evaluate_retrieval, load_cases
from raglab.generation import (
    ExtractiveGenerator,
    GenerationError,
    Generator,
    OpenAICompatibleGenerator,
)
from raglab.models import Document
from raglab.pipeline import RAGPipeline
from raglab.reranking import CrossTermReranker
from raglab.retrievers import RETRIEVER_NAMES, create_retriever

PREVIEW_LENGTH = 110


def _add_index_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="directory with .md or .txt files to index")
    parser.add_argument("--k", type=int, default=3, help="number of chunks to retrieve")
    parser.add_argument("--max-chars", type=int, default=500, help="maximum characters per chunk")
    parser.add_argument(
        "--overlap", type=int, default=1, help="sentences shared by neighboring chunks"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="preserve Markdown section headings and fenced code blocks",
    )


def _add_ranking_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retriever", choices=RETRIEVER_NAMES, default="hybrid")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="apply second stage lexical cross feature reranking",
    )
    parser.add_argument(
        "--mmr",
        type=float,
        default=None,
        help="apply Maximal Marginal Relevance with lambda between 0 and 1",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="restrict retrieval to chunks matching metadata key=value",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab",
        description="Reference Retrieval Augmented Generation engine and evaluation CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="answer a question from retrieved context")
    ask.add_argument("question")
    _add_index_options(ask)
    _add_ranking_options(ask)
    ask.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="pack retrieved chunks up to a maximum context token budget",
    )
    ask.add_argument("--generator", choices=("extractive", "openai"), default="extractive")
    ask.add_argument("--model", default="gpt-4o-mini", help="model name for the openai generator")
    ask.add_argument("--base-url", default=None, help="base URL of an OpenAI compatible API")
    ask.add_argument(
        "--show-prompt", action="store_true", help="print the prompt sent to the model"
    )
    ask.add_argument(
        "--show-trace",
        action="store_true",
        help="print stage latency and grounding verification details",
    )
    ask.set_defaults(handler=_run_ask)

    search = commands.add_parser("search", help="show the chunks a query retrieves")
    search.add_argument("query")
    _add_index_options(search)
    _add_ranking_options(search)
    search.set_defaults(handler=_run_search)

    evaluate = commands.add_parser("evaluate", help="compare retrievers on labeled questions")
    _add_index_options(evaluate)
    evaluate.add_argument("--cases", required=True, help="JSON file with evaluation cases")
    evaluate.add_argument(
        "--rerank",
        action="store_true",
        help="apply second stage reranking on top of each retriever",
    )
    evaluate.set_defaults(handler=_run_evaluate)
    return parser


def _load_documents(path: str) -> list[Document]:
    documents = load_directory(path)
    if not documents:
        raise ValueError(f"No .md or .txt files found in {path}")
    return documents


def _parse_cli_filters(items: Sequence[str]) -> dict[str, str] | None:
    if not items:
        return None
    parsed: dict[str, str] = {}
    for entry in items:
        if "=" not in entry:
            raise ValueError(f"Invalid filter '{entry}'. Expected KEY=VALUE")
        key, value = entry.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _build_generator(args: argparse.Namespace) -> Generator:
    if args.generator == "openai":
        return OpenAICompatibleGenerator(args.model, base_url=args.base_url)
    return ExtractiveGenerator()


def _build_pipeline(
    args: argparse.Namespace,
    retriever_name: str,
    documents: Sequence[Document],
    generator: Generator | None = None,
) -> RAGPipeline:
    reranker = CrossTermReranker() if getattr(args, "rerank", False) else None
    pipeline = RAGPipeline(
        create_retriever(retriever_name),
        generator,
        reranker=reranker,
        mmr_lambda=getattr(args, "mmr", None),
        max_context_tokens=getattr(args, "max_tokens", None),
        markdown_aware=getattr(args, "markdown", False),
        max_chars=args.max_chars,
        overlap=args.overlap,
        top_k=args.k,
    )
    pipeline.ingest(documents)
    return pipeline


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_LENGTH else flat[: PREVIEW_LENGTH - 3] + "..."


def _run_ask(args: argparse.Namespace) -> int:
    documents = _load_documents(args.data)
    pipeline = _build_pipeline(args, args.retriever, documents, _build_generator(args))
    filters = _parse_cli_filters(args.filter)
    answer = pipeline.ask(args.question, filters=filters)
    if args.show_prompt:
        print("Prompt:")
        print(answer.prompt)
        print()
    print("Answer:")
    print(answer.text)
    print()
    print("Sources:")
    if not answer.sources:
        print("  none")
    for number, result in enumerate(answer.sources, start=1):
        section = result.chunk.metadata.get("section")
        label = f"{result.chunk.source} ({section})" if section else result.chunk.source
        print(f"  [{number}] {label} (score {result.score:.4f})")
    if args.show_trace:
        print()
        status = "verified" if answer.grounding.is_verified else "unverified"
        print(
            f"Grounding: {status} (ratio {answer.grounding.grounded_ratio:.2f}, "
            f"abstained={answer.grounding.abstained})"
        )
        print(
            f"Trace: total={answer.trace.total_ms:.2f}ms "
            f"retrieval={answer.trace.retrieval_ms:.2f}ms "
            f"rerank={answer.trace.rerank_ms:.2f}ms "
            f"tokens={answer.trace.context_tokens}"
        )
    return 0


def _run_search(args: argparse.Namespace) -> int:
    documents = _load_documents(args.data)
    pipeline = _build_pipeline(args, args.retriever, documents)
    filters = _parse_cli_filters(args.filter)
    results = pipeline.retrieve(args.query, filters=filters)
    if not results:
        print("No matching chunks.")
        return 0
    for number, result in enumerate(results, start=1):
        print(f"[{number}] {result.chunk.id}  score={result.score:.4f}")
        print(f"    {_preview(result.chunk.text)}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    documents = _load_documents(args.data)
    cases = load_cases(args.cases)
    answerable_count = sum(1 for case in cases if case.answerable)
    print(f"Evaluated {answerable_count} questions with k={args.k}")
    print()
    print(f"{'retriever':<10} {'hit rate':>9} {'mrr':>7} {'ndcg':>7} {'recall':>7}")
    for name in RETRIEVER_NAMES:
        pipeline = _build_pipeline(args, name, documents)
        report = evaluate_retrieval(pipeline.retrieve, cases, args.k)
        print(
            f"{name:<10} {report.hit_rate:>9.2f} {report.mrr:>7.2f} "
            f"{report.ndcg:>7.2f} {report.recall:>7.2f}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError, GenerationError) as error:
        print(f"raglab: error: {error}", file=sys.stderr)
        return 1
