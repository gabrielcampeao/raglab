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
from raglab.retrievers import RETRIEVER_NAMES, create_retriever

PREVIEW_LENGTH = 110


def _add_index_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="directory with .md or .txt files to index")
    parser.add_argument("--k", type=int, default=3, help="number of chunks to retrieve")
    parser.add_argument("--max-chars", type=int, default=500, help="maximum characters per chunk")
    parser.add_argument(
        "--overlap", type=int, default=1, help="sentences shared by neighboring chunks"
    )


def _add_retriever_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retriever", choices=RETRIEVER_NAMES, default="hybrid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab",
        description="Explore Retrieval Augmented Generation on your own documents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="answer a question from retrieved context")
    ask.add_argument("question")
    _add_index_options(ask)
    _add_retriever_option(ask)
    ask.add_argument("--generator", choices=("extractive", "openai"), default="extractive")
    ask.add_argument("--model", default="gpt-4o-mini", help="model name for the openai generator")
    ask.add_argument("--base-url", default=None, help="base URL of an OpenAI compatible API")
    ask.add_argument(
        "--show-prompt", action="store_true", help="print the prompt sent to the model"
    )
    ask.set_defaults(handler=_run_ask)

    search = commands.add_parser("search", help="show the chunks a query retrieves")
    search.add_argument("query")
    _add_index_options(search)
    _add_retriever_option(search)
    search.set_defaults(handler=_run_search)

    evaluate = commands.add_parser("evaluate", help="compare retrievers on labeled questions")
    _add_index_options(evaluate)
    evaluate.add_argument("--cases", required=True, help="JSON file with evaluation cases")
    evaluate.set_defaults(handler=_run_evaluate)
    return parser


def _load_documents(path: str) -> list[Document]:
    documents = load_directory(path)
    if not documents:
        raise ValueError(f"No .md or .txt files found in {path}")
    return documents


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
    pipeline = RAGPipeline(
        create_retriever(retriever_name),
        generator,
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
    answer = pipeline.ask(args.question)
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
        print(f"  [{number}] {result.chunk.source} (score {result.score:.4f})")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    documents = _load_documents(args.data)
    pipeline = _build_pipeline(args, args.retriever, documents)
    results = pipeline.retrieve(args.query)
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
    print(f"Evaluated {len(cases)} questions with k={args.k}")
    print()
    print(f"{'retriever':<10} {'hit rate':>9} {'mrr':>7} {'recall':>7}")
    for name in RETRIEVER_NAMES:
        pipeline = _build_pipeline(args, name, documents)
        report = evaluate_retrieval(pipeline.retrieve, cases, args.k)
        print(f"{name:<10} {report.hit_rate:>9.2f} {report.mrr:>7.2f} {report.recall:>7.2f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError, GenerationError) as error:
        print(f"raglab: error: {error}", file=sys.stderr)
        return 1
