from pathlib import Path

from raglab import RETRIEVER_NAMES, RAGPipeline, create_retriever, evaluate_retrieval, load_cases

EXAMPLES_DIR = Path(__file__).resolve().parent
K_VALUES = (1, 3)


def main() -> None:
    cases = load_cases(EXAMPLES_DIR / "eval_cases.json")
    print(f"Comparing retrievers on {len(cases)} labeled questions")
    print()
    print(f"{'retriever':<10} {'k':>2} {'hit rate':>9} {'mrr':>6} {'recall':>7}")

    for name in RETRIEVER_NAMES:
        pipeline = RAGPipeline(create_retriever(name))
        pipeline.ingest_directory(EXAMPLES_DIR / "data")
        for k in K_VALUES:
            report = evaluate_retrieval(pipeline.retrieve, cases, k)
            scores = f"{report.hit_rate:>9.2f} {report.mrr:>6.2f} {report.recall:>7.2f}"
            print(f"{name:<10} {k:>2} {scores}")

    print()
    print("Hit rate: share of questions with at least one relevant chunk in the top k.")
    print("MRR: average of 1 / rank of the first relevant chunk.")
    print("Recall: share of the relevant documents that showed up in the top k.")


if __name__ == "__main__":
    main()
