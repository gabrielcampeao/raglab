from pathlib import Path

from raglab import RAGPipeline

DATA_DIR = Path(__file__).resolve().parent / "data"

QUESTIONS = (
    "How long do I have to request a refund?",
    "How much does the Plus plan cost per year?",
    "How do I turn on two factor authentication?",
    "What happens when I edit the same note on two devices while offline?",
    "Can I pay with cryptocurrency?",
)


def main() -> None:
    pipeline = RAGPipeline()
    count = pipeline.ingest_directory(DATA_DIR)
    print(f"Indexed {count} chunks from {DATA_DIR.name}/")
    print()

    for question in QUESTIONS:
        answer = pipeline.ask(question)
        sources = sorted({result.chunk.source for result in answer.sources})
        print(f"Q: {question}")
        print(f"A: {answer.text}")
        print(f"   retrieved from: {', '.join(sources) or 'nothing'}")
        print()


if __name__ == "__main__":
    main()
