from pathlib import Path

from raglab import (
    ExtractiveGenerator,
    HybridRetriever,
    build_prompt,
    chunk_documents,
    load_directory,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
QUESTION = "Can I get a refund if I bought Plus through the App Store?"


def heading(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def main() -> None:
    heading("Step 1. Load documents")
    documents = load_directory(DATA_DIR)
    for document in documents:
        print(f"{document.id:<24} {len(document.text):>5} characters")

    heading("Step 2. Split them into chunks")
    chunks = chunk_documents(documents, max_chars=300, overlap=1)
    print(f"{len(chunks)} chunks of at most 300 characters. The first one:")
    print()
    print(chunks[0].text)

    heading("Step 3. Index the chunks")
    retriever = HybridRetriever()
    retriever.index(chunks)
    print("Each chunk now has a hashed vector for dense search and term counts for BM25.")

    heading("Step 4. Retrieve context for the question")
    print(f"Question: {QUESTION}")
    print()
    results = retriever.retrieve(QUESTION, k=3)
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result.chunk.id:<26} fused score {result.score:.4f}")

    heading("Step 5. Build the prompt")
    prompt = build_prompt(QUESTION, results)
    print(prompt)

    heading("Step 6. Generate the answer")
    print(ExtractiveGenerator().generate(QUESTION, prompt, results))


if __name__ == "__main__":
    main()
