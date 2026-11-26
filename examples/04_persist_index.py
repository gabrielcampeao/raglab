import tempfile
from pathlib import Path

from raglab import HashingEmbedder, InMemoryVectorStore, chunk_documents, load_directory

DATA_DIR = Path(__file__).resolve().parent / "data"
QUERY = "How do I enable two factor authentication?"


def main() -> None:
    chunks = chunk_documents(load_directory(DATA_DIR), max_chars=400)
    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    store.add(chunks, embedder.embed([chunk.text for chunk in chunks]))

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "index.json"
        store.save(path)
        print(f"Saved {len(store)} vectors to {path.name} ({path.stat().st_size:,} bytes)")
        restored = InMemoryVectorStore.load(path)
        print(f"Loaded {len(restored)} vectors with {restored.dimensions} dimensions each")

    print()
    print("HashingEmbedder has no learned state, so the reloaded index works without refitting.")
    print(f"Query: {QUERY}")
    print()
    for result in restored.search(embedder.embed_query(QUERY), k=2):
        preview = " ".join(result.chunk.text.split())[:90]
        print(f"{result.score:.3f}  {result.chunk.id:<22} {preview}...")


if __name__ == "__main__":
    main()
