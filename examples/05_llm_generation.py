import os
from pathlib import Path

from raglab import GenerationError, OpenAICompatibleGenerator, RAGPipeline

DATA_DIR = Path(__file__).resolve().parent / "data"
QUESTION = "Can I get my money back if I bought Plus through the App Store?"


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_BASE_URL"):
        print("No language model is configured, so this example stops here.")
        print("Set OPENAI_API_KEY to use OpenAI, or set OPENAI_BASE_URL to a local server")
        print("such as Ollama (http://localhost:11434/v1) and RAGLAB_MODEL to a model it serves.")
        return

    generator = OpenAICompatibleGenerator(model=os.environ.get("RAGLAB_MODEL", "gpt-4o-mini"))
    pipeline = RAGPipeline(generator=generator)
    pipeline.ingest_directory(DATA_DIR)

    try:
        answer = pipeline.ask(QUESTION)
    except GenerationError as error:
        print(f"The model call failed: {error}")
        return

    print(f"Q: {QUESTION}")
    print(f"A: {answer.text}")
    print()
    for number, result in enumerate(answer.sources, start=1):
        print(f"[{number}] {result.chunk.source}")


if __name__ == "__main__":
    main()
