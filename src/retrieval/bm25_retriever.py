import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def load_chunks(path: Path) -> list[dict]:
    """Load chunk records from JSONL."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.

    Lowercase the text and keep alphanumeric tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """Build a BM25 index over all chunk text."""
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def retrieve(
    query: str,
    chunks: list[dict],
    bm25: BM25Okapi,
    top_k: int = 5,
) -> list[dict]:
    """Return the top-k BM25 results for a query."""

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


def main() -> None:
    print(f"Loading chunks from: {CHUNKS_PATH}")

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded chunks: {len(chunks)}")

    print("Building BM25 index...")
    bm25 = build_bm25(chunks)

    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retrieve(
            query=query,
            chunks=chunks,
            bm25=bm25,
            top_k=5,
        )

        print("\nTop results:")
        print("=" * 80)

        for rank, result in enumerate(results, start=1):
            print(f"\nRank       : {rank}")
            print(f"Chunk ID   : {result['chunk_id']}")
            print(f"Section    : {result['section_number']}")
            print(f"Score      : {result['score']:.4f}")
            print(f"Text       : {result['text'][:500]}")


if __name__ == "__main__":
    main()