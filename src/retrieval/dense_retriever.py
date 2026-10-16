import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embeddings(path: Path) -> list[dict]:
    """Load stored chunk embeddings."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrieve(
    query: str,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """Return the top-k chunks most similar to the query."""

    # Convert the user's query into a vector.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # Convert stored embeddings into a NumPy matrix.
    chunk_embeddings = np.array(
        [chunk["embedding"] for chunk in chunks],
        dtype=np.float32,
    )

    # Because both vectors are normalized,
    # their dot product is cosine similarity.
    scores = chunk_embeddings @ query_embedding

    # Get indices of the highest-scoring chunks.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


def main() -> None:
    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")

    chunks = load_embeddings(EMBEDDINGS_PATH)
    print(f"Loaded embeddings: {len(chunks)}")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retrieve(
            query=query,
            chunks=chunks,
            model=model,
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